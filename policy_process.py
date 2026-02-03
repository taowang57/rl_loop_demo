# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
policy_process.py - 策略进程 (PyTorch 神经网络版本)

负责:
1. 接收 CUDA IPC 句柄
2. 导入共享 GPU 缓冲区
3. 使用 PyTorch MLP 进行神经网络推理
4. 通过 DLPack 实现 CuPy ↔ PyTorch 零拷贝转换
5. 与 Env 进程同步
"""

try:
    import cupy as cp
    from cupy.cuda import runtime as cuda_runtime
except ImportError:
    print("错误: 请安装 CuPy (pip install cupy-cuda12x)")
    exit(1)

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("错误: 请安装 PyTorch (pip install torch)")
    exit(1)

from ipc_utils import receive_ipc_handle
from shared_types import (
    ACTION_DIM,
    DTYPE,
    LAYOUT,
    METADATA,
    NUM_ENVS,
    STATE_DIM,
)


def format_addr(ptr: int) -> str:
    """格式化地址为十六进制字符串"""
    return f"0x{ptr:016X}"


def print_memory_region_info(name: str, array, base_ptr: int, process: str = "Policy"):
    """
    打印内存区域的地址信息

    Args:
        name: 区域名称 (State/Action/Reward)
        array: CuPy 数组
        base_ptr: IPC 共享缓冲区基地址
        process: 进程名称
    """
    virt_addr = array.data.ptr  # 设备虚拟地址
    offset = virt_addr - base_ptr
    size = array.nbytes

    print(
        f"  [{process}] {name:8s} | 虚拟地址: {format_addr(virt_addr)} | "
        f"偏移: +{offset:5d} | 大小: {size:5d} bytes"
    )


class SimpleMLP(nn.Module):
    """
    简单的多层感知器 (MLP) 策略网络

    架构: State → Linear(64) → ReLU → Linear(64) → ReLU → Linear(Action) → Tanh

    输出范围 [-1, 1]，适用于连续动作空间。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        # 初始化权重 (Xavier 初始化)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyProcess:
    """策略进程：管理神经网络推理（使用 PyTorch）"""

    def __init__(self):
        self.buffer_ptr = None
        self.state_gpu = None
        self.action_gpu = None
        self.reward_gpu = None
        self.metadata_gpu = None
        self.imported_ptr = None
        self.base_ptr = None  # IPC 导入的基地址

        # PyTorch 模型
        self.model = None
        self.device = None

    def init_pytorch_model(self):
        """初始化 PyTorch 神经网络模型"""
        # 确保 PyTorch 使用与 CuPy 相同的 CUDA 设备
        self.device = torch.device("cuda:0")

        # 创建 MLP 策略网络
        self.model = SimpleMLP(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=64
        ).to(self.device)

        # 设置为评估模式 (禁用 dropout 等)
        self.model.eval()

        # 统计模型参数量
        num_params = sum(p.numel() for p in self.model.parameters())

        print("[Policy] PyTorch MLP 初始化完成:")
        print(f"  设备: {self.device}")
        print(f"  架构: {STATE_DIM} → 64 → 64 → {ACTION_DIM}")
        print(f"  参数量: {num_params:,}")
        print("  输出激活: Tanh (范围 [-1, 1])")

    def import_shared_buffer(self, handle: bytes, total_size: int):
        """从 IPC 句柄导入共享缓冲区"""
        print(f"[Policy] 导入共享缓冲区: {total_size} bytes")

        # 打开 IPC 句柄
        self.imported_ptr = cuda_runtime.ipcOpenMemHandle(handle)
        self.base_ptr = self.imported_ptr  # 导入的基地址

        print(f"[Policy] IPC 导入的基地址: {format_addr(self.base_ptr)}")
        print("  注: Policy 进程的虚拟地址与 Env 进程不同，但映射到相同的 GPU 物理地址")

        # 创建内存指针包装
        mem = cp.cuda.UnownedMemory(self.imported_ptr, total_size, owner=None)

        # 创建各区域的视图
        self.metadata_gpu = cp.ndarray(
            shape=(LAYOUT.metadata_size // 4,),
            dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.metadata_offset),
        )

        self.state_gpu = cp.ndarray(
            shape=(NUM_ENVS, STATE_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.state_offset),
        )

        self.action_gpu = cp.ndarray(
            shape=(NUM_ENVS, ACTION_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.action_offset),
        )

        self.reward_gpu = cp.ndarray(
            shape=(NUM_ENVS,),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.reward_offset),
        )

        cp.cuda.Stream.null.synchronize()

        # 打印各区域地址信息
        print("[Policy] 共享缓冲区导入完成，内存布局:")
        print_memory_region_info("Metadata", self.metadata_gpu, self.base_ptr)
        print_memory_region_info("State", self.state_gpu, self.base_ptr)
        print_memory_region_info("Action", self.action_gpu, self.base_ptr)
        print_memory_region_info("Reward", self.reward_gpu, self.base_ptr)

    def wait_for_env(self) -> bool:
        """
        等待 Env 进程完成

        Returns:
            bool: True 表示继续循环，False 表示应该退出
        """
        while True:
            env_ready = int(self.metadata_gpu[METADATA.env_ready // 4])
            done = int(self.metadata_gpu[METADATA.done // 4])

            if done == 1:
                return False

            if env_ready == 1:
                # 重置标志
                self.metadata_gpu[METADATA.env_ready // 4] = 0
                cp.cuda.Stream.null.synchronize()
                return True

            # Busy wait for low latency (< 5μs)
            pass

    def signal_policy_ready(self):
        """通知 Env 进程 Policy 已完成"""
        self.metadata_gpu[METADATA.policy_ready // 4] = 1
        cp.cuda.Stream.null.synchronize()

    def policy_inference(self, step: int):
        """
        执行策略推理（使用 PyTorch 神经网络）

        数据流:
        1. CuPy 数组 (IPC 共享) → DLPack → PyTorch Tensor (零拷贝)
        2. PyTorch 神经网络推理
        3. PyTorch Tensor → DLPack → CuPy 数组 (零拷贝写回)
        """
        show_addr = step % 10 == 0  # 每 10 步打印一次详细信息

        # ============================================================
        # Step 1: CuPy → PyTorch (零拷贝，通过 DLPack)
        # ============================================================
        state_cupy_ptr = self.state_gpu.data.ptr

        # 使用 DLPack 协议进行零拷贝转换
        # CuPy array → DLPack capsule → PyTorch tensor
        state_torch = torch.from_dlpack(self.state_gpu)

        if show_addr:
            state_torch_ptr = state_torch.data_ptr()
            print(f"[Policy] Step {step:3d} 零拷贝转换验证:")
            print(f"         CuPy  State 地址: {format_addr(state_cupy_ptr)}")
            print(f"         Torch State 地址: {format_addr(state_torch_ptr)}")
            print(
                f"         地址相同: {state_cupy_ptr == state_torch_ptr} ✓"
                if state_cupy_ptr == state_torch_ptr
                else "         地址不同: ✗"
            )

        # ============================================================
        # Step 2: PyTorch 神经网络推理
        # ============================================================
        with torch.no_grad():
            action_torch = self.model(state_torch)

        if show_addr:
            print(
                f"         MLP 推理完成: [{NUM_ENVS}, {STATE_DIM}] → [{NUM_ENVS}, {ACTION_DIM}]"
            )

        # ============================================================
        # Step 3: PyTorch → CuPy (零拷贝写回)
        # ============================================================
        action_cupy_ptr = self.action_gpu.data.ptr

        # 方法: 将 PyTorch 输出转换为 CuPy，然后复制到共享缓冲区
        # 注意: action_torch 是新分配的内存，需要复制到 IPC 共享区域
        action_from_torch = cp.from_dlpack(action_torch)
        self.action_gpu[:] = action_from_torch

        if show_addr:
            action_torch_ptr = action_torch.data_ptr()
            print(
                f"         Torch Action 地址: {format_addr(action_torch_ptr)} (MLP 输出)"
            )
            print(
                f"         CuPy  Action 地址: {format_addr(action_cupy_ptr)} (IPC 共享)"
            )
            print("         注: MLP 输出是新内存，需复制到 IPC 共享区域")

        # 确保 GPU 操作完成
        cp.cuda.Stream.null.synchronize()

    def run_loop(self):
        """运行策略推理循环"""
        print("[Policy] 开始 PyTorch 策略推理循环")
        print()

        step = 0
        while True:
            # 1. 等待 Env 完成
            should_continue = self.wait_for_env()
            if not should_continue:
                break

            # 2. 执行策略推理 (PyTorch MLP)
            self.policy_inference(step)

            # 3. 通知 Env 进程
            self.signal_policy_ready()

            if step % 10 == 0:
                current_step = int(self.metadata_gpu[METADATA.step_count // 4])
                avg_action = float(cp.mean(cp.abs(self.action_gpu)))
                print(
                    f"[Policy] Step {current_step:3d} | Avg |Action|: {avg_action:.4f}"
                )
                print()

            step += 1

        print("[Policy] PyTorch 策略推理循环结束")

    def cleanup(self):
        """清理资源"""
        if self.imported_ptr is not None:
            cuda_runtime.ipcCloseMemHandle(self.imported_ptr)
            print("[Policy] 已关闭 IPC 句柄")


def main():
    policy = PolicyProcess()

    try:
        # 1. 初始化 PyTorch 模型
        policy.init_pytorch_model()
        print()

        # 2. 接收 IPC 句柄
        handle, total_size, info = receive_ipc_handle()
        print(f"[Policy] 缓冲区信息: {info}")

        # 3. 导入共享缓冲区
        policy.import_shared_buffer(handle, total_size)
        print()

        # 4. 运行策略推理循环
        policy.run_loop()

    finally:
        policy.cleanup()

    print("[Policy] 进程退出")


if __name__ == "__main__":
    main()
