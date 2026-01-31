# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
policy_process.py - 策略进程

负责:
1. 接收 CUDA IPC 句柄
2. 导入共享 GPU 缓冲区
3. 运行神经网络推理 (模拟)
4. 与 Env 进程同步
"""

import time
import numpy as np

try:
    import cupy as cp
    from cupy.cuda import runtime as cuda_runtime
except ImportError:
    print("错误: 请安装 CuPy (pip install cupy-cuda12x)")
    exit(1)

from shared_types import (
    LAYOUT, METADATA, NUM_ENVS, STATE_DIM, ACTION_DIM, 
    DTYPE, MAX_STEPS
)
from ipc_utils import receive_ipc_handle


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
    
    print(f"  [{process}] {name:8s} | 虚拟地址: {format_addr(virt_addr)} | "
          f"偏移: +{offset:5d} | 大小: {size:5d} bytes")


class PolicyProcess:
    """策略进程：管理神经网络推理"""
    
    def __init__(self):
        self.buffer_ptr = None
        self.state_gpu = None
        self.action_gpu = None
        self.reward_gpu = None
        self.metadata_gpu = None
        self.imported_ptr = None
        self.base_ptr = None  # IPC 导入的基地址
    
    def import_shared_buffer(self, handle: bytes, total_size: int):
        """从 IPC 句柄导入共享缓冲区"""
        print(f"[Policy] 导入共享缓冲区: {total_size} bytes")
        
        # 打开 IPC 句柄
        self.imported_ptr = cuda_runtime.ipcOpenMemHandle(handle)
        self.base_ptr = self.imported_ptr  # 导入的基地址
        
        print(f"[Policy] IPC 导入的基地址: {format_addr(self.base_ptr)}")
        print(f"  注: Policy 进程的虚拟地址与 Env 进程不同，但映射到相同的 GPU 物理地址")
        
        # 创建内存指针包装
        mem = cp.cuda.UnownedMemory(self.imported_ptr, total_size, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        
        # 创建各区域的视图
        self.metadata_gpu = cp.ndarray(
            shape=(LAYOUT.metadata_size // 4,),
            dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.metadata_offset)
        )
        
        self.state_gpu = cp.ndarray(
            shape=(NUM_ENVS, STATE_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.state_offset)
        )
        
        self.action_gpu = cp.ndarray(
            shape=(NUM_ENVS, ACTION_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.action_offset)
        )
        
        self.reward_gpu = cp.ndarray(
            shape=(NUM_ENVS,),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(mem, LAYOUT.reward_offset)
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
            
            time.sleep(0.0001)  # 100μs 轮询
    
    def signal_policy_ready(self):
        """通知 Env 进程 Policy 已完成"""
        self.metadata_gpu[METADATA.policy_ready // 4] = 1
        cp.cuda.Stream.null.synchronize()
    
    def policy_inference(self, step: int):
        """
        执行策略推理 (模拟)
        
        真实场景中这里会运行 PyTorch/JAX 神经网络推理。
        这里用简单的线性策略模拟。
        """
        show_addr = (step % 10 == 0)  # 每 10 步打印一次地址信息
        
        # 读取 State (由 Env 写入)
        state_ptr = self.state_gpu.data.ptr
        if show_addr:
            print(f"[Policy] Step {step:3d} 读取 State:  "
                  f"虚拟地址={format_addr(state_ptr)} "
                  f"(基址+{state_ptr - self.base_ptr})")
        
        state = self.state_gpu
        
        # 模拟神经网络推理: Action = -0.1 * State[:, :ACTION_DIM]
        # 真实场景: action = model(state)
        action_ptr = self.action_gpu.data.ptr
        if show_addr:
            print(f"[Policy] Step {step:3d} 写入 Action: "
                  f"虚拟地址={format_addr(action_ptr)} "
                  f"(基址+{action_ptr - self.base_ptr})")
        
        self.action_gpu[:] = -0.1 * state[:, :ACTION_DIM]
        self.action_gpu[:] = cp.clip(self.action_gpu, -1.0, 1.0)
        
        cp.cuda.Stream.null.synchronize()
    
    def run_loop(self):
        """运行策略推理循环"""
        print("[Policy] 开始策略推理循环")
        
        step = 0
        while True:
            # 1. 等待 Env 完成
            should_continue = self.wait_for_env()
            if not should_continue:
                break
            
            # 2. 执行策略推理
            self.policy_inference(step)
            
            # 3. 通知 Env 进程
            self.signal_policy_ready()
            
            if step % 10 == 0:
                current_step = int(self.metadata_gpu[METADATA.step_count // 4])
                avg_action = float(cp.mean(cp.abs(self.action_gpu)))
                print(f"[Policy] Step {current_step:3d} | Avg |Action|: {avg_action:.4f}")
                print("")
            
            step += 1
        
        print("[Policy] 策略推理循环结束")
    
    def cleanup(self):
        """清理资源"""
        if self.imported_ptr is not None:
            cuda_runtime.ipcCloseMemHandle(self.imported_ptr)
            print("[Policy] 已关闭 IPC 句柄")


def main():
    policy = PolicyProcess()
    
    try:
        # 1. 接收 IPC 句柄
        handle, total_size, info = receive_ipc_handle()
        print(f"[Policy] 缓冲区信息: {info}")
        
        # 2. 导入共享缓冲区
        policy.import_shared_buffer(handle, total_size)
        
        # 3. 运行策略推理循环
        policy.run_loop()
        
    finally:
        policy.cleanup()
    
    print("[Policy] 进程退出")


if __name__ == "__main__":
    main()
