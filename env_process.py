# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
env_process.py - 环境进程

负责:
1. 分配 GPU 共享缓冲区
2. 导出 CUDA IPC 句柄
3. 运行物理仿真 (模拟)
4. 与 Policy 进程同步
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
from ipc_utils import send_ipc_handle, cleanup_socket


def format_addr(ptr: int) -> str:
    """格式化地址为十六进制字符串"""
    return f"0x{ptr:016X}"


def print_memory_region_info(name: str, array, base_ptr: int, process: str = "Env"):
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


class EnvProcess:
    """环境进程：管理物理仿真和共享缓冲区"""
    
    def __init__(self):
        self.buffer_ptr = None
        self.state_gpu = None
        self.action_gpu = None
        self.reward_gpu = None
        self.metadata_gpu = None
    
    def allocate_shared_buffer(self):
        """分配 GPU 共享缓冲区"""
        print(f"[Env] 分配 GPU 共享缓冲区: {LAYOUT.total_size} bytes")
        
        # 分配连续 GPU 内存
        self.buffer_ptr = cp.cuda.alloc(LAYOUT.total_size)
        self.base_ptr = self.buffer_ptr.ptr
        
        print(f"[Env] GPU 共享缓冲区基地址: {format_addr(self.base_ptr)}")
        print(f"  注: GPU 物理地址由 NVIDIA 驱动管理，用户空间只能访问设备虚拟地址")
        print(f"  两个进程通过 CUDA IPC 句柄映射到相同的 GPU 物理显存区域")
        
        # 创建各区域的视图
        self.metadata_gpu = cp.ndarray(
            shape=(LAYOUT.metadata_size // 4,),
            dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(self.buffer_ptr.mem, LAYOUT.metadata_offset)
        )
        
        self.state_gpu = cp.ndarray(
            shape=(NUM_ENVS, STATE_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(self.buffer_ptr.mem, LAYOUT.state_offset)
        )
        
        self.action_gpu = cp.ndarray(
            shape=(NUM_ENVS, ACTION_DIM),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(self.buffer_ptr.mem, LAYOUT.action_offset)
        )
        
        self.reward_gpu = cp.ndarray(
            shape=(NUM_ENVS,),
            dtype=DTYPE,
            memptr=cp.cuda.MemoryPointer(self.buffer_ptr.mem, LAYOUT.reward_offset)
        )
        
        # 初始化元数据
        self.metadata_gpu[METADATA.state_dim // 4] = STATE_DIM
        self.metadata_gpu[METADATA.action_dim // 4] = ACTION_DIM
        self.metadata_gpu[METADATA.num_envs // 4] = NUM_ENVS
        self.metadata_gpu[METADATA.step_count // 4] = 0
        self.metadata_gpu[METADATA.env_ready // 4] = 0
        self.metadata_gpu[METADATA.policy_ready // 4] = 0
        self.metadata_gpu[METADATA.done // 4] = 0
        
        # 初始化 State (随机初始状态)
        self.state_gpu[:] = cp.random.randn(NUM_ENVS, STATE_DIM).astype(DTYPE)
        
        cp.cuda.Stream.null.synchronize()
        
        # 打印各区域地址信息
        print("[Env] GPU 缓冲区初始化完成，内存布局:")
        print_memory_region_info("Metadata", self.metadata_gpu, self.base_ptr)
        print_memory_region_info("State", self.state_gpu, self.base_ptr)
        print_memory_region_info("Action", self.action_gpu, self.base_ptr)
        print_memory_region_info("Reward", self.reward_gpu, self.base_ptr)
    
    def get_ipc_handle(self) -> bytes:
        """获取 CUDA IPC 句柄"""
        handle = cuda_runtime.ipcGetMemHandle(self.buffer_ptr.ptr)
        print("[Env] 已生成 IPC 句柄")
        return bytes(handle)
    
    def wait_for_policy(self):
        """等待 Policy 进程完成"""
        while True:
            flag = int(self.metadata_gpu[METADATA.policy_ready // 4])
            if flag == 1:
                # 重置标志
                self.metadata_gpu[METADATA.policy_ready // 4] = 0
                cp.cuda.Stream.null.synchronize()
                return
            time.sleep(0.0001)  # 100μs 轮询
    
    def signal_env_ready(self):
        """通知 Policy 进程 Env 已完成"""
        self.metadata_gpu[METADATA.env_ready // 4] = 1
        cp.cuda.Stream.null.synchronize()
    
    def env_step(self, step: int):
        """
        执行一步物理仿真 (模拟)
        
        真实场景中这里会调用 MJX/MuJoCo 的物理仿真 Kernel。
        这里用简单的数学运算模拟。
        """
        show_addr = (step % 10 == 0)  # 每 10 步打印一次地址信息
        
        # 读取 Action (由 Policy 写入)
        action_ptr = self.action_gpu.data.ptr
        if show_addr:
            print(f"[Env] Step {step:3d} 读取 Action: "
                  f"虚拟地址={format_addr(action_ptr)} "
                  f"(基址+{action_ptr - self.base_ptr})")
        
        action = self.action_gpu
        
        # 模拟物理仿真: State' = State + Action (简化的线性动力学)
        # 真实场景: new_state = mjx.step(model, state, action)
        state_ptr = self.state_gpu.data.ptr
        if show_addr:
            print(f"[Env] Step {step:3d} 写入 State:  "
                  f"虚拟地址={format_addr(state_ptr)} "
                  f"(基址+{state_ptr - self.base_ptr})")
        
        self.state_gpu[:, :ACTION_DIM] += action * 0.1
        self.state_gpu[:, :ACTION_DIM] = cp.clip(self.state_gpu[:, :ACTION_DIM], -10, 10)
        
        # 计算 Reward (模拟)
        reward_ptr = self.reward_gpu.data.ptr
        if show_addr:
            print(f"[Env] Step {step:3d} 写入 Reward: "
                  f"虚拟地址={format_addr(reward_ptr)} "
                  f"(基址+{reward_ptr - self.base_ptr})")
        
        self.reward_gpu[:] = -cp.sum(cp.abs(self.state_gpu), axis=1)
        
        # 更新步数
        self.metadata_gpu[METADATA.step_count // 4] = step
        
        cp.cuda.Stream.null.synchronize()
    
    def run_loop(self):
        """运行 RL 循环"""
        print(f"[Env] 开始 RL 循环 (共 {MAX_STEPS} 步)")
        
        for step in range(MAX_STEPS):
            # 1. 执行物理仿真
            self.env_step(step)
            
            # 2. 通知 Policy 进程
            self.signal_env_ready()
            
            # 3. 等待 Policy 完成
            self.wait_for_policy()
            
            if step % 10 == 0:
                avg_reward = float(cp.mean(self.reward_gpu))
                print(f"[Env] Step {step:3d} | Avg Reward: {avg_reward:.4f}")
                print("")
        
        # 设置结束标志
        self.metadata_gpu[METADATA.done // 4] = 1
        self.signal_env_ready()
        
        print("[Env] RL 循环结束")


def main():
    cleanup_socket()
    
    env = EnvProcess()
    
    # 1. 分配共享缓冲区
    env.allocate_shared_buffer()
    
    # 2. 获取并发送 IPC 句柄
    handle = env.get_ipc_handle()
    buffer_info = {
        "num_envs": NUM_ENVS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM
    }
    send_ipc_handle(handle, buffer_info)
    
    # 3. 运行 RL 循环
    env.run_loop()
    
    print("[Env] 进程退出")


if __name__ == "__main__":
    main()
