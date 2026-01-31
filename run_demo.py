#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
run_demo.py - CUDA IPC RL Demo 入口点

这个模块是 `uv run run_demo.py` 命令的入口点。
它会自动设置 CUDA 库路径并启动 Env 和 Policy 两个进程。
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path


def setup_cuda_library_path():
    """
    设置 CUDA 库路径
    
    CuPy 需要在运行时加载 NVIDIA CUDA 库（如 libnvrtc.so.12）。
    这些库由 nvidia-cuda-nvrtc-cu12 等 pip 包提供，安装在 site-packages 中。
    我们需要将它们的路径添加到 LD_LIBRARY_PATH 环境变量。
    """
    import site
    
    # 查找 nvidia 包的安装位置
    nvidia_base = None
    search_paths = site.getsitepackages() + [site.getusersitepackages()]
    
    for sp in search_paths:
        if sp is None:
            continue
        candidate = Path(sp) / "nvidia"
        if candidate.exists() and candidate.is_dir():
            nvidia_base = candidate
            break
    
    if nvidia_base is None:
        print("警告: 未找到 nvidia CUDA 库目录")
        return
    
    # CUDA 库子目录
    cuda_lib_dirs = [
        "cuda_nvrtc/lib",
        "cuda_runtime/lib", 
        "curand/lib",
        "cublas/lib",
        "cufft/lib",
        "cusolver/lib",
        "cusparse/lib",
        "nvjitlink/lib",
    ]
    
    lib_paths = []
    for subdir in cuda_lib_dirs:
        lib_path = nvidia_base / subdir
        if lib_path.exists():
            lib_paths.append(str(lib_path))
    
    if lib_paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new_path = ":".join(lib_paths)
        if existing:
            new_path = f"{new_path}:{existing}"
        os.environ["LD_LIBRARY_PATH"] = new_path
        print(f"[Setup] 已设置 CUDA 库路径 ({len(lib_paths)} 个目录)")


def cleanup_socket():
    """清理残留的 Unix Domain Socket"""
    socket_path = "/tmp/rl_loop_demo.sock"
    if os.path.exists(socket_path):
        os.remove(socket_path)


def main():
    """
    主入口函数
    
    启动顺序:
    1. 设置 CUDA 库路径
    2. 清理残留 socket
    3. 启动 Env 进程（后台）
    4. 等待 socket 创建
    5. 启动 Policy 进程（前台）
    6. 等待两个进程完成
    """
    print("=" * 50)
    print("  CUDA IPC 跨进程 GPU 显存共享 RL Demo")
    print("=" * 50)
    print()
    
    # 1. 设置 CUDA 库路径
    setup_cuda_library_path()
    
    # 2. 清理残留 socket
    cleanup_socket()
    
    # 获取脚本目录
    script_dir = Path(__file__).parent.resolve()
    
    # 准备环境变量（继承当前环境，包括 LD_LIBRARY_PATH）
    env = os.environ.copy()
    
    print("[Demo] 自动启动 Env 和 Policy 进程...")
    print()
    
    env_process = None
    policy_process = None
    
    try:
        # 3. 启动 Env 进程（后台）
        env_process = subprocess.Popen(
            [sys.executable, str(script_dir / "env_process.py")],
            cwd=str(script_dir),
            env=env,
        )
        print(f"[Demo] Env 进程已启动 (PID: {env_process.pid})")
        
        # 4. 等待 socket 创建
        socket_path = "/tmp/rl_loop_demo.sock"
        for _ in range(50):  # 最多等待 5 秒
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        else:
            print("[Demo] 警告: 等待 socket 创建超时，继续启动 Policy...")
        
        # 5. 启动 Policy 进程（前台）
        policy_process = subprocess.Popen(
            [sys.executable, str(script_dir / "policy_process.py")],
            cwd=str(script_dir),
            env=env,
        )
        print(f"[Demo] Policy 进程已启动 (PID: {policy_process.pid})")
        print()
        
        # 6. 等待两个进程完成
        policy_process.wait()
        env_process.wait()
        
        print()
        print("[Demo] 演示完成!")
        
    except KeyboardInterrupt:
        print("\n[Demo] 收到中断信号，正在清理...")
        
    finally:
        # 清理进程
        for proc in [env_process, policy_process]:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        # 清理 socket
        cleanup_socket()


if __name__ == "__main__":
    main()
