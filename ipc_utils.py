# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
ipc_utils.py - CUDA IPC 句柄传递工具

通过 Unix Domain Socket 在进程间传递 CUDA IPC 句柄。
使用 raw bytes 协议。

数据格式 (固定 80 字节):
  - handle: 64 bytes (CUDA IPC handle)
  - total_size: 4 bytes (uint32, little-endian)
  - num_envs: 4 bytes (uint32, little-endian)
  - state_dim: 4 bytes (uint32, little-endian)
  - action_dim: 4 bytes (uint32, little-endian)
"""

import os
import socket
import struct
from shared_types import IPC_SOCKET_PATH, LAYOUT

# CUDA IPC 句柄固定大小
CUDA_IPC_HANDLE_SIZE = 64

# 数据包格式: 64 bytes handle + 4 个 uint32
PAYLOAD_FORMAT = f'{CUDA_IPC_HANDLE_SIZE}s4I'
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)  # 80 bytes


def cleanup_socket():
    """清理残留的 socket 文件"""
    if os.path.exists(IPC_SOCKET_PATH):
        os.remove(IPC_SOCKET_PATH)


def pack_ipc_payload(handle: bytes, total_size: int, 
                     num_envs: int, state_dim: int, action_dim: int) -> bytes:
    """
    打包 IPC 数据为 raw bytes
    
    Args:
        handle: CUDA IPC 句柄 (64 bytes)
        total_size: 缓冲区总大小
        num_envs: 环境数量
        state_dim: 状态维度
        action_dim: 动作维度
    
    Returns:
        bytes: 打包后的数据 (80 bytes)
    """
    if len(handle) != CUDA_IPC_HANDLE_SIZE:
        raise ValueError(f"IPC handle 大小错误: 期望 {CUDA_IPC_HANDLE_SIZE}, 实际 {len(handle)}")
    
    return struct.pack(PAYLOAD_FORMAT, handle, total_size, num_envs, state_dim, action_dim)


def unpack_ipc_payload(data: bytes) -> tuple:
    """
    解包 IPC 数据
    
    Args:
        data: 打包的数据 (80 bytes)
    
    Returns:
        tuple: (handle, total_size, num_envs, state_dim, action_dim)
    """
    if len(data) != PAYLOAD_SIZE:
        raise ValueError(f"数据大小错误: 期望 {PAYLOAD_SIZE}, 实际 {len(data)}")
    
    handle, total_size, num_envs, state_dim, action_dim = struct.unpack(PAYLOAD_FORMAT, data)
    return handle, total_size, num_envs, state_dim, action_dim


def send_ipc_handle(handle: bytes, buffer_info: dict) -> None:
    """
    Env 进程: 发送 IPC 句柄给 Policy 进程
    
    Args:
        handle: CUDA IPC 内存句柄 (bytes)
        buffer_info: 缓冲区信息 {"num_envs", "state_dim", "action_dim"}
    """
    cleanup_socket()
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(IPC_SOCKET_PATH)
    sock.listen(1)
    
    print(f"[Env] 等待 Policy 进程连接... ({IPC_SOCKET_PATH})")
    conn, _ = sock.accept()
    print("[Env] Policy 进程已连接")
    
    # 打包并发送数据 (固定 80 字节，无需发送长度)
    data = pack_ipc_payload(
        handle=handle,
        total_size=LAYOUT.total_size,
        num_envs=buffer_info["num_envs"],
        state_dim=buffer_info["state_dim"],
        action_dim=buffer_info["action_dim"]
    )
    conn.sendall(data)
    
    # 等待 ACK
    ack = conn.recv(3)
    if ack == b"ACK":
        print("[Env] Policy 进程已确认接收句柄")
    
    conn.close()
    sock.close()


def receive_ipc_handle() -> tuple:
    """
    Policy 进程: 接收 IPC 句柄
    
    Returns:
        tuple: (handle, total_size, buffer_info_dict)
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    
    print(f"[Policy] 连接 Env 进程... ({IPC_SOCKET_PATH})")
    sock.connect(IPC_SOCKET_PATH)
    print("[Policy] 已连接到 Env 进程")
    
    # 接收固定大小的数据
    data = b""
    while len(data) < PAYLOAD_SIZE:
        chunk = sock.recv(PAYLOAD_SIZE - len(data))
        if not chunk:
            raise RuntimeError("连接中断")
        data += chunk
    
    # 解包数据
    handle, total_size, num_envs, state_dim, action_dim = unpack_ipc_payload(data)
    
    # 发送 ACK
    sock.sendall(b"ACK")
    sock.close()
    
    print("[Policy] 已接收 IPC 句柄")
    
    # 返回与原接口兼容的格式
    buffer_info = {
        "num_envs": num_envs,
        "state_dim": state_dim,
        "action_dim": action_dim
    }
    return handle, total_size, buffer_info
