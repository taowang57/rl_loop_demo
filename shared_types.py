# SPDX-License-Identifier: MIT
# Copyright (c) 2025
"""
shared_types.py - 共享数据结构定义

定义 Policy 进程和 Env 进程之间共享的数据结构布局。
"""

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple


# ============================================================
# 配置常量
# ============================================================

NUM_ENVS = 64          # 并行环境数量
STATE_DIM = 12         # 状态向量维度 (如机器人的 12 个关节角度)
ACTION_DIM = 6         # 动作向量维度 (如 6 个电机力矩)
DTYPE = np.float32     # 数据类型


# ============================================================
# 内存布局计算
# ============================================================

@dataclass
class BufferLayout:
    """共享缓冲区内存布局"""
    
    # 元数据区 (256 字节对齐)
    metadata_offset: int = 0
    metadata_size: int = 256
    
    # State 缓冲区
    state_offset: int = 256
    state_size: int = NUM_ENVS * STATE_DIM * np.dtype(DTYPE).itemsize
    
    # Action 缓冲区
    @property
    def action_offset(self) -> int:
        return self.state_offset + self.state_size
    
    @property
    def action_size(self) -> int:
        return NUM_ENVS * ACTION_DIM * np.dtype(DTYPE).itemsize
    
    # Reward 缓冲区
    @property
    def reward_offset(self) -> int:
        return self.action_offset + self.action_size
    
    @property
    def reward_size(self) -> int:
        return NUM_ENVS * np.dtype(DTYPE).itemsize
    
    # 总大小
    @property
    def total_size(self) -> int:
        return self.reward_offset + self.reward_size


# 全局布局实例
LAYOUT = BufferLayout()


# ============================================================
# 元数据结构 (存储在共享缓冲区开头)
# ============================================================

class MetadataFields(NamedTuple):
    """元数据字段偏移量 (相对于 metadata_offset)"""
    state_dim: int = 0          # int32: 状态维度
    action_dim: int = 4         # int32: 动作维度
    num_envs: int = 8           # int32: 环境数量
    step_count: int = 12        # int32: 当前步数
    env_ready: int = 16         # int32: Env 完成标志 (0/1)
    policy_ready: int = 20      # int32: Policy 完成标志 (0/1)
    done: int = 24              # int32: 结束标志 (0/1)


METADATA = MetadataFields()


# ============================================================
# IPC 通信配置
# ============================================================

IPC_SOCKET_PATH = "/tmp/rl_loop_demo.sock"
MAX_STEPS = 100
