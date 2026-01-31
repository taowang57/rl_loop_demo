#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025
# run_demo.sh - 启动 CUDA IPC RL Demo

set -e

echo "=========================================="
echo "  CUDA IPC 跨进程 GPU 显存共享 RL Demo"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 确保虚拟环境存在，如果不存在则使用 uv 创建
if [ ! -d ".venv" ]; then
    echo "[Setup] 创建虚拟环境并安装依赖..."
    uv sync
fi

# 获取 Python 解释器路径
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# 动态检测 Python 版本并设置 CUDA 库路径
PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
VENV_SITE="$SCRIPT_DIR/.venv/lib/python${PYTHON_VERSION}/site-packages/nvidia"

if [ -d "$VENV_SITE" ]; then
    export LD_LIBRARY_PATH="$VENV_SITE/cuda_nvrtc/lib:$VENV_SITE/cuda_runtime/lib:$VENV_SITE/curand/lib:$VENV_SITE/cublas/lib:$VENV_SITE/cufft/lib:$VENV_SITE/cusolver/lib:$VENV_SITE/cusparse/lib:$VENV_SITE/nvjitlink/lib:$LD_LIBRARY_PATH"
    echo "[Setup] 已设置 CUDA 库路径 (Python $PYTHON_VERSION)"
fi

# 清理残留 socket
rm -f /tmp/rl_loop_demo.sock

# 自动启动 Env 和 Policy 进程
echo "[Demo] 自动启动 Env 和 Policy 进程..."
echo ""

# 启动 Env 进程 (后台)
$PYTHON env_process.py &
ENV_PID=$!
echo "[Demo] Env 进程已启动 (PID: $ENV_PID)"

# 等待 socket 创建
sleep 1

# 启动 Policy 进程 (前台)
$PYTHON policy_process.py

# 等待 Env 进程结束
wait $ENV_PID

echo ""
echo "[Demo] 演示完成!"
