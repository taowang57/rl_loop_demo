# CUDA IPC RL Demo (v1.1 - PyTorch 集成版)

跨进程 GPU 显存零拷贝共享演示项目，展示如何在 RL（强化学习）场景中使用 CUDA IPC 实现 Env 进程和 Policy 进程之间的高效通信。

**v1.1 新特性：** Policy 进程现在使用真实的 PyTorch MLP 神经网络，通过 DLPack 协议实现 CuPy ↔ PyTorch 零拷贝互操作。

## 快速开始

### 方式一：一键运行（推荐）

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目后，一键运行
cd rl_loop_demo
uv run run_demo.py
```

该命令会自动：
1. 创建虚拟环境
2. 安装所有依赖（包括 CuPy 和 NVIDIA CUDA 库）
3. 配置 CUDA 库路径
4. 启动 Env 和 Policy 两个进程
5. 运行 100 步 RL 循环

### 方式二：Shell 脚本

```bash
./run_demo.sh
```

---

## 项目架构

```
┌─────────────────┐     CUDA IPC      ┌─────────────────┐
│   Env 进程      │◄═══════════════►   │  Policy 进程    │
│ (物理仿真)      │   GPU 共享显存     │ (PyTorch MLP)   │
└─────────────────┘                    └─────────────────┘
        │                                      │
        └──────────── 零拷贝通信 ──────────────┘
                     (DLPack 协议)
```

- **Env 进程**: 分配 GPU 内存，运行物理仿真，写入 State/Reward
- **Policy 进程**: 导入共享 GPU 内存，运行 PyTorch MLP 推理，写入 Action

### 数据流 (零拷贝)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               GPU 共享显存 (CUDA IPC)                                    │
├──────────────────────────────────────────┬──────────────────────────────────────────────┤
│             Env 进程                      │              Policy 进程                     │
│                                          │                                              │
│                                          │   ┌────────────────────────────────────────┐ │
│                                          │   │         PyTorch MLP 推理               │ │
│                                          │   │  ┌──────────┐      ┌──────────┐        │ │
│                                          │   │  │  Input   │ ───► │  Output  │        │ │
│                                          │   │  │  Tensor  │      │  Tensor  │        │ │
│                                          │   │  └────▲─────┘      └────┬─────┘        │ │
│                                          │   └───────┼─────────────────┼──────────────┘ │
│                                          │           │ DLPack          │ DLPack         │
│                                          │           │ (Zero-Copy)     │                │
│   ┌──────────────┐     IPC Handle        │           │                 ▼                │
│   │ State Buffer │───────────────────────┼──────────►│ CuPy       CuPy Array            │
│   │              │     (Zero-Copy)       │            (映射)            │                │
│   └──────────────┘                       │                             │                │
│          ▲                               │                             │ D2D Copy       │
│          │ CuPy 写入                      │                             │                │
│          │                               │   ┌──────────────┐          │                │
│   ┌──────┴────────────────────┐          │   │              │◄─────────│                │
│   │  物理仿真 / 状态更新         │          │   │Action Buffer │     (GPU 内存间拷贝)       │
│   │  reward = f(state, action)│◄─────────┼───│              │                           │
│   └───────────────────────────┘   CuPy   │   └──────────────┘                           │
│                                   读取    │                                              │
└──────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 硬件与软件环境配置

### 硬件环境

| 组件 | 规格 |
|------|------|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| **GPU 显存** | 97,887 MiB (~96 GB) |
| **驱动版本** | 590.48.01 |
| **最高支持 CUDA** | 13.1 |

### 软件环境

| 组件 | 版本 |
|------|------|
| **操作系统** | Ubuntu 24.04.3 LTS (Noble Numbat) |
| **内核** | 6.14.0-37-generic |
| **Python** | 3.12.3 |
| **uv** | 0.9.27 |

### Python 包依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| `cupy-cuda12x` | 13.6.0 | GPU 计算库（NumPy 替代） |
| `torch` | ≥2.2.0 | PyTorch 深度学习框架 |
| `numpy` | 2.4.1 | 数值计算 |
| `nvidia-cuda-nvrtc-cu12` | 12.9.86 | CUDA 运行时编译器 |
| `nvidia-cuda-runtime-cu12` | 12.9.79 | CUDA 运行时库 |
| `nvidia-curand-cu12` | 10.3.10.19 | CUDA 随机数库 |
| `nvidia-cublas-cu12` | 12.9.1.4 | CUDA 线性代数库 |
| `nvidia-cufft-cu12` | 11.4.1.4 | CUDA FFT 库 |
| `nvidia-cusolver-cu12` | 11.7.5.82 | CUDA 求解器库 |
| `nvidia-cusparse-cu12` | 12.5.10.65 | CUDA 稀疏矩阵库 |
| `nvidia-nvjitlink-cu12` | 12.9.86 | CUDA JIT 链接器 |
| `fastrlock` | 0.8.3 | CuPy 依赖 |

---

## 一键运行指南：`uv run run_demo.py`

### 命令执行流程详解

`uv run run_demo.py` 命令会自动完成以下所有步骤：

```
┌─────────────────────────────────────────────────────────────┐
│                    uv run run_demo.py                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: uv 读取 pyproject.toml                              │
│ - 解析项目依赖配置                                           │
│ - 确定 Python 版本要求 (>=3.10)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: uv 创建虚拟环境 (.venv)                              │
│ - 自动选择系统 Python 解释器                                 │
│ - 创建隔离的虚拟环境                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: uv 安装依赖 (~1.5 GB)                               │
│ - cupy-cuda12x (GPU 计算库)                                 │
│ - nvidia-cuda-nvrtc-cu12 (运行时编译器)                     │
│ - nvidia-curand-cu12 (随机数)                               │
│ - nvidia-cublas-cu12 (线性代数)                             │
│ - ... (其他 CUDA 库)                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: uv 执行 run_demo.py                                 │
│ - 使用虚拟环境中的 Python 解释器                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: run_demo.py 设置 CUDA 库路径                         │
│ - 自动检测 site-packages/nvidia/ 目录                       │
│ - 配置 LD_LIBRARY_PATH 环境变量                             │
│ - 包含: cuda_nvrtc, curand, cublas, cufft 等               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: run_demo.py 启动 Env 进程                           │
│ - 分配 5120 bytes GPU 共享缓冲区                            │
│ - 创建 CUDA IPC 句柄                                        │
│ - 通过 Unix Socket 等待 Policy 连接                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 7: run_demo.py 启动 Policy 进程                         │
│ - 连接 Env 进程                                              │
│ - 接收 IPC 句柄                                              │
│ - 映射到相同的 GPU 物理内存                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 8: 运行 RL 循环 (100 步)                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Env 进程                      Policy 进程              │ │
│ │   │                              │                      │ │
│ │   │◀─── 读取 Action ──────────── │                      │ │
│ │   │──── 写入 State ────────────▶ │                      │ │
│ │   │──── 写入 Reward ───────────▶ │                      │ │
│ │   │                              │                      │ │
│ │   │     [env_ready = 1]          │                      │ │
│ │   │─────────────────────────────▶│                      │ │
│ │   │                              │ 读取 State           │ │
│ │   │                              │ 计算并写入 Action    │ │
│ │   │     [policy_ready = 1]       │                      │ │
│ │   │◀─────────────────────────────│                      │ │
│ │   │                              │                      │ │
│ │   └──────── 交替循环 (100 步) ───┘                      │ │
│ └─────────────────────────────────────────────────────────┘ │
│ - 两个进程通过 GPU 显存直接通信                              │
│ - 零拷贝，无 CPU-GPU 数据传输开销                           │
│ - 通过 GPU 共享内存标志位 (env_ready/policy_ready) 同步     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 9: 演示完成                                             │
│ - 清理 IPC 句柄                                              │
│ - 清理 Unix Socket                                          │
│ - 两个进程正常退出                                           │
└─────────────────────────────────────────────────────────────┘
```

### pyproject.toml 配置说明

```toml
[project]
name = "rl-loop-demo"
version = "1.1.0"
description = "CUDA IPC 跨进程 GPU 显存共享 RL 演示项目 (含 PyTorch 神经网络)"
requires-python = ">=3.10"
dependencies = [
    # GPU 计算核心库
    "cupy-cuda12x>=13.0.0",
    "numpy>=2.0.0",
    # PyTorch (使用 DLPack 与 CuPy 零拷贝互操作)
    "torch>=2.2.0",

    # NVIDIA CUDA 运行时库
    # 这些库解决了 "libnvrtc.so.12 not found" 等问题
    "nvidia-cuda-nvrtc-cu12",    # 运行时编译器
    "nvidia-cuda-runtime-cu12",  # 运行时基础库
    "nvidia-curand-cu12",        # 随机数生成
    "nvidia-cublas-cu12",        # 线性代数 (BLAS)
    "nvidia-cufft-cu12",         # 快速傅里叶变换
    "nvidia-cusolver-cu12",      # 求解器
    "nvidia-cusparse-cu12",      # 稀疏矩阵
]
```

### 输出示例

```
==========================================
  CUDA IPC 跨进程 GPU 显存共享 RL Demo
==========================================

[Setup] 已设置 CUDA 库路径 (8 个目录)
[Demo] 自动启动 Env 和 Policy 进程...

[Demo] Env 进程已启动 (PID: 44944)
[Policy] PyTorch MLP 初始化完成:
  设备: cuda:0
  架构: 12 → 64 → 64 → 6
  参数量: 5,062
  输出激活: Tanh (范围 [-1, 1])

[Env] 分配 GPU 共享缓冲区: 5120 bytes
[Env] GPU 共享缓冲区基地址: 0x00007165AFA00000
  注: GPU 物理地址由 NVIDIA 驱动管理，用户空间只能访问设备虚拟地址
  两个进程通过 CUDA IPC 句柄映射到相同的 GPU 物理显存区域
[Env] GPU 缓冲区初始化完成，内存布局:
  [Env] Metadata | 虚拟地址: 0x00007165AFA00000 | 偏移: +    0 | 大小:   256 bytes
  [Env] State    | 虚拟地址: 0x00007165AFA00100 | 偏移: +  256 | 大小:  3072 bytes
  [Env] Action   | 虚拟地址: 0x00007165AFA00D00 | 偏移: + 3328 | 大小:  1536 bytes
  [Env] Reward   | 虚拟地址: 0x00007165AFA01300 | 偏移: + 4864 | 大小:   256 bytes
...

[Policy] Step   0 零拷贝转换验证:
         CuPy  State 地址: 0x00007CF715A00100
         Torch State 地址: 0x00007CF715A00100
         地址相同: True ✓
         MLP 推理完成: [64, 12] → [64, 6]
         Torch Action 地址: 0x00007CF715B00000 (MLP 输出)
         CuPy  Action 地址: 0x00007CF715A00D00 (IPC 共享)
         注: MLP 输出是新内存，需复制到 IPC 共享区域

[Env] Step   0 | Avg Reward: -9.9058
[Policy] Step   0 | Avg |Action|: 0.2156
...
[Env] RL 循环结束
[Policy] PyTorch 策略推理循环结束
[Demo] 演示完成!
```

---

## 项目文件结构

```
rl_loop_demo/
├── pyproject.toml       # 项目配置和依赖定义 (含 PyTorch)
├── run_demo.py          # 一键启动入口 (uv run run_demo.py)
├── run_demo.sh          # Shell 启动脚本 (备用)
├── env_process.py       # 环境进程 (分配GPU内存, 物理仿真)
├── policy_process.py    # 策略进程 (PyTorch MLP + DLPack 零拷贝)
├── shared_types.py      # 共享数据结构定义
├── ipc_utils.py         # CUDA IPC 句柄传递工具
├── DS.rl_loop_...md     # 详细设计文档 (架构、数据结构、同步机制)
└── README.md            # 项目说明
```

---

## 开发指南

### 代码格式化

格式化代码:
```bash
uv run ruff format .
```

检查代码风格:
```bash
uv run ruff check --fix .
```

---

## 核心技术: DLPack 零拷贝协议

### 什么是 DLPack?

DLPack 是一个跨框架的张量内存共享标准，允许不同的深度学习框架（PyTorch、CuPy、JAX、TensorFlow 等）共享 GPU 内存而无需数据拷贝。

### 本项目中的使用

```python
# CuPy → PyTorch (零拷贝)
state_cupy = ...  # 来自 CUDA IPC 共享缓冲区的 CuPy 数组
state_torch = torch.from_dlpack(state_cupy)  # 共享同一块 GPU 内存

# PyTorch → CuPy (零拷贝)
action_torch = model(state_torch)  # MLP 输出
action_cupy = cp.from_dlpack(action_torch)  # 转换为 CuPy
```

### 地址验证

运行 Demo 时，每 10 步会打印地址验证信息：

```
[Policy] Step   0 零拷贝转换验证:
         CuPy  State 地址: 0x00007CF715A00100
         Torch State 地址: 0x00007CF715A00100
         地址相同: True ✓
```

地址相同证明 CuPy 和 PyTorch 共享同一块 GPU 内存，实现了真正的零拷贝。

---

## 常见问题 FAQ

### Q1: 为什么需要单独安装 nvidia-* CUDA 库包？

**A**: CuPy 的 `cupy-cuda12x` 包是一个**轻量级**包，它不包含 CUDA 运行时库本身。这种设计允许用户选择：
1. 使用系统安装的 CUDA Toolkit（传统方式）
2. 使用 pip 安装的 CUDA 库（推荐，更便携）

我们选择方案 2，因为它不需要 root 权限，也不会污染系统环境。

### Q2: 为什么两个进程的虚拟地址不同？

**A**: 这是正常现象。CUDA IPC 的工作原理是：
1. Env 进程分配 GPU 内存，获得自己的虚拟地址（如 `0x00007165AFA00000`）
2. 生成 IPC 句柄（包含物理内存位置信息）
3. Policy 进程使用 IPC 句柄映射同一块物理内存
4. Policy 获得自己的虚拟地址（如 `0x00007CF715A00000`）

虽然虚拟地址不同，但它们都映射到相同的 GPU 物理内存，因此可以实现零拷贝通信。

### Q3: DLPack 转换时 CuPy 和 PyTorch 的地址为什么相同？

**A**: 这正是 DLPack 的设计目标！DLPack 不复制数据，而是：
1. CuPy 创建一个 DLPack capsule，包含 GPU 内存指针
2. PyTorch 从 capsule 中读取指针，直接使用该内存
3. 两个框架共享同一块 GPU 内存

这就是为什么 Demo 输出显示：
```
CuPy  State 地址: 0x00007CF715A00100
Torch State 地址: 0x00007CF715A00100
地址相同: True ✓
```

### Q4: MLP 输出为什么需要复制到 IPC 共享区域？

**A**: PyTorch 神经网络的输出是新分配的 GPU 内存（用于存储计算结果），它与 IPC 共享缓冲区是两块不同的内存：

```
IPC 共享缓冲区 (Env 分配)     MLP 输出 (PyTorch 分配)
┌─────────────────┐           ┌─────────────────┐
│ Action Buffer   │  ◄─copy── │ action_torch    │
│ 0x...A00D00     │           │ 0x...B00000     │
└─────────────────┘           └─────────────────┘
      ▲                              │
      │                              │
 Env 进程读取                    MLP forward() 输出
```

虽然这一步需要 GPU 内存拷贝（cudaMemcpy D2D），但这比跨 CPU 的拷贝快得多（~1μs vs ~50μs）。

### Q5: 如何验证两个进程确实在共享同一块内存？

**A**: 有两种佐证方式：

**1. 交替读写模式**：观察 demo 输出中的数据传递：
```
[Env]    读取 Action -> 用于下一步仿真
[Env]    写入 State
[Env]    写入 Reward
[Policy] 读取 State  -> 计算 Action
[Policy] 写入 Action
```
如果不是同一块内存，数据传递会失败，RL 循环无法正常进行。

**2. 偏移量一致性**：观察两个进程的内存布局输出：

| 区域 | Env 进程虚拟地址 | Policy 进程虚拟地址 | 偏移量 | 大小 |
|------|-----------------|-------------------|-------|------|
| Metadata | `0x00007165AFA00000` | `0x00007CF715A00000` | +0 | 256 bytes |
| State | `0x00007165AFA00100` | `0x00007CF715A00100` | +256 | 3072 bytes |
| Action | `0x00007165AFA00D00` | `0x00007CF715A00D00` | +3328 | 1536 bytes |
| Reward | `0x00007165AFA01300` | `0x00007CF715A01300` | +4864 | 256 bytes |

在用户空间，我们无法直接获取 GPU 物理地址，因为 GPU 物理地址只有驱动程序和硬件知道。但是我们可以显示：
1. **虚拟地址（设备指针）**：CuPy 数组的 `.data.ptr` 值
2. **基地址 + 偏移量**：显示每个区域相对于 IPC 共享内存基地址的偏移

虽然两个进程的虚拟地址不同，但 **偏移量完全一致**（+256, +3328, +4864），这证明它们通过 CUDA IPC 映射到了同一块 GPU 物理显存。

---

## 参考链接

- [CuPy 官方文档](https://docs.cupy.dev/)
- [CUDA IPC 官方文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)
- [uv 官方文档](https://docs.astral.sh/uv/)
- [NVIDIA CUDA Toolkit PyPI 包](https://pypi.org/project/nvidia-cuda-nvrtc-cu12/)

---

## 许可证

MIT
