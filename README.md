# 光子条件化低秩 MLP 蒸馏项目 (QKD-Photonic)

---

## 📌 项目简介

本项目以 `Qwen/Qwen3.5-0.8B-Base` 为冻结基座模型（Teacher），将其指定 Transformer Decoder 层的 SwiGLU MLP 替换为基于**连续变量（CV）高斯环形光子电路条件化门控**的低秩 MLP 模块。

通过将 **截断 SVD 低秩分解** 与 **光子量子电路的非线性演化** 结合，在大幅压缩模型参数的同时，利用蒸馏技术保留大语言模型的语义表达与推理能力。


---

## 🔬 核心架构与数学原理

一个标准的 SwiGLU MLP 包含 **Gate** (`gate_proj`)、**Up** (`up_proj`) 和 **Down** (`down_proj`) 三个线性投影通道。每个投影通道均由教师权重的 rank-$r$ 截断 SVD 初始化，并融合光子电路演化：

$$\begin{aligned} t &= B \cdot s & \text{(1. 截断 SVD 低秩基底映射，冻结/学习初始化)} \\ e &= \kappa \cdot \tanh(R \cdot t) & \text{(2. 特征压缩并映射至光子电路输入空间，} R \text{ 为正交投影)} \\ z &= Q_\theta(e) & \text{(3. 可插拔光子特征提供器：八模高斯环形电路演化/Mock)} \\ g &= 1 + 0.1 \cdot \tanh(C \cdot z + b) & \text{(4. 生成光子条件化增益门控 Gain)} \\ q(s) &= P \cdot (g \odot t) & \text{(5. 门控调制与低秩特征重构)} \end{aligned}$$

### ⚛️ 光子特征提供器 ($Q_\theta$)

* **`mock`**：模拟量接口，用于 CPU 快速开发与逻辑验证。
* **`deepquantum`**：基于 DeepQuantum 驱动的**八模连续变量高斯环形电路**。输出 16 维特征向量（包含 8 个平均光子数 + 8 个环状相邻相关量）。

> 🛡️ **隔离设计原则**：每层的 Gate、Up、Down 各自拥有**完全独立**的低秩矩阵 ($P, B, C, b, R$)、Provider 实例、光子电路调用、物理控制参数 $\theta$ 以及 EMA 统计状态；不同 Transformer 层之间互不共享。

---

## 📂 项目源码目录结构

```text
QKD_demo/
├── 📄 configs/                        # 实验配置文件 (YAML 驱动)
│   ├── stage1.yaml                   # 阶段一：单层局部重建配置
│   └── stage2.yaml                   # 阶段二：多层合并与端到端蒸馏配置
│
├── 📜 scripts/                        # 辅助与训练入口脚本
│   ├── prepare_data.py               # JSONL 对话数据 Token 化与数据集切分
│   ├── train_stage1.py               # 阶段一训练入口
│   └── train_stage2.py               # 阶段二训练入口
│
└── 📦 src/qkd/                       # 核心源码包
    ├── data.py                       # JSONL 数据集读取与批次动态 Padding
    ├── modeling.py                   # Transformers 模型与 Tokenizer 加载封装
    │
    ├── ⚛️ photonic/                    # 光子低秩模块与硬件仿真层
    │   ├── model.py                  # 低秩 MLP 架构、SVD 初始化与层替换
    │   ├── provider.py               # Mock / DeepQuantum 特征提供器与 EMA 维护
    │   ├── circuit.py                # 八模 DeepQuantum 连续变量高斯电路
    │   └── checkpoint.py             # 增量模块保存、恢复与 Stage 1 多层 Checkpoint 缝合
    │
    └── 🏋️ training/                  # 损失函数与双轨优化器
        ├── stage1_loss.py            # 阶段一 Output 重建损失与 Gate/Up/Down 诊断面板
        ├── stage2_loss.py            # 阶段二 0.5 CE + 0.5 τ² Top-K KD 联合损失
        ├── spsa.py                   # 物理参数 θ 的无梯度 SPSA 随机采样优化器
        └── tools.py                  # 运行期 YAML、命令行覆盖、硬件分发与无限 DataLoader

```

---

## 🚀 训练工作流 (配置驱动)

所有实验参数均以 `configs/` 中的 YAML 文件为唯一事实来源（Single Source of Truth）。训练过程分为两个阶段：

除直接修改 YAML 外，也可用可重复的 `--set '段.键=YAML值'` 临时覆盖参数；实际生效的配置会保存到输出目录的 `run_config.yaml`。例如：

```bash
python scripts/train_stage1.py --config configs/stage1.yaml \
  --set 'model.target_layers=[22]' \
  --set 'experiment.output_dir=outputs/stage1-layer22-r64'
```

```
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│       阶段一：局部单层重建 (Stage 1)  │     │      阶段二：端到端融合蒸馏 (Stage 2) │
├──────────────────────────────────────┤     ├──────────────────────────────────────┤
│ • 针对单个目标层 (如 Layer 21)       │     │ • 缝合多个 Stage 1 Checkpoints       │
│ • Hook 拦截特征，对齐单层 Output    │ ──► │ • 全网 Logits 蒸馏 (0.5 CE + 0.5 KD) │
│ • Adam 优化 P/B/C/b; SPSA 优化 θ     │     │ • 全局调优大模型最终文本生成能力     │
└──────────────────────────────────────┘     └──────────────────────────────────────┘

```

### 1️⃣ 数据准备

处理原始 JSONL 消息数据并切分训练集/验证集：

```bash
python scripts/prepare_data.py

```

### 2️⃣ 运行阶段一 (Stage 1：单层局部重建)

在 `configs/stage1.yaml` 中设置单目标层（例如 `target_layers: [21]`），运行训练：

```bash
python scripts/train_stage1.py --config configs/stage1.yaml

```

* **优化目标**：仅通过相对 MSE 加余弦惩罚对齐 Teacher 输出：$\mathcal{L}_{\text{stage1}} = \text{NMSE}(y_{\text{student}}, y_{\text{teacher}}) + 0.1 \times (1 - \text{cos}(y_{\text{student}}, y_{\text{teacher}}))$。
* **分步训练**：如需替换 22 或 23 层，只需修改 YAML 中的 `target_layers` 与 `experiment.output_dir` 分别运行。

### 3️⃣ 运行阶段二 (Stage 2：多层端到端融合蒸馏)

在 `configs/stage2.yaml` 的 `initialization.stage1_checkpoints` 中配置各层 Stage 1 产出目录：

```yaml
initialization:
  stage1_checkpoints:
    - outputs/stage1-layer21-r64
    - outputs/stage1-layer22-r64

```

随后运行端到端蒸馏：

```bash
python scripts/train_stage2.py --config configs/stage2.yaml

```

* **优化目标**：按 $0.5 \times \text{CE} + 0.5 \times \tau^2 \text{KD}$ 优化。蒸馏损失在右移预测位置上提取 Teacher Top-$K$ Logits 并补入真实 Target Token，去重后计算高效率 KL 散度。

---

## ⚙️ 关键配置参数一览

| 配置参数分类 | 参数 Key | 典型示例 | 解释与功能描述 |
| --- | --- | --- | --- |
| **模型与目标** | `model.target_layers` | `[21]` / `[21, 22]` | 指定被替换为光子 Low-Rank MLP 的 Layer 索引列表 |
| **低秩压缩** | `compression.rank` | `64` | SVD 截断低秩维度 $r$（决定模型参数压缩率） |
| **物理空间** | `compression.z_dim` | `16` | 光子电路特征空间维度（对应 8 光子数 + 8 环状相关量） |
| **光子提供器** | `photonic.provider` | `deepquantum` | 选择 `deepquantum` 高斯模拟器或 `mock` CPU 快速测试器 |
| **量子测量** | `photonic.shots` | `None` / `1024` | 测量 Shot 数（`None` 为理论解析期望，整数则注入量子采样噪声） |
| **双轨优化器** | `optimization.spsa_*` | `perturbation=0.01` | **SPSA 随机扰动优化器**试探步长与学习率（专门更新冻结梯度的 $\theta$） |
| **早停** | `optimization.early_stop_loss` | `null` / `0.1` | 当前训练总 loss 小于等于阈值时结束训练并保存；`null` 表示关闭 |

---

## 📦 产出文件目录规范

训练完成后，所有结果将自动保存至指定 `outputs/...` 目录（*不重复保存庞大的冻结基座模型权重*）：

```text
outputs/stage1-layer21/
├── 📄 photonic_modules.pt      # 轻量化权重：替换模块的 P/B/C/b/R/theta 与各 Provider 的 EMA 状态
├── 📄 photonic_config.json     # 模块配置元数据（包含 rank, z_dim, target_layers, teacher 名称等）
├── 📄 run_config.yaml          # 实验可复现快照：包含输入 YAML 完整副本、硬件设备与 UTC 运行时间
├── 📂 tokenizer/               # 关联的预训练 Tokenizer 完整文件 (便于后续评测与独立推理)
└── 📂 best/                    # (仅阶段一) 验证集指标最佳时的 Checkpoint 备份目录

```
