# JaT-AudioSR: Audio Super-Resolution with Diffusion Transformers

高质量音频超分辨率系统，基于 Diffusion Transformer (DiT) 架构，使用 Flow Matching 训练范式和 DAC 编码器在潜空间进行生成。

## 主要特性

- **DiT 架构**: Diffusion Transformer with Grouped-Query Attention (GQA) 和 RoPE 位置编码
- **Flow Matching**: x-prediction 训练方法，直接预测干净信号
- **DAC 潜空间**: 使用 Descript Audio Codec (44.1kHz) 在压缩潜空间训练，效率提升 512x
- **Latent Perceptual Loss**: 频域感知损失 + 多尺度损失 + 一致性约束，无需解码开销
- **GQA 优化**: 20 Q-heads, 4 KV-heads，节省 80% KV 参数
- **CFG 推理**: Classifier-Free Guidance 支持可控生成质量

## 架构说明

### 模型结构
```
Low-Res Audio (16kHz) → DAC Encoder → LR Latent [B, 1024, T/512]
                                          ↓
                                    DiT Transformer (28 layers)
                                    - GQA (20Q/4KV heads)
                                    - RoPE 位置编码
                                    - AdaLN-Zero 调制
                                          ↓
                            HR Latent [B, 1024, T/512] → DAC Decoder
                                          ↓
                                High-Res Audio (44.1kHz)
```

### 训练脚本对比

| 脚本 | 参数量 | 显存需求 | 主要特性 | 适用场景 |
|------|--------|---------|---------|---------|
| `train_ddp_v2.py` | 288M | ~11GB | 小模型，快速训练 | 资源受限，快速实验 |
| `train_ddp_v3.py` | 766M | ~30GB | 大模型，仅 MSE loss | 基础 baseline |
| `train_ddp_v3mod1.py` | 766M | ~30GB | + 频域损失 + 多尺度损失 | 初步改进（有频域 bug） |
| **`train_ddp_v3mod2.py`** | **766M** | **~30GB** | **完整 Latent Perceptual Loss** | **推荐使用** |

#### MOD2 核心改进

**train_ddp_v3mod2.py** 包含以下关键创新：

1. **Fixed Frequency-Domain Loss** (解决金属音和噪点)
   - 对数幅度损失 `L1(log(Mag))` 替代线性损失（符合听觉感知）
   - 智能相位约束：仅约束低 30% 频率（避免高频随机相位噪声）
   - 移除高频加权（防止过度强调高频导致的噪点）

2. **Hybrid Consistency Loss** (修复 Mel L1 恶化问题)
   - 物理约束：`Downsample(Generated_HR) ≈ Input_LR`
   - 三段式频率策略：
     - 严格带 (0-0.3Fs): Complex L1（严格约束）
     - 过渡带 (0.3-0.36Fs): 幅度 L1 + 线性衰减（平滑过渡）
     - 高频带 (0.36-0.5Fs): 无约束（允许自由生成）
   - 线性衰减 mask 模拟抗混叠滤波器

3. **Multi-Scale Latent Loss**
   - 多分辨率潜空间监督 (scales: 1, 2, 4)
   - 捕获不同时间尺度的结构信息

4. **FP32 FFT 处理**
   - 解决 cuFFT FP16 对非 2 的幂次长度限制
   - 所有频域操作强制 `.float()` 转换

**性能提升预期**:
- LSD: 13.08 dB → **8-10 dB** (目标)
- Mel L1: 4.30 dB → **3.5-3.8 dB** (修复恶化)
- Mel L2: 5.80 dB → **持续改进**
- 金属音、高频噪点完全消除

## 安装依赖

### 环境要求
- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐 12.1)
- 显卡: 2× RTX 4090 (24GB) 或 2× A100 (40GB+)

### 安装步骤

```bash
# 克隆仓库
git clone <your-repo-url>
cd JaT

# 创建虚拟环境
conda create -n audiosr python=3.10
conda activate audiosr

# 安装 PyTorch (示例，请根据 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install descript-audio-codec  # DAC codec
pip install tensorboard            # 训练监控
pip install torchaudio soundfile   # 音频处理
pip install einops                 # 张量操作
pip install tqdm                   # 进度条
```

## 数据准备

### 1. 准备源音频
将高质量音频 (44.1kHz, FLAC/WAV) 放入 `1_source_audio/` 目录。

### 2. 数据预处理

```bash
# 使用 v5 预处理脚本（最新版本）
python prepare_dataset_v5.py
```

预处理步骤：
- 自动重采样至 44.1kHz
- 生成 16kHz 低分辨率对（Low-Pass + Downsampling）
- DAC 编码至潜空间 (1024 channels, T/512 compression)
- 计算全局归一化统计量
- 分割 train/val 集（90/10）

输出目录：`data_processed_v13_final/`

## 训练

### 快速开始（推荐 MOD2）

```bash
# 双卡 DDP 训练 (推荐)
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_ddp_v3mod2.py

# 单卡训练（调试用）
python train_ddp_v3mod2.py
```

### 训练配置参数

关键超参数（在脚本中修改）：

```python
# 模型配置
hidden_size = 1280        # v3mod2: 1280, v2: 1024
depth = 28                # v3mod2: 28, v2: 16
num_q_heads = 20          # Query heads
num_kv_heads = 4          # KV heads (GQA)

# 训练配置
batch_size = 28           # 每卡 batch size
num_steps = 100000        # 总训练步数
lr = 1e-4                 # 学习率
warmup_steps = 500        # Warmup 步数

# Loss 权重
latent_loss_weight = 0.3  # Latent Perceptual Loss 总权重
freq_loss_weight = 0.5    # 频域损失权重
ms_loss_weight = 0.5      # 多尺度损失权重
consistency_weight = 0.1  # 一致性损失权重（降低因其数值较大）

# CFG 配置
cfg_scale = 3.0           # 推理时 CFG 强度
classifier_free_prob = 0.1  # 训练时 10% 无条件
condition_noise_ratio = 0.05  # 条件噪声增强
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir runs/v3mod2_full_run --port 6006
```

关键监控指标：
- `Loss/Total`: 总损失
- `Loss/MSE`: MSE 损失
- `LatentPerc_FreqDomain`: 频域损失
- `LatentPerc_MultiScale`: 多尺度损失
- `LatentPerc_Consistency`: 一致性损失（修复 Mel L1 恶化）

### 断点恢复

```bash
# 脚本会自动从最新 checkpoint 恢复
# Checkpoint 保存路径: checkpoints/v3mod2_full_run/
```

## 推理

### 基础推理

```bash
# 使用 v3mod2 模型推理
python infer_test_v3m2.py \
    --input_audio path/to/lowres.wav \
    --output_dir inference_output \
    --checkpoint checkpoints/v3mod2_full_run/step_100000.pt \
    --cfg_scale 3.0 \
    --num_steps 50
```

### 推理参数说明

- `--cfg_scale`: CFG 强度 (推荐 2.0-4.0)
  - 2.0: 自然，可能略模糊
  - 3.0: **推荐平衡点**
  - 4.0: 更清晰，可能过增强
- `--num_steps`: 采样步数 (推荐 25-50)
  - 25: 快速，质量略降
  - **50: 推荐**
  - 100: 高质量，速度慢 2x

### 分块处理（长音频）

推理脚本自动处理长音频分块：
- Chunk size: 10 秒
- Overlap: 1 秒
- 自动交叉淡化拼接

## 项目结构

```
JaT/
├── train_ddp_v2.py              # 小模型训练脚本 (288M)
├── train_ddp_v3.py              # 大模型 baseline (766M, MSE only)
├── train_ddp_v3mod1.py          # MOD1: 频域+多尺度 (有 bug)
├── train_ddp_v3mod2.py          # MOD2: 完整改进 (推荐)
├── infer_test_v2.py             # v2 推理脚本
├── infer_test_v3.py             # v3 推理脚本
├── infer_test_v3m2.py           # v3mod2 推理脚本
├── prepare_dataset_v5.py        # 数据预处理脚本
├── calculate_metrics.py         # 音频质量评估 (LSD, Mel Loss)
├── calculate_model_params.py    # 模型参数量分析
├── compare_v2_v3_params.py      # v2 vs v3mod2 配置对比
├── test_consistency_methods.py  # 一致性损失方法测试
├── src/
│   └── models/
│       └── jat_audiosr_v2.py    # 核心模型定义
├── scripts/
│   └── prepare_dataset_*.py     # 历史预处理脚本
├── data_processed_v13_final/    # 预处理数据集
├── 1_source_audio/              # 原始音频
├── checkpoints/                 # 训练 checkpoint
├── runs/                        # TensorBoard 日志
└── inference_output/            # 推理输出
```

## 评估指标

使用 `calculate_metrics.py` 评估生成音频质量：

```bash
python calculate_metrics.py \
    --generated path/to/generated.wav \
    --reference path/to/reference.wav \
    --lowres path/to/lowres.wav
```

**核心指标**:
- **LSD (Log-Spectral Distance)**: 对数谱距离，越低越好
  - 当前: 13.08 dB
  - 目标: < 10 dB
- **Mel L1**: Mel 谱 L1 距离
  - 当前: 4.30 dB (vs LR 4.04 dB，恶化 -6.4%)
  - MOD2 目标: 3.5-3.8 dB
- **Mel L2**: Mel 谱 L2 距离
  - 当前: 5.80 dB (vs LR 7.30 dB，改进 +20.5%)

## 技术细节

### Latent Perceptual Loss 详解

**MOD2 完整损失函数**:
```
L_total = L_MSE + λ_latent × (L_freq + L_ms + L_consistency)

其中:
  L_freq = log_mag_loss + 0.1 × low_freq_phase_loss
  L_ms   = (1/K) × Σ L1(pred_scale_k, target_scale_k)
  L_consistency = strict_loss + transition_loss
```

**权重配置**:
```python
λ_latent = 0.3          # 总潜空间损失权重
freq_weight = 0.5       # 频域损失
ms_weight = 0.5         # 多尺度损失
consistency_weight = 0.1  # 一致性损失（因数值大 ~20 而降低）
```

### 为什么 MOD2 修复了金属音问题？

**问题根源**:
1. 线性幅度损失 `L1(Mag)` 对高频小误差过度惩罚
2. 全频段相位约束包括高频随机相位噪声
3. 高频加权 `high_freq_weight=2.0` 过度强调高频

**MOD2 解决方案**:
1. **对数幅度损失**: `L1(log(Mag + eps))` 符合人耳对数感知
2. **智能相位约束**: 仅约束低 30% 频率（高频相位人耳不敏感）
3. **移除高频加权**: 避免过度优化导致的伪影

### 为什么需要 Consistency Loss？

**观察到的问题**:
- Mel L1: 4.04 → **4.30 dB** (恶化 -6.4%)
- 模型"幻想"高频内容与 LR 输入不一致

**物理约束**:
```
Downsample(Generated_HR) ≈ Input_LR
```

**实现策略**:
- 严格约束低频 (0-0.3Fs): 确保与 LR 完全一致
- 平滑过渡 (0.3-0.36Fs): 线性衰减避免频谱不连续
- 自由生成高频 (0.36-0.5Fs): 允许模型重建细节

## 常见问题

### Q1: 训练时显存溢出怎么办？
**A**: 使用 `train_ddp_v2.py` (288M, 11GB) 或降低 batch_size。

### Q2: MOD1 和 MOD2 的主要区别？
**A**: MOD2 修复了 MOD1 的频域损失 bug（金属音+噪点），并新增一致性损失修复 Mel L1 恶化。

### Q3: 推荐的 CFG scale 是多少？
**A**: 3.0 为最佳平衡点。2.0 偏自然但模糊，4.0 更清晰但可能过增强。

### Q4: 为什么 consistency_weight 这么小 (0.1)？
**A**: 一致性损失数值较大 (~20)，有效权重为 0.3 × 0.1 = 0.03，与其他损失同量级。

### Q5: 如何选择 v2 还是 v3mod2？
**A**:
- **v2 (288M)**: 显卡 < 30GB，需要快速迭代
- **v3mod2 (766M)**: 显卡 ≥ 40GB，追求最佳质量

## 引用

如果使用本项目，请引用以下工作：

### DAC (Descript Audio Codec)
```bibtex
@inproceedings{kumar2024high,
  title={High-Fidelity Audio Compression with Improved RVQGAN},
  author={Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  booktitle={NeurIPS},
  year={2024}
}
```

### Flow Matching
```bibtex
@article{lipman2023flow,
  title={Flow matching for generative modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  journal={ICLR},
  year={2023}
}
```

### DiT (Diffusion Transformer)
```bibtex
@inproceedings{peebles2023scalable,
  title={Scalable diffusion models with transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={ICCV},
  year={2023}
}
```

## 致谢

感谢以下开源项目的启发：
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
- [DiT](https://github.com/facebookresearch/DiT)
- [AudioLDM 2](https://github.com/haoheliu/AudioLDM2)
- [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution)

## 许可证

本项目仅供研究和教育用途。

## 联系方式

如有问题或建议，请提交 GitHub Issue。

---

**最后更新**: 2025-12-09
**推荐版本**: `train_ddp_v3mod2.py` (完整 Latent Perceptual Loss)
