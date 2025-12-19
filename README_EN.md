<div align="center">

```
╔═════════════════════════════════════════════╗
║                                             ║
║       ██╗ █████╗ ████████╗███████╗██████╗   ║
║       ██║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗  ║
║       ██║███████║   ██║   ███████╗██████╔╝  ║
║  ██   ██║██╔══██║   ██║   ╚════██║██╔══██╗  ║
║  ╚█████╔╝██║  ██║   ██║   ███████║██║  ██║  ║
║   ╚════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝  ║
║                                             ║
║    Just Audio Transformer Super Solution    ║
║                                             ║
╚═════════════════════════════════════════════╝
```

# JaT-AudioSR: Audio Super-Resolution with Diffusion Transformers

🎵 **High-Quality Audio Super-Resolution System** 🎵

Based on Diffusion Transformer (DiT) architecture, using Flow Matching training paradigm and DAC encoder for latent space generation.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)](LICENSE)

</div>

## Key Features

- **DiT Architecture**: Diffusion Transformer with Grouped-Query Attention (GQA) and RoPE positional encoding
- **Flow Matching**: x-prediction training method, directly predicts clean signal
- **DAC Latent Space**: Train in compressed latent space using Descript Audio Codec (44.1kHz) with 512x efficiency improvement
- **Latent Perceptual Loss**: Frequency-domain perceptual loss + multi-scale loss + consistency constraints, without decoding overhead
- **GQA Optimization**: 20 Q-heads, 4 KV-heads, saving 80% KV parameters
- **CFG Inference**: Classifier-Free Guidance support for controllable generation quality

## Architecture Overview

### Model Structure
```
Low-Res Audio (16kHz) → DAC Encoder → LR Latent [B, 1024, T/512]
                                          ↓
                                    DiT Transformer (28 layers)
                                    - GQA (20Q/4KV heads)
                                    - RoPE positional encoding
                                    - AdaLN-Zero modulation
                                          ↓
                            HR Latent [B, 1024, T/512] → DAC Decoder
                                          ↓
                                High-Res Audio (44.1kHz)
```

### Training Script Comparison

| Script | Parameters | VRAM Required | Key Features | Use Case |
|--------|-----------|---------------|--------------|----------|
| `train_ddp_v1.py` | 60+75M | ~7GB | Minimal model, quick validation | Proof of concept |
| `train_ddp_v2.py` | 288+75M | ~13GB | Small model, fast training | Resource-constrained, rapid experiments |
| `train_ddp_v3.py` | 766+75M | ~34GB | Large model, MSE loss only | Basic baseline |
| `train_ddp_v3mod1.py` | 780+75M | ~34GB | + Frequency loss + Multi-scale loss | Initial improvements (has frequency bug) |
| **`train_ddp_v3mod2.py`** | **766M** | **~30GB** | **Complete Latent Perceptual Loss** | **Recommended** |

#### MOD2 Core Improvements

**train_ddp_v3mod2.py** includes the following key innovations:

1. **Fixed Frequency-Domain Loss** (Eliminates metallic artifacts and noise)
   - Log-magnitude loss `L1(log(Mag))` instead of linear loss (matches auditory perception)
   - Intelligent phase constraint: Only constrains lower 30% frequencies (avoids high-frequency random phase noise)
   - Removes high-frequency weighting (prevents artifacts from over-emphasizing high frequencies)

2. **Hybrid Consistency Loss** (Fixes Mel L1 degradation issue)
   - Physical constraint: `Downsample(Generated_HR) ≈ Input_LR`
   - Three-band frequency strategy:
     - Strict band (0-0.3Fs): Complex L1 (strict constraint)
     - Transition band (0.3-0.36Fs): Magnitude L1 + linear decay (smooth transition)
     - High-frequency band (0.36-0.5Fs): No constraint (allows free generation)
   - Linear decay mask simulates anti-aliasing filter

3. **Multi-Scale Latent Loss**
   - Multi-resolution latent space supervision (scales: 1, 2, 4)
   - Captures structural information at different temporal scales

4. **FP32 FFT Processing**
   - Resolves cuFFT FP16 limitations for non-power-of-2 lengths
   - All frequency-domain operations forced to `.float()` conversion

**Expected Performance Improvements**:
- LSD: 13.08 dB → **8-10 dB** (target)
- Mel L1: 4.30 dB → **3.5-3.8 dB** (fixes degradation)
- Mel L2: 5.80 dB → **Continuous improvement**
- Complete elimination of metallic artifacts and high-frequency noise

## Installation

### Requirements
- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (12.1 recommended)
- GPU: 2× RTX 4090 (24GB) or 2× A100 (40GB+)

### Installation Steps

```bash
# Clone repository
git clone https://github.com/HUSRCF/JaTSR-Just-audio-transformer-super-solution.git
cd JaTSR-Just-audio-transformer-super-solution

# Create virtual environment
conda create -n audiosr python=3.10
conda activate audiosr

# Install PyTorch (adjust based on your CUDA version)
# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**Note**: If you don't have CUDA, you can install the CPU version:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data Preparation

### 1. Prepare Source Audio
Place high-quality audio files (44.1kHz, FLAC/WAV) into the `1_source_audio/` directory.

### 2. Data Preprocessing

```bash
# Use v5 preprocessing script (latest version)
python prepare_dataset_v5.py
```

Preprocessing steps:
- Automatic resampling to 44.1kHz
- Generate 16kHz low-resolution pairs (Low-Pass + Downsampling)
- DAC encoding to latent space (1024 channels, T/512 compression)
- Calculate global normalization statistics
- Split train/val sets (90/10)

Output directory: `data_processed_v13_final/`

## Training

### Quick Start (MOD2 Recommended)

```bash
# Dual-GPU DDP training (recommended)
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_ddp_v3mod2.py

# Single GPU training (for debugging)
python train_ddp_v3mod2.py
```

### Training Configuration Parameters

Key hyperparameters (modify in script):

```python
# Model configuration
hidden_size = 1280        # v3mod2: 1280, v2: 1024
depth = 28                # v3mod2: 28, v2: 16
num_q_heads = 20          # Query heads
num_kv_heads = 4          # KV heads (GQA)

# Training configuration
batch_size = 28           # Batch size per GPU
num_steps = 100000        # Total training steps
lr = 1e-4                 # Learning rate
warmup_steps = 500        # Warmup steps

# Loss weights
latent_loss_weight = 0.3  # Total Latent Perceptual Loss weight
freq_loss_weight = 0.5    # Frequency-domain loss weight
ms_loss_weight = 0.5      # Multi-scale loss weight
consistency_weight = 0.1  # Consistency loss weight (reduced due to larger magnitude)

# CFG configuration
cfg_scale = 3.0           # CFG strength during inference
classifier_free_prob = 0.1  # 10% unconditional during training
condition_noise_ratio = 0.05  # Condition noise augmentation
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir runs/v3mod2_full_run --port 6006
```

Key monitoring metrics:
- `Loss/Total`: Total loss
- `Loss/MSE`: MSE loss
- `LatentPerc_FreqDomain`: Frequency-domain loss
- `LatentPerc_MultiScale`: Multi-scale loss
- `LatentPerc_Consistency`: Consistency loss (fixes Mel L1 degradation)

### Resume from Checkpoint

```bash
# Script automatically resumes from latest checkpoint
# Checkpoint save path: checkpoints/v3mod2_full_run/
```

## Inference

### Basic Inference

```bash
# Inference using v3mod2 model
python infer_test_v3m2.py \
    --input_audio path/to/lowres.wav \
    --output_dir inference_output \
    --checkpoint checkpoints/v3mod2_full_run/step_100000.pt \
    --cfg_scale 3.0 \
    --num_steps 50
```

### Inference Parameter Guide

- `--cfg_scale`: CFG strength (recommended 2.0-4.0)
  - 2.0: Natural, may be slightly blurry
  - 3.0: **Recommended balance**
  - 4.0: Sharper, may be over-enhanced
- `--num_steps`: Sampling steps (recommended 25-50)
  - 25: Fast, slightly lower quality
  - **50: Recommended**
  - 100: High quality, 2x slower

### Chunked Processing (Long Audio)

Inference script automatically handles long audio chunking:
- Chunk size: 10 seconds
- Overlap: 1 second
- Automatic crossfade stitching

## Project Structure

```
JaT/
├── train_ddp_v2.py              # Small model training script (288M)
├── train_ddp_v3.py              # Large model baseline (766M, MSE only)
├── train_ddp_v3mod1.py          # MOD1: Frequency+multi-scale (has bugs)
├── train_ddp_v3mod2.py          # MOD2: Complete improvements (recommended)
├── infer_test_v2.py             # v2 inference script
├── infer_test_v3.py             # v3 inference script
├── infer_test_v3m2.py           # v3mod2 inference script
├── prepare_dataset_v5.py        # Data preprocessing script
├── calculate_metrics.py         # Audio quality evaluation (LSD, Mel Loss)
├── calculate_model_params.py    # Model parameter analysis
├── compare_v2_v3_params.py      # v2 vs v3mod2 configuration comparison
├── test_consistency_methods.py  # Consistency loss method testing
├── src/
│   └── models/
│       └── jat_audiosr_v2.py    # Core model definition
├── scripts/
│   └── prepare_dataset_*.py     # Historical preprocessing scripts
├── data_processed_v13_final/    # Preprocessed dataset
├── 1_source_audio/              # Original audio files
├── checkpoints/                 # Training checkpoints
├── runs/                        # TensorBoard logs
└── inference_output/            # Inference outputs
```

## Evaluation Metrics

Use `calculate_metrics.py` to evaluate generated audio quality:

```bash
python calculate_metrics.py \
    --generated path/to/generated.wav \
    --reference path/to/reference.wav \
    --lowres path/to/lowres.wav
```

**Core Metrics**:
- **LSD (Log-Spectral Distance)**: Log-spectral distance, lower is better
  - Current: 13.08 dB
  - Target: < 10 dB
- **Mel L1**: Mel-spectrogram L1 distance
  - Current: 4.30 dB (vs LR 4.04 dB, degraded -6.4%)
  - MOD2 Target: 3.5-3.8 dB
- **Mel L2**: Mel-spectrogram L2 distance
  - Current: 5.80 dB (vs LR 7.30 dB, improved +20.5%)

## Technical Details

### Latent Perceptual Loss Explained

**MOD2 Complete Loss Function**:
```
L_total = L_MSE + λ_latent × (L_freq + L_ms + L_consistency)

Where:
  L_freq = log_mag_loss + 0.1 × low_freq_phase_loss
  L_ms   = (1/K) × Σ L1(pred_scale_k, target_scale_k)
  L_consistency = strict_loss + transition_loss
```

**Weight Configuration**:
```python
λ_latent = 0.3          # Total latent space loss weight
freq_weight = 0.5       # Frequency-domain loss
ms_weight = 0.5         # Multi-scale loss
consistency_weight = 0.1  # Consistency loss (reduced due to larger magnitude ~20)
```

### Why Does MOD2 Fix Metallic Artifacts?

**Root Causes**:
1. Linear magnitude loss `L1(Mag)` over-penalizes small high-frequency errors
2. Full-band phase constraint includes high-frequency random phase noise
3. High-frequency weighting `high_freq_weight=2.0` over-emphasizes high frequencies

**MOD2 Solutions**:
1. **Log-magnitude loss**: `L1(log(Mag + eps))` matches logarithmic human perception
2. **Intelligent phase constraint**: Only constrains lower 30% frequencies (high-frequency phase is perceptually insensitive)
3. **Remove high-frequency weighting**: Avoids artifacts from over-optimization

### Why is Consistency Loss Needed?

**Observed Problem**:
- Mel L1: 4.04 → **4.30 dB** (degraded -6.4%)
- Model "hallucinates" high-frequency content inconsistent with LR input

**Physical Constraint**:
```
Downsample(Generated_HR) ≈ Input_LR
```

**Implementation Strategy**:
- Strict constraint on low frequencies (0-0.3Fs): Ensure complete consistency with LR
- Smooth transition (0.3-0.36Fs): Linear decay to avoid spectral discontinuities
- Free high-frequency generation (0.36-0.5Fs): Allow model to reconstruct details

## FAQ

### Q1: What if I run out of VRAM during training?
**A**: Use `train_ddp_v2.py` (288M, 11GB) or reduce batch_size.

### Q2: What's the main difference between MOD1 and MOD2?
**A**: MOD2 fixes MOD1's frequency-domain loss bugs (metallic artifacts + noise) and adds consistency loss to fix Mel L1 degradation.

### Q3: What's the recommended CFG scale?
**A**: 3.0 is the optimal balance. 2.0 is more natural but blurry, 4.0 is sharper but may be over-enhanced.

### Q4: Why is consistency_weight so small (0.1)?
**A**: Consistency loss has larger magnitude (~20), so effective weight is 0.3 × 0.1 = 0.03, on par with other losses.

### Q5: How to choose between v2 and v3mod2?
**A**:
- **v2 (288M)**: GPU < 30GB, need rapid iteration
- **v3mod2 (766M)**: GPU ≥ 40GB, pursue best quality

## Citations

If you use this project, please cite the following works:

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

## Acknowledgments

Thanks to the following open-source projects for inspiration:
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)
- [DiT](https://github.com/facebookresearch/DiT)
- [AudioLDM 2](https://github.com/haoheliu/AudioLDM2)
- [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution)

## License

This project is for research and educational purposes only.

## Contact

For questions or suggestions, please submit a GitHub Issue.

---

**Last Updated**: 2025-12-09
**Recommended Version**: `train_ddp_v3mod2.py` (Complete Latent Perceptual Loss)
