"""
JaT-AudioSR V3 MOD1 Training Script (Latent Perceptual Loss Edition)

V3 MOD1 Improvements (基于2024-2025最新研究):
1. ✅ Dropout 0.1 (Attention + MLP)
2. ✅ DropPath 0.05 (Stochastic Depth, 线性递增)
3. ✅ Conditional Noise 0.05 (基于batch std的自适应噪声)
4. 🔥 Frequency-Domain Latent Loss (FFT高频增强，减少"毛刺感")
5. 🔥 Multi-Scale Latent Loss (多时间尺度结构捕捉)
6. ✅ Phase 1 TensorBoard指标 + 新增Latent Loss监控
7. ✅ 智能检查点管理 (时间戳文件夹 + auto-resume)

Latent Perceptual Loss特性:
- ✅ 在Latent Space直接施加感知约束 (无需DAC解码)
- ✅ 计算开销极小 (~2-3ms/step vs Mel loss的~15ms)
- ✅ 与torch.compile()完全兼容
- ✅ 改善高频细节，减少音频artifact

参考文献:
- Boosting Latent Diffusion with Perceptual Objectives (2024)
- FreSca: Scaling in Frequency Space (2024)
- Smooth Diffusion: Crafting Smooth Latent Spaces (CVPR 2024)
"""

import os
import argparse
import json
import random
import math
import time
from pathlib import Path
from datetime import timedelta, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 引入 V2 模型
# 确保您的 src 目录在 python path 中
from src.models.jat_audiosr_v2 import JaT_AudioSR_V2

# ==============================================================================
# 🎯 Latent Perceptual Loss Modules (2024-2025 SOTA)
# ==============================================================================

class FrequencyDomainLatentLoss(nn.Module):
    """
    频域Latent损失 - 在FFT域施加约束，强调高频细节

    参考: FreSca: Scaling in Frequency Space (2024)
    https://arxiv.org/html/2504.02154v2

    特性:
    - 纯PyTorch操作 (torch.fft)
    - 高频增强权重 (减少"毛刺感")
    - 幅度 + 相位双重约束
    - 完全兼容torch.compile()
    """
    def __init__(self, high_freq_weight=2.0):
        super().__init__()
        self.high_freq_weight = high_freq_weight

    def forward(self, pred_latent, target_latent):
        """
        Args:
            pred_latent: [B, C, T] 预测的latent
            target_latent: [B, C, T] GT latent
        Returns:
            loss: scalar
        """
        # 🔥 核心修复: 强制转为 FP32
        # 1. 解决 cuFFT FP16 必须是 2 的幂的限制 (1378 不是 2 的幂)
        # 2. 解决 FP16 下相位计算精度极低的问题
        pred_latent = pred_latent.float()
        target_latent = target_latent.float()

        # 沿时间维度做FFT (实信号用rfft)
        pred_fft = torch.fft.rfft(pred_latent, dim=-1)  # [B, C, T//2+1] complex
        target_fft = torch.fft.rfft(target_latent, dim=-1)

        # 1. Complex L1 Loss (隐式包含幅度+相位，数值更稳定)
        # 优势：自动忽略低幅度频段的相位噪声，不会有梯度混乱问题
        complex_l1_loss = torch.abs(pred_fft - target_fft).mean()

        # 2. 幅度谱损失 (显式约束幅度)
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = F.l1_loss(pred_mag, target_mag)

        # 3. 高频增强 (线性权重递增，强调高频细节)
        freq_bins = pred_mag.shape[-1]
        freq_weights = torch.linspace(1.0, self.high_freq_weight, freq_bins,
                                       device=pred_latent.device)
        freq_weights = freq_weights.view(1, 1, -1)  # [1, 1, F]

        weighted_mag_loss = (freq_weights * torch.abs(pred_mag - target_mag)).mean()

        # 组合损失：Complex L1 + 幅度 + 高频增强
        # Complex L1已包含相位信息（隐式幅度加权），无需额外相位损失
        # 降低 mag_loss 权重避免与 complex_l1 冗余，主要依赖 weighted_mag_loss 做高频增强
        total_loss = 0.5 * complex_l1_loss + 0.2 * mag_loss + 0.5 * weighted_mag_loss

        return total_loss

        # 📝 备注：幅度加权相位损失的实现方案（未使用，但作为参考）
        # pred_phase = torch.angle(pred_fft)
        # target_phase = torch.angle(target_fft)
        # mag_weight = target_mag / (target_mag.sum(dim=-1, keepdim=True) + 1e-8)
        # cosine_sim = F.cosine_similarity(...)
        # phase_loss = (mag_weight * (1.0 - cosine_sim)).sum(dim=-1).mean()


class MultiScaleLatentLoss(nn.Module):
    """
    多尺度Latent损失 - 在不同时间尺度上计算损失

    参考: Smooth Diffusion: Crafting Smooth Latent Spaces (CVPR 2024)
    https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_Smooth_Diffusion_CVPR_2024_paper.pdf

    特性:
    - 多时间尺度 (原始 + 2x, 4x降采样)
    - 捕捉粗粒度到细粒度结构
    - 改善latent space平滑性
    - 极小计算开销
    """
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales

        # 预创建降采样层 (Average Pooling)
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=s)
            for s in scales[1:]  # 跳过scale=1
        ])

    def forward(self, pred_latent, target_latent):
        """
        Args:
            pred_latent: [B, C, T]
            target_latent: [B, C, T]
        Returns:
            loss: scalar
        """
        total_loss = 0.0

        # 1. 原始尺度 (scale=1)
        total_loss += F.l1_loss(pred_latent, target_latent)

        # 2. 降采样尺度
        for pool in self.pools:
            pred_down = pool(pred_latent)
            target_down = pool(target_latent)
            total_loss += F.l1_loss(pred_down, target_down)

        # 平均 (避免scale数量影响loss数值)
        return total_loss / len(self.scales)


class CombinedLatentPerceptualLoss(nn.Module):
    """
    组合Latent感知损失 - Frequency + Multi-Scale

    特性:
    - 频域高频增强 (针对"毛刺感")
    - 多尺度时间结构
    - 可调权重平衡
    """
    def __init__(self, freq_weight=1.0, ms_weight=0.5):
        super().__init__()
        self.freq_loss = FrequencyDomainLatentLoss(high_freq_weight=2.0)
        self.ms_loss = MultiScaleLatentLoss(scales=[1, 2, 4])
        self.freq_weight = freq_weight
        self.ms_weight = ms_weight

    def forward(self, pred_latent, target_latent):
        """
        Args:
            pred_latent: [B, C, T]
            target_latent: [B, C, T]
        Returns:
            loss: scalar
            loss_dict: dict with individual losses for logging
        """
        freq_loss = self.freq_loss(pred_latent, target_latent)
        ms_loss = self.ms_loss(pred_latent, target_latent)

        total = self.freq_weight * freq_loss + self.ms_weight * ms_loss

        # 返回组合损失 + 单独损失 (用于监控)
        loss_dict = {
            'freq_loss': freq_loss.item(),
            'ms_loss': ms_loss.item(),
            'total_latent_loss': total.item()
        }

        return total, loss_dict

# ==============================================================================
# ⚙️ 配置中心 (Configuration)
# ==============================================================================

class TrainConfig:
    # --- 硬件与并行 ---
    seed = 42
    num_workers = 16         
    
    # --- 数据 ---
    data_dir = "data_processed_v13_final"  # 指向 prepare_dataset_v5 生成的目录
    stats_file = "global_stats_separated.json"  # 分离的 HR/LR 统计量
    target_duration = 16.0                 # 切片长度 (秒)
    dac_sample_rate = 44100
    dac_hop_length = 512    # DAC 默认压缩率
    
    # 计算 Latent 帧数: 16 * 44100 / 512 ≈ 1378
    target_frames = int(target_duration * dac_sample_rate / dac_hop_length)

    # --- 模型 (V2 Specs) ---
    model_params = {
        'input_channels': 1024,
        'cond_channels': 1024,
        'patch_len': 4,
        'hidden_size': 1280,
        'depth': 28,
        'num_q_heads': 20,
        'num_kv_heads': 4,
        'bottleneck_dim': 512,
        'mlp_ratio': 4.0,
        'dropout': 0.1,         # 🔥 修复过拟合：开启Dropout
        'drop_path_rate': 0.05  # 🔥 Stochastic Depth：线性递增 0→0.05
    }

    # --- 正则化增强 (V2) ---
    condition_noise_ratio = 0.05  # Conditional Noise: 5%相对噪声
    use_adaptive_noise = True     # 使用自适应噪声（基于batch std）

    # --- Latent Perceptual Loss (MOD1 新增) ---
    use_latent_perceptual_loss = True  # 启用Latent感知损失
    latent_loss_weight = 0.3           # Latent损失权重 (相对MSE)
    freq_loss_weight = 0.5             # 频域损失权重
    ms_loss_weight = 0.5               # 多尺度损失权重
    high_freq_emphasis = 2.0           # 高频增强倍数 (1.0-3.0)

    # --- 训练超参 ---
    batch_size = 28         # 单卡 Batch (总 Batch = 60)
    lr = 5e-5               # 稳健学习率
    weight_decay = 0.1
    warmup_steps = 1000     # 预热步数
    num_epochs = 300       # 跑满为止
    grad_clip = 1.0

    # --- 保存与日志 ---
    save_dir_base = "checkpoints/v3mod1_full_run"  # MOD1基础目录
    log_dir_base = "runs/v3mod1_full_run"          # MOD1基础目录
    save_interval_steps = 1000  # 每 1000 步保存一个 interval checkpoint

# ==============================================================================
# 🛠️ 工具函数
# ==============================================================================

def get_timestamp_folder():
    """生成时间戳文件夹名（格式：MMDDHHMM）"""
    now = datetime.now()
    return now.strftime("%m%d%H%M")

def find_latest_checkpoint_dir(base_dir):
    """
    在base_dir下查找最新的时间戳子文件夹
    返回：(最新文件夹路径, checkpoint路径) 或 (None, None)
    """
    if not os.path.exists(base_dir):
        return None, None

    # 查找所有时间戳格式的子文件夹（8位数字）
    subdirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
            subdirs.append((item, item_path))

    if not subdirs:
        return None, None

    # 按时间戳排序，取最新的
    subdirs.sort(reverse=True)
    latest_folder = subdirs[0][1]

    # 检查是否有last.pt
    checkpoint_path = os.path.join(latest_folder, "last.pt")
    if os.path.exists(checkpoint_path):
        return latest_folder, checkpoint_path

    return latest_folder, None

def setup_ddp():
    """初始化 DDP 环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # 默认单机多卡
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def u_shaped_timestep_sampling(batch_size, device, alpha=0.5):
    """V2 核心: U-shaped Timestep Sampling"""
    u = torch.rand(batch_size, device=device)
    t = torch.where(
        u < 0.5,
        (2 * u) ** alpha / 2,
        1 - ((2 * (1 - u)) ** alpha) / 2
    )
    return t

def load_global_stats(stats_path, device):
    """加载并广播全局统计量"""
    with open(stats_path, 'r') as f:
        data = json.load(f)
    
    # 转换为 Tensor (1, C, 1) 用于广播
    hr_mean = torch.tensor(data['hr_mean']).view(1, -1, 1).to(device)
    hr_std = torch.tensor(data['hr_std']).view(1, -1, 1).to(device)
    lr_mean = torch.tensor(data['lr_mean']).view(1, -1, 1).to(device)
    lr_std = torch.tensor(data['lr_std']).view(1, -1, 1).to(device)
    
    return hr_mean, hr_std, lr_mean, lr_std

# ==============================================================================
# 💾 数据集 (Dataset)
# ==============================================================================

class LatentDataset(Dataset):
    """
    训练数据集：使用LRU cache缓存mmap文件引用
    由于已修复Float Bomb，现在每个样本只有~11MB，缓存完全可行
    """
    def __init__(self, data_dir, split, target_frames, samples_per_epoch_multiplier=6, cache_size=128):
        super().__init__()
        self.target_frames = target_frames
        self.multiplier = samples_per_epoch_multiplier

        # 扫描所有 .pt 文件
        search_path = Path(data_dir) / split
        self.files = sorted(list(search_path.glob("*.pt")))

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {search_path}")

        print(f"[{split}] Found {len(self.files)} files (x{self.multiplier} samples/epoch = {len(self.files) * self.multiplier} total samples)")
        print(f"[{split}] Using LRU cache (size={cache_size}, ~{cache_size * 11 / 1024:.1f}GB per worker)")

        # LRU缓存mmap的文件引用（不缓存转换后的数据）
        from functools import lru_cache
        @lru_cache(maxsize=cache_size)
        def load_file_mmap(path_str):
            """缓存mmap加载的文件（内存占用极小，只是引用）"""
            data = torch.load(path_str, map_location='cpu', mmap=True, weights_only=False)
            return data['hr_latent'], data['lr_latent']  # 保持FP16

        self._load_file = load_file_mmap

    def __len__(self):
        return len(self.files) * self.multiplier

    def __getitem__(self, idx):
        # 映射到实际文件索引
        file_idx = idx % len(self.files)
        path = self.files[file_idx]

        # 从cache加载mmap引用（FP16，几乎不占内存）
        hr, lr = self._load_file(str(path))

        length = hr.shape[-1]

        # 🔥 关键修复2: 先切片（FP16），再转换（FP32）
        if length < self.target_frames:
            # A. 短音频: 循环拼接
            repeats = math.ceil(self.target_frames / length)
            hr = hr.repeat(1, repeats)[..., :self.target_frames]
            lr = lr.repeat(1, repeats)[..., :self.target_frames]
        else:
            # B. 长音频: 随机切片
            start = random.randint(0, length - self.target_frames)
            hr = hr[..., start : start + self.target_frames]
            lr = lr[..., start : start + self.target_frames]

        # 最后才转FP32（只转换16s片段，内存占用小）
        hr = hr.float()
        lr = lr.float()

        return hr, lr

class ValidationDataset(Dataset):
    """验证集使用确定性采样，保证可重复性，带LRU cache"""
    def __init__(self, data_dir, split, target_frames, samples_per_epoch_multiplier=6, cache_size=128):
        super().__init__()
        self.target_frames = target_frames
        self.multiplier = samples_per_epoch_multiplier
        search_path = Path(data_dir) / split
        self.files = sorted(list(search_path.glob("*.pt")))

        print(f"[{split}] Found {len(self.files)} files (x{self.multiplier} samples/epoch = {len(self.files) * self.multiplier} total samples)")
        print(f"[{split}] Using LRU cache (size={cache_size}, ~{cache_size * 11 / 1024:.1f}GB per worker)")

        # LRU缓存mmap引用
        from functools import lru_cache
        @lru_cache(maxsize=cache_size)
        def load_file_mmap(path_str):
            data = torch.load(path_str, map_location='cpu', mmap=True, weights_only=False)
            return data['hr_latent'], data['lr_latent']

        self._load_file = load_file_mmap

    def __len__(self):
        return len(self.files) * self.multiplier

    def __getitem__(self, idx):
        # 映射到实际文件和采样位置
        file_idx = idx % len(self.files)
        sample_idx = idx // len(self.files)  # 0, 1, 2, ..., multiplier-1
        path = self.files[file_idx]

        # 从cache加载mmap引用
        hr, lr = self._load_file(str(path))

        length = hr.shape[-1]

        # 🔥 关键修复2: 先切片（FP16），再转换（FP32）
        if length < self.target_frames:
            # 短音频: 循环拼接
            repeats = math.ceil(self.target_frames / length)
            hr = hr.repeat(1, repeats)[..., :self.target_frames]
            lr = lr.repeat(1, repeats)[..., :self.target_frames]
        else:
            # 长音频: 确定性采样（根据sample_idx均匀分布）
            # 将音频分成 multiplier 个区间，每次从对应区间采样
            if self.multiplier == 1:
                # 只采样1次：center crop
                start = (length - self.target_frames) // 2
            else:
                # 多次采样：均匀分布
                segment_length = max(length - self.target_frames, 1)
                start = int(segment_length * sample_idx / (self.multiplier - 1))
                start = min(start, length - self.target_frames)

            hr = hr[..., start : start + self.target_frames]
            lr = lr[..., start : start + self.target_frames]

        # 最后才转FP32（只转换16s片段）
        hr = hr.float()
        lr = lr.float()

        return hr, lr

# ==============================================================================
# 🚀 训练主流程
# ==============================================================================

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='JaT V2 Training with DDP')
    parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                        help='Resume training. Use --resume for auto (latest), or --resume <path> for specific checkpoint')
    args = parser.parse_args()

    # 2. DDP 初始化
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)

    cfg = TrainConfig()

    # 3. 智能时间戳目录设置
    resume_path = None
    timestamp_folder = None
    save_dir = ""  # 初始化，避免类型检查警告
    log_dir = ""   # 初始化，避免类型检查警告

    if args.resume:
        if args.resume == 'auto':
            # 自动从最新时间戳文件夹恢复
            latest_dir, latest_ckpt = find_latest_checkpoint_dir(cfg.save_dir_base)
            if latest_ckpt:
                resume_path = latest_ckpt
                save_dir = latest_dir
                log_dir = latest_dir.replace("checkpoints", "runs")
                if is_master:
                    timestamp_folder = os.path.basename(latest_dir)
                    print(f"🔄 Auto resume from latest: {timestamp_folder}")
                    print(f"   Checkpoint: {resume_path}")
            else:
                if is_master:
                    print("⚠️  No checkpoint found, starting new training")
                timestamp_folder = get_timestamp_folder()
                save_dir = os.path.join(cfg.save_dir_base, timestamp_folder)
                log_dir = os.path.join(cfg.log_dir_base, timestamp_folder)
                if is_master:
                    print(f"📁 New training session: {timestamp_folder}")
        else:
            # 手动指定checkpoint路径
            resume_path = args.resume
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            # 从checkpoint路径提取目录
            save_dir = str(Path(resume_path).parent)
            log_dir = save_dir.replace("checkpoints", "runs")
            if is_master:
                timestamp_folder = os.path.basename(save_dir)
                print(f"📂 Manual resume from: {timestamp_folder}")
                print(f"   Checkpoint: {resume_path}")
    else:
        # 新训练，创建时间戳文件夹
        timestamp_folder = get_timestamp_folder()
        save_dir = os.path.join(cfg.save_dir_base, timestamp_folder)
        log_dir = os.path.join(cfg.log_dir_base, timestamp_folder)
        if is_master:
            print(f"📁 New training session: {timestamp_folder}")

    # 创建目录
    if is_master:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print("="*60)
        print(f"🚀 JaT V2 Training Started")
        print(f"   Save Dir: {save_dir}")
        print(f"   Log Dir: {log_dir}")
        print(f"   GPUs: {world_size} | Batch/GPU: {cfg.batch_size} | Total Batch: {cfg.batch_size * world_size}")
        print(f"   Target Duration: {cfg.target_duration}s ({cfg.target_frames} frames)")
        print(f"   Resume Mode: {'ON' if resume_path else 'OFF'}")
        print("="*60)
    
    # 3. 准备数据
    train_dataset = LatentDataset(cfg.data_dir, "train", cfg.target_frames)
    val_dataset = ValidationDataset(cfg.data_dir, "val", cfg.target_frames)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) # 验证集不需要 shuffle
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True, # 加速 Host -> Device
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 4. 加载统计量 (每个进程都加载)
    stats_path = os.path.join(cfg.data_dir, cfg.stats_file)
    hr_mean, hr_std, lr_mean, lr_std = load_global_stats(stats_path, f"cuda:{local_rank}")
    
    # 5. 模型初始化
    model = JaT_AudioSR_V2(**cfg.model_params).to(local_rank)
    
    # 6. 优化器 & 调度器
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 手动实现 Cosine with Warmup，方便状态保存
    def get_lr(step, total_steps, warmup_steps, base_lr):
        if step < warmup_steps:
            return base_lr * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # 6.5 Latent Perceptual Loss (MOD1 新增)
    latent_loss_fn = None
    if cfg.use_latent_perceptual_loss:
        latent_loss_fn = CombinedLatentPerceptualLoss(
            freq_weight=cfg.freq_loss_weight,
            ms_weight=cfg.ms_loss_weight
        ).to(local_rank)

        # 设为评估模式 (无需训练参数，仅用于梯度传递)
        latent_loss_fn.eval()

        if is_master:
            print("=" * 60)
            print("🎯 Latent Perceptual Loss Enabled (MOD1)")
            print("=" * 60)
            print(f"  Latent Loss Weight: {cfg.latent_loss_weight}")
            print(f"  Frequency Loss Weight: {cfg.freq_loss_weight}")
            print(f"  Multi-Scale Loss Weight: {cfg.ms_loss_weight}")
            print(f"  High-Freq Emphasis: {cfg.high_freq_emphasis}x")
            print("=" * 60)

    # 7. 混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # 8. 恢复训练逻辑 (Perfect Resume)
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if resume_path and os.path.exists(resume_path):
        if is_master:
            print(f"♻️  Resuming from {resume_path}...")

        # map_location 确保加载到当前 GPU
        # weights_only=False: PyTorch 2.6+需要，因为checkpoint包含RNG状态等
        checkpoint = torch.load(resume_path, map_location=f"cuda:{local_rank}", weights_only=False)

        # 处理 torch.compile() 的 _orig_mod. 前缀（兼容已编译的检查点）
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            if is_master:
                print("🔧 Removed '_orig_mod.' prefix from compiled checkpoint")

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 恢复 RNG 状态 (Python, Numpy, PyTorch, CUDA)
        if 'rng_state' in checkpoint:
            try:
                random.setstate(checkpoint['rng_state']['python'])
                np.random.set_state(checkpoint['rng_state']['numpy'])

                # PyTorch RNG 状态需要是 ByteTensor (dtype=torch.uint8)
                torch_rng = checkpoint['rng_state']['torch']
                if torch_rng.dtype != torch.uint8:
                    torch_rng = torch_rng.to(torch.uint8)
                torch.set_rng_state(torch_rng.cpu())

                # CUDA RNG 状态恢复（兼容新旧格式）
                if 'cuda_all' in checkpoint['rng_state']:
                    # 新格式：多GPU状态
                    cuda_states = checkpoint['rng_state']['cuda_all']
                    # 确保每个状态都是正确类型
                    cuda_states = [s.to(torch.uint8).cpu() if s.dtype != torch.uint8 else s.cpu()
                                   for s in cuda_states]
                    torch.cuda.set_rng_state_all(cuda_states)
                elif 'cuda' in checkpoint['rng_state']:
                    # 旧格式：单GPU状态（向后兼容）
                    cuda_state = checkpoint['rng_state']['cuda']
                    if cuda_state.dtype != torch.uint8:
                        cuda_state = cuda_state.to(torch.uint8)
                    torch.cuda.set_rng_state(cuda_state.cpu())

                if is_master:
                    print("🎲 RNG states restored")
            except Exception as e:
                if is_master:
                    print(f"⚠️  Warning: Could not restore RNG states: {e}")
                    print("   Training will continue with random initialization")
            
        if is_master:
            print(f"✅ Resumed at Epoch {start_epoch}, Step {global_step}")

    # 9. torch.compile 优化（保守模式）
    if is_master:
        print("🔥 Compiling model with torch.compile (mode='default')...")

    model = torch.compile(model, mode='default', backend='inductor')

    if is_master:
        print("✅ Model compiled successfully")

    # 10. 包装 DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # JaT无未使用参数，False更快

    # ==========================================================================
    # 🏋️ 训练循环
    # ==========================================================================
    
    total_steps_approx = len(train_loader) * cfg.num_epochs
    
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch) # DDP 必须
        
        start_time = time.time()
        epoch_loss = 0.0

        # 创建 tqdm 进度条（只在 GPU0 显示）
        pbar = tqdm(total=len(train_loader), disable=(not is_master),
                    desc=f"Epoch {epoch}/{cfg.num_epochs}",
                    leave=False, dynamic_ncols=True)

        for hr, lr in train_loader:
            # LR Scheduler Step
            current_lr = get_lr(global_step, total_steps_approx, cfg.warmup_steps, cfg.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
            # 数据搬运
            hr = hr.to(local_rank, non_blocking=True)
            lr = lr.to(local_rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                # A. 归一化 (Normalize)
                hr_norm = (hr - hr_mean) / hr_std
                lr_norm = (lr - lr_mean) / lr_std

                # A2. Conditional Noise Augmentation (防止过度依赖LR输入)
                if cfg.condition_noise_ratio > 0:
                    if cfg.use_adaptive_noise:
                        # 自适应方案：基于batch实际标准差
                        with torch.no_grad():
                            lr_batch_std = lr_norm.std().clamp(0.5, 2.0)  # 保护异常值
                    else:
                        # 简化方案：假设归一化完美，std≈1
                        lr_batch_std = 1.0

                    # 生成相对噪声并添加到LR条件
                    cond_noise = torch.randn_like(lr_norm) * (cfg.condition_noise_ratio * lr_batch_std)
                    lr_norm = lr_norm + cond_noise

                # B. 采样 Timesteps (U-shaped)
                B = hr.shape[0]
                t = u_shaped_timestep_sampling(B, hr.device) # [B]

                # C. 加噪 (Flow Matching: z_t = t*x + (1-t)*noise)
                noise = torch.randn_like(hr_norm)
                t_view = t.view(-1, 1, 1)
                z_t = t_view * hr_norm + (1 - t_view) * noise

                # D. 模型预测 (Pred x0)
                pred_x0 = model(z_t, t, lr_norm)

                # E. Loss (MSE + Latent Perceptual)
                mse_loss = F.mse_loss(pred_x0, hr_norm)

                # E2. Latent Perceptual Loss (MOD1)
                if cfg.use_latent_perceptual_loss and latent_loss_fn is not None:
                    latent_perc_loss, latent_loss_dict = latent_loss_fn(pred_x0, hr_norm)
                    # 组合损失
                    loss = mse_loss + cfg.latent_loss_weight * latent_perc_loss
                else:
                    loss = mse_loss
                    latent_loss_dict = {}

                # 📊 计算训练指标 (Phase 1 + MOD1)
                with torch.no_grad():
                    # 预测质量统计
                    pred_mean = pred_x0.mean().item()
                    pred_std = pred_x0.std().item()

                    # 信噪比 (SNR in dB)
                    signal_power = (hr_norm ** 2).mean()
                    noise_power = ((pred_x0 - hr_norm) ** 2).mean()
                    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                    snr_db = snr_db.item()

                    # Conditional Noise监控（如果启用）
                    if cfg.condition_noise_ratio > 0:
                        # 记录实际添加的噪声标准差（用于验证）
                        if cfg.use_adaptive_noise:
                            actual_noise_std = (cfg.condition_noise_ratio * lr_batch_std).item()
                        else:
                            actual_noise_std = cfg.condition_noise_ratio
            
            # 反向传播 & 优化
            scaler.scale(loss).backward()

            # 📊 计算梯度范数 (Phase 1)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_norm = grad_norm.item()
            
            scaler.step(optimizer)
            scaler.update()
            
            # 记录
            loss_val = loss.item()
            epoch_loss += loss_val

            if is_master:
                # 更新 tqdm 进度条
                pbar.set_postfix({
                    'loss': f'{loss_val:.5f}',
                    'lr': f'{current_lr:.2e}',
                    'step': global_step
                })
                pbar.update(1)

                # TensorBoard 日志 (每10步一次)
                if global_step % 10 == 0:
                    writer.add_scalar("Train/Loss", loss_val, global_step)
                    writer.add_scalar("Train/LR", current_lr, global_step)
                    # Phase 1 指标
                    writer.add_scalar("Train/GradNorm", grad_norm, global_step)
                    writer.add_scalar("Train/SNR_dB", snr_db, global_step)
                    writer.add_scalar("Train/PredictionMean", pred_mean, global_step)
                    writer.add_scalar("Train/PredictionStd", pred_std, global_step)
                    # V2 正则化指标
                    if cfg.condition_noise_ratio > 0:
                        writer.add_scalar("Train/CondNoiseStd", actual_noise_std, global_step)
                    # MOD1 Latent Perceptual Loss指标
                    if cfg.use_latent_perceptual_loss and latent_loss_dict:
                        writer.add_scalar("Train/MSE_Loss", mse_loss.item(), global_step)
                        writer.add_scalar("Train/LatentPerc_FreqLoss", latent_loss_dict['freq_loss'], global_step)
                        writer.add_scalar("Train/LatentPerc_MSLoss", latent_loss_dict['ms_loss'], global_step)
                        writer.add_scalar("Train/LatentPerc_TotalLoss", latent_loss_dict['total_latent_loss'], global_step)

                # 保存 Interval Checkpoint
                if global_step > 0 and global_step % cfg.save_interval_steps == 0:
                    save_path = os.path.join(save_dir, f"interval_step_{global_step}.pt")
                    save_checkpoint(
                        model, optimizer, scaler, epoch, global_step, best_val_loss, save_path
                    )

            global_step += 1

        # End of Epoch
        if is_master:
            pbar.close()  # 关闭进度条
            avg_epoch_loss = epoch_loss / len(train_loader)
            time_elapsed = str(timedelta(seconds=int(time.time() - start_time)))
            print(f"\n✅ Epoch {epoch} Done in {time_elapsed} | Avg Loss: {avg_epoch_loss:.5f}")
            
            # 保存 Last Checkpoint (每个 Epoch 更新)
            save_path = os.path.join(save_dir, "last.pt")
            save_checkpoint(
                model, optimizer, scaler, epoch, global_step, best_val_loss, save_path
            )

        # ======================================================================
        # 🧪 验证循环 (每个 Epoch 一次)
        # ======================================================================
        if is_master: print(f"🧪 Validating Epoch {epoch}...")

        val_loss, val_loss_std, val_latent_metrics = validate(
            model, val_loader, hr_mean, hr_std, lr_mean, lr_std, local_rank,
            cfg=cfg, latent_loss_fn=latent_loss_fn
        )

        if is_master:
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Loss_Std", val_loss_std, epoch)  # Phase 1 指标
            # MOD1: Latent Perceptual Loss验证指标
            if val_latent_metrics:
                writer.add_scalar("Val/MSE_Loss", val_latent_metrics['mse_loss'], epoch)
                writer.add_scalar("Val/LatentPerc_FreqLoss", val_latent_metrics['freq_loss'], epoch)
                writer.add_scalar("Val/LatentPerc_MSLoss", val_latent_metrics['ms_loss'], epoch)
                writer.add_scalar("Val/LatentPerc_TotalLoss", val_latent_metrics['total_latent_loss'], epoch)
            print(f"📉 Val Loss: {val_loss:.5f} ± {val_loss_std:.5f} (Best: {best_val_loss:.5f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "best.pt")
                # Best只存状态，如果硬盘够，也可以存完整
                save_checkpoint(
                    model, optimizer, scaler, epoch, global_step, best_val_loss, save_path
                )
                print(f"🏆 New Best Model Saved!")
                
        dist.barrier() # 同步等待主进程保存完毕

    cleanup_ddp()

# ==============================================================================
# 🧩 辅助功能
# ==============================================================================

def validate(model, loader, hr_mean, hr_std, lr_mean, lr_std, device, cfg=None, latent_loss_fn=None):
    """
    DDP 验证：计算所有卡的平均 Loss 和标准差
    MOD1: 支持latent perceptual loss
    返回: (avg_loss, loss_std, latent_metrics_dict)
    """
    model.eval()
    total_loss = torch.zeros(1).to(device)
    total_steps = torch.zeros(1).to(device)
    # 收集每个batch的loss用于计算标准差
    all_losses = []

    # MOD1: Latent loss累加器
    total_mse = torch.zeros(1).to(device)
    total_freq_loss = torch.zeros(1).to(device)
    total_ms_loss = torch.zeros(1).to(device)
    total_latent_loss = torch.zeros(1).to(device)

    with torch.no_grad():
        for hr, lr in loader:
            hr = hr.to(device, non_blocking=True)
            lr = lr.to(device, non_blocking=True)

            # Normalize
            hr_norm = (hr - hr_mean) / hr_std
            lr_norm = (lr - lr_mean) / lr_std

            # Fixed Timestep sampling for deterministic validation (Optional)
            # 也可以继续随机，因为数据量够大
            t = torch.rand(hr.shape[0], device=device)

            noise = torch.randn_like(hr_norm)
            t_view = t.view(-1, 1, 1)
            z_t = t_view * hr_norm + (1 - t_view) * noise

            # 验证集一定要用 autocast 保持一致性
            with torch.amp.autocast('cuda'):
                pred_x0 = model(z_t, t, lr_norm)
                mse_loss = F.mse_loss(pred_x0, hr_norm)

                # MOD1: Latent Perceptual Loss
                if cfg is not None and cfg.use_latent_perceptual_loss and latent_loss_fn is not None:
                    latent_perc_loss, latent_loss_dict = latent_loss_fn(pred_x0, hr_norm)
                    loss = mse_loss + cfg.latent_loss_weight * latent_perc_loss

                    # 累加各项指标
                    total_mse += mse_loss.detach()
                    total_freq_loss += latent_loss_dict['freq_loss']
                    total_ms_loss += latent_loss_dict['ms_loss']
                    total_latent_loss += latent_loss_dict['total_latent_loss']
                else:
                    loss = mse_loss

            total_loss += loss.detach()  # 使用detach()防止显存泄漏
            total_steps += 1
            all_losses.append(loss.item())

    # DDP All-Reduce: 汇总所有卡的 Loss
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_steps, op=dist.ReduceOp.SUM)

    # MOD1: All-Reduce latent metrics
    if cfg is not None and cfg.use_latent_perceptual_loss and latent_loss_fn is not None:
        dist.all_reduce(total_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_freq_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ms_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_latent_loss, op=dist.ReduceOp.SUM)

    avg_loss = total_loss / total_steps

    # 计算标准差 (只在当前GPU上计算，因为DDP会导致不同GPU看到不同数据)
    if len(all_losses) > 1:
        loss_std = torch.tensor(all_losses).std().item()
    else:
        loss_std = 0.0

    # MOD1: 构建latent metrics字典
    latent_metrics = {}
    if cfg is not None and cfg.use_latent_perceptual_loss and latent_loss_fn is not None:
        latent_metrics = {
            'mse_loss': (total_mse / total_steps).item(),
            'freq_loss': (total_freq_loss / total_steps).item(),
            'ms_loss': (total_ms_loss / total_steps).item(),
            'total_latent_loss': (total_latent_loss / total_steps).item()
        }

    model.train()
    return avg_loss.item(), loss_std, latent_metrics

def save_checkpoint(model, optimizer, scaler, epoch, step, best_loss, path):
    """保存完整训练状态 (Perfect Resume)"""
    # 获取原始 model (去除 DDP wrapper)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    # 移除 torch.compile() 的 _orig_mod. 前缀（如果存在）
    if any(k.startswith('_orig_mod.') for k in model_state.keys()):
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}

    # 收集 RNG 状态（多GPU环境）
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda_all': torch.cuda.get_rng_state_all()  # 保存所有GPU状态
    }
    
    state = {
        'epoch': epoch,
        'global_step': step,
        'best_val_loss': best_loss,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'rng_state': rng_state,
        'config': TrainConfig.model_params # 保存配置方便推理
    }
    
    torch.save(state, path)

if __name__ == "__main__":
    # 必须为 DDP 设置 spawn 启动方式
    # 虽然 torch.distributed.launch 也可以，但直接 python 运行更方便
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 获取可用 GPU 数量并启动多进程
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"⚠️ Warning: Found {n_gpus} GPUs. DDP meant for multi-gpu.")
    
    # 使用 torchrun 风格启动 (推荐):
    # python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
    # 或者直接运行此脚本 (会自动 fallback 到单机单卡或需要手动 torchrun)
    
    # 为了简化，这里检测如果已经是 rank 环境直接 main，否则提示
    if "LOCAL_RANK" in os.environ:
        main()
    else:
        print("请使用以下命令启动训练 (自动利用双卡):")
        print(f"torchrun --nproc_per_node={n_gpus} train_ddp.py")
#```

### 🏃‍♂️ 启动命令 (Start Command)

#由于使用了 DDP，请务必使用 `torchrun` 启动，这样 PyTorch 会自动分配 Rank 和端口：

#```bash
#cd /home/husrcf/Code/AIAA/JaT
#conda activate AIAA

# 启动双卡训练
#torchrun --nproc_per_node=2 train_ddp.py

### 🔬 监控训练 (Monitor)

#在另一个终端打开 TensorBoard：

#```bash
#tensorboard --logdir=runs/v2_full_run --port 6006 --bind_all