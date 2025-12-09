"""
JaT-AudioSR V3 推理脚本（支持自动切片+拼接）

特性：
- 固定16秒chunk（与训练对齐）
- 自动切片长音频
- 2秒overlap + 线性crossfade拼接
- 支持指定总输出长度
"""

import torch
import torchaudio
import os
import sys
import json
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.jat_audiosr_v2 import JaT_AudioSR_V2
import dac


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"📦 Loading checkpoint from: {checkpoint_path}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从checkpoint获取模型配置
    model_config = checkpoint.get('config', {
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
        'drop_path_rate': 0.05
    })

    # 创建模型
    model = JaT_AudioSR_V2(**model_config).to(device)

    # 加载权重（处理可能的兼容性问题）
    state_dict = checkpoint['model_state_dict']

    # 1. 移除 torch.compile 前缀
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # 2. 处理 Dropout 层索引变化（旧checkpoint没有Dropout）
    # 旧模型: mlp = [Linear(0), GELU(1), Linear(2)]
    # 新模型: mlp = [Linear(0), GELU(1), Dropout(2), Linear(3), Dropout(4)]
    # 需要将旧的 mlp.2.* 映射到新的 mlp.3.*
    new_state_dict = {}
    for k, v in state_dict.items():
        # 检查是否是 blocks.*.mlp.2.* (旧的最后一层Linear)
        if '.mlp.2.weight' in k or '.mlp.2.bias' in k:
            # 映射到 mlp.3.*
            k = k.replace('.mlp.2.', '.mlp.3.')
            print(f"  Remapping: mlp.2 → mlp.3 (adding Dropout compatibility)")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('global_step', 0)
    print(f"✅ Model loaded (Epoch {epoch}, Step {step})")

    return model


def load_dac_codec(device='cuda'):
    """加载DAC编解码器"""
    print("📦 Loading DAC codec (44.1kHz)...")
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path).to(device)
    dac_model.eval()
    print("✅ DAC codec loaded")
    return dac_model


@torch.no_grad()
def flow_matching_sample(model, lr_latent, num_steps=50, device='cuda', verbose=True):
    """
    Flow Matching ODE采样（Euler方法）
    使用 x-prediction (预测干净的 x_0)

    训练时插值公式: z_t = t * x_0 + (1-t) * noise
    - t=0: 纯噪声
    - t=1: 干净数据

    Args:
        model: JaT模型
        lr_latent: [1, C, T] LR条件latent (已归一化)
        num_steps: ODE求解步数
        device: 设备
        verbose: 是否打印进度
    Returns:
        [1, C, T] 生成的HR latent (已归一化)
    """
    if verbose:
        print(f"  🎲 Flow Matching sampling ({num_steps} steps)...")

    B, C, T = lr_latent.shape

    # 初始化：从纯噪声开始 (t=0)
    z_t = torch.randn(B, C, T, device=device)

    # 时间步: 0.0 → 1.0 (从噪声走向干净数据)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr  # 正数，从0到1

        # 模型输入：当前时间步
        t_batch = torch.full((B,), t_curr, device=device)

        # 模型预测 x_0 (干净的HR latent)
        x_pred = model(z_t, t_batch, lr_latent)

        # Flow Matching ODE推导:
        # z_t = t * x_0 + (1-t) * noise
        # dz/dt = x_0 - noise = x_0 - (z_t - t*x_0)/(1-t) = (x_0 - z_t) / (1-t)

        if t_curr < 0.999:
            # 正常ODE步进
            velocity = (x_pred - z_t) / (1 - t_curr + 1e-5)
            z_t = z_t + velocity * dt
        else:
            # t接近1时，直接使用预测值（避免除以0）
            z_t = x_pred

        # 打印进度
        if verbose and ((i + 1) % 10 == 0 or i == num_steps - 1):
            print(f"    Step {i+1}/{num_steps}, t={t_curr:.3f} → {t_next:.3f}")

    return z_t


def crossfade_chunks(chunks, overlap_frames):
    """
    使用线性crossfade拼接多个chunk

    Args:
        chunks: List of [1, C, T] tensors
        overlap_frames: overlap区域的帧数
    Returns:
        [1, C, total_T] 拼接后的tensor
    """
    if len(chunks) == 0:
        return None
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]

    for i in range(1, len(chunks)):
        prev_chunk = result
        curr_chunk = chunks[i]

        if overlap_frames > 0 and prev_chunk.shape[-1] >= overlap_frames:
            # 提取overlap区域
            prev_overlap = prev_chunk[..., -overlap_frames:]  # [1, C, overlap]
            curr_overlap = curr_chunk[..., :overlap_frames]   # [1, C, overlap]

            # 线性crossfade权重
            fade_out = torch.linspace(1.0, 0.0, overlap_frames, device=prev_chunk.device)
            fade_in = torch.linspace(0.0, 1.0, overlap_frames, device=curr_chunk.device)
            fade_out = fade_out.view(1, 1, -1)
            fade_in = fade_in.view(1, 1, -1)

            # Crossfade
            blended = prev_overlap * fade_out + curr_overlap * fade_in

            # 拼接：prev[:-overlap] + blended + curr[overlap:]
            result = torch.cat([
                prev_chunk[..., :-overlap_frames],
                blended,
                curr_chunk[..., overlap_frames:]
            ], dim=-1)
        else:
            # 无overlap，直接拼接
            result = torch.cat([prev_chunk, curr_chunk], dim=-1)

    return result


def main():
    parser = argparse.ArgumentParser(description='JaT-AudioSR V2 Inference Test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v2_full_run/last.pt',
                        help='Checkpoint path')
    parser.add_argument('--val-dir', type=str, default='data_processed_v13_final/val',
                        help='Validation latents directory')
    parser.add_argument('--stats-file', type=str, default='data_processed_v13_final/global_stats_separated.json',
                        help='Normalization stats file (JSON or PT)')
    parser.add_argument('--output-dir', type=str, default='inference_output',
                        help='Output directory')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--total-seconds', type=float, default=60.0,
                        help='Total output duration in seconds (None=use full input length)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Specific input file (e.g., "04 - 打开信箱.pt"). If not specified, uses first file in val-dir')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("JaT-AudioSR V3 推理测试")
    print("=" * 60)

    # 1. 加载模型
    model = load_model(args.checkpoint, device=device)

    # 2. 加载DAC编解码器
    dac_codec = load_dac_codec(device=device)

    # 3. 加载验证集文件
    if args.input_file:
        # 使用指定的文件
        input_path = Path(args.val_dir) / args.input_file
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            print(f"Available files in {args.val_dir}:")
            for f in sorted(Path(args.val_dir).glob("*.pt"))[:10]:
                print(f"  - {f.name}")
            return
        first_file = input_path
        print(f"\n📂 Loading specified file: {first_file.name}")
    else:
        # 使用第一个文件（默认行为）
        val_files = sorted(Path(args.val_dir).glob("*.pt"))
        if len(val_files) == 0:
            print(f"❌ No files found in {args.val_dir}")
            return
        first_file = val_files[0]
        print(f"\n📂 Loading validation file (first): {first_file.name}")

    data = torch.load(first_file, map_location='cpu', mmap=True, weights_only=False)
    hr_latent = data['hr_latent'].float()  # [1024, T]
    lr_latent = data['lr_latent'].float()  # [1024, T]

    # 归一化参数
    print(f"\n📊 Loading normalization stats from: {args.stats_file}")

    # 根据文件扩展名加载
    if args.stats_file.endswith('.json'):
        # JSON格式
        with open(args.stats_file, 'r') as f:
            stats = json.load(f)
        hr_mean = torch.tensor(stats['hr_mean']).view(1, -1, 1).to(device)
        hr_std = torch.tensor(stats['hr_std']).view(1, -1, 1).to(device)
        lr_mean = torch.tensor(stats['lr_mean']).view(1, -1, 1).to(device)
        lr_std = torch.tensor(stats['lr_std']).view(1, -1, 1).to(device)
    else:
        # PyTorch格式
        stats = torch.load(args.stats_file, weights_only=False)

        if 'hr_mean' in stats:
            # 格式1: 直接的 mean/std
            hr_mean = stats['hr_mean'].view(1, -1, 1).to(device)
            hr_std = stats['hr_std'].view(1, -1, 1).to(device)
            lr_mean = stats['lr_mean'].view(1, -1, 1).to(device)
            lr_std = stats['lr_std'].view(1, -1, 1).to(device)
        elif 'sum' in stats:
            # 格式2: 累积统计量 (sum, sq_sum, count)
            print("  Computing mean/std from running stats...")
            count = stats['count']
            mean = stats['sum'] / count
            var = (stats['sq_sum'] / count) - (mean ** 2)
            std = torch.sqrt(var + 1e-8)

            # 假设前1024维是HR，后1024维是LR
            hr_mean = mean[:1024].view(1, -1, 1).to(device)
            hr_std = std[:1024].view(1, -1, 1).to(device)
            lr_mean = mean[1024:].view(1, -1, 1).to(device)
            lr_std = std[1024:].view(1, -1, 1).to(device)
        else:
            raise ValueError(f"Unknown stats format. Keys: {list(stats.keys())}")

    print(f"  HR mean: {hr_mean.mean().item():.4f}, std: {hr_std.mean().item():.4f}")
    print(f"  LR mean: {lr_mean.mean().item():.4f}, std: {lr_std.mean().item():.4f}")

    print(f"  Original length: {hr_latent.shape[-1]} frames")

    # 4. 自动切片处理（与训练对齐：16秒chunk + 2秒overlap）
    dac_sr = 44100
    dac_hop = 512
    CHUNK_DURATION = 16.0  # 固定16秒，与训练对齐
    OVERLAP_DURATION = 2.0  # 2秒overlap用于crossfade

    chunk_frames = int(CHUNK_DURATION * dac_sr / dac_hop)  # 1378 frames
    overlap_frames = int(OVERLAP_DURATION * dac_sr / dac_hop)  # ~172 frames

    # 计算需要处理的总帧数
    total_frames = hr_latent.shape[-1]
    if args.total_seconds is not None:
        total_frames = min(total_frames, int(args.total_seconds * dac_sr / dac_hop))

    print(f"\n🎯 Processing configuration:")
    print(f"  Chunk size: {CHUNK_DURATION}s ({chunk_frames} frames)")
    print(f"  Overlap: {OVERLAP_DURATION}s ({overlap_frames} frames)")
    print(f"  Total to process: {total_frames} frames ({total_frames * dac_hop / dac_sr:.1f}s)")

    # 计算需要多少个chunk
    stride = chunk_frames - overlap_frames
    num_chunks = (total_frames - overlap_frames + stride - 1) // stride
    print(f"  Number of chunks: {num_chunks}")

    # 5. 逐个chunk处理
    print("\n🎨 Generating HR audio (chunk by chunk)...")
    generated_chunks = []
    hr_chunks_list = []
    lr_chunks_list = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * stride
        end = min(start + chunk_frames, total_frames)

        print(f"\n📦 Chunk {chunk_idx + 1}/{num_chunks} (frames {start}:{end})")

        # 提取chunk
        hr_chunk = hr_latent[:, start:end].unsqueeze(0).to(device)  # [1, 1024, T]
        lr_chunk = lr_latent[:, start:end].unsqueeze(0).to(device)  # [1, 1024, T]

        # 归一化
        hr_chunk_norm = (hr_chunk - hr_mean) / hr_std
        lr_chunk_norm = (lr_chunk - lr_mean) / lr_std

        # Flow Matching采样
        gen_chunk = flow_matching_sample(
            model, lr_chunk_norm, num_steps=args.steps, device=device, verbose=True
        )

        # 反归一化
        gen_chunk = gen_chunk * hr_std + hr_mean

        generated_chunks.append(gen_chunk)
        hr_chunks_list.append(hr_chunk)
        lr_chunks_list.append(lr_chunk)

    # 6. Crossfade拼接
    print(f"\n🔗 Crossfading {len(generated_chunks)} chunks...")
    generated_latent = crossfade_chunks(generated_chunks, overlap_frames)
    hr_chunk = crossfade_chunks(hr_chunks_list, overlap_frames)
    lr_chunk = crossfade_chunks(lr_chunks_list, overlap_frames)

    print(f"  Final length: {generated_latent.shape[-1]} frames ({generated_latent.shape[-1] * dac_hop / dac_sr:.1f}s)")

    # 8. DAC解码
    print("\n🔊 Decoding with DAC...")

    # 解码生成的音频
    with torch.no_grad():
        # DAC expects [B, latent_dim, n_codebooks, T]
        # 但我们的latent是 [B, 1024, T]，1024 = 72 codebooks * 1024 per book?
        # 实际上DAC的latent维度应该是 [B, n_codebooks, T]
        # 检查DAC的输入格式

        # 将 [1, 1024, T] reshape 为 DAC格式
        # DAC 44.1kHz: n_codebooks=9, codebook_size=1024
        # 所以 generated_latent 应该已经是正确的格式了

        generated_audio = dac_codec.decode(generated_latent)  # [1, 1, samples]
        hr_audio = dac_codec.decode(hr_chunk)  # [1, 1, samples]
        lr_audio = dac_codec.decode(lr_chunk)  # [1, 1, samples]

    generated_audio = generated_audio.squeeze(0).cpu()  # [1, samples]
    hr_audio = hr_audio.squeeze(0).cpu()
    lr_audio = lr_audio.squeeze(0).cpu()

    # 9. 保存音频
    print("\n💾 Saving results...")

    output_prefix = first_file.stem
    torchaudio.save(
        f"{args.output_dir}/{output_prefix}_generated.wav",
        generated_audio, dac_sr
    )
    torchaudio.save(
        f"{args.output_dir}/{output_prefix}_hr_gt.wav",
        hr_audio, dac_sr
    )
    torchaudio.save(
        f"{args.output_dir}/{output_prefix}_lr_input.wav",
        lr_audio, dac_sr
    )

    print(f"✅ Saved to {args.output_dir}/")
    print(f"   - {output_prefix}_generated.wav (模型生成)")
    print(f"   - {output_prefix}_hr_gt.wav (Ground Truth)")
    print(f"   - {output_prefix}_lr_input.wav (LR输入)")
    print("=" * 60)


if __name__ == '__main__':
    main()
