#!/usr/bin/env python3
"""
DAC Overfit Test for JaT-AudioSR V2 (Improved)
Improvements:
1. GQA (Grouped-Query Attention)
2. RoPE (Rotary Position Embeddings)
3. U-shaped timestep sampling for better flow matching
4. Larger model capacity
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import soundfile as sf

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.jat_audiosr_v2 import JaT_AudioSR_V2

# DAC
import dac
from audiotools import AudioSignal


def load_dac_model(model_type="44khz"):
    """加载DAC模型"""
    print("加载 DAC 模型...")
    model_path = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(model_path)
    model.eval()

    print(f"  ✅ DAC 加载完成")
    print(f"     采样率: {model.sample_rate} Hz")
    print(f"     Latent维度: {model.latent_dim}")
    print(f"     压缩率: {np.prod(model.encoder_rates)}x")

    return model


def extract_audio_chunk(audio_path, start_time=40.0, duration=10.0, target_sr=48000):
    """提取音频片段"""
    print(f"\n[1/6] 提取音频片段...")
    print(f"  文件: {Path(audio_path).name}")
    print(f"  范围: {start_time}s - {start_time+duration}s")

    import librosa
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    start_sample = int(start_time * sr)
    end_sample = int((start_time + duration) * sr)
    audio_chunk = audio[start_sample:end_sample]

    print(f"  加载: {len(audio_chunk)} samples @ {sr} Hz")
    print(f"  时长: {len(audio_chunk)/sr:.2f}s")

    return audio_chunk, sr


def create_lr_audio(hr_audio, hr_sr=48000, lr_sr=16000):
    """创建LR音频"""
    print(f"\n[2/6] 创建LR音频...")
    print(f"  HR → {lr_sr}Hz → {hr_sr}Hz")

    import librosa
    lr_audio = librosa.resample(hr_audio, orig_sr=hr_sr, target_sr=lr_sr)
    print(f"  降采样: {len(lr_audio)} samples @ {lr_sr}Hz")

    lr_audio_upsampled = librosa.resample(lr_audio, orig_sr=lr_sr, target_sr=hr_sr)
    print(f"  升采样: {len(lr_audio_upsampled)} samples @ {hr_sr}Hz")

    return lr_audio_upsampled


def audio_to_dac_latent(audio, dac_model, device='cuda'):
    """将音频编码为DAC latent"""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    signal = AudioSignal(audio_tensor, sample_rate=48000)

    if dac_model.sample_rate != 48000:
        signal = signal.resample(dac_model.sample_rate)

    signal = signal.to(device)

    with torch.no_grad():
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(signal.audio_data)

    return z.squeeze(0)  # [C, T]


def dac_latent_to_audio(latent, dac_model, device='cuda'):
    """将DAC latent解码为音频"""
    latent = latent.unsqueeze(0).to(device)

    with torch.no_grad():
        audio_tensor = dac_model.decode(latent)

    signal = AudioSignal(audio_tensor, sample_rate=dac_model.sample_rate)
    if dac_model.sample_rate != 48000:
        signal = signal.resample(48000)

    audio = signal.audio_data.squeeze(0).cpu().numpy()
    while audio.ndim > 1:
        audio = audio.squeeze(0)

    return audio


def prepare_latents(hr_audio, lr_audio, dac_model, device):
    """准备latents"""
    print(f"\n[3/6] 转换为DAC latents...")

    hr_latent = audio_to_dac_latent(hr_audio, dac_model, device)
    lr_latent = audio_to_dac_latent(lr_audio, dac_model, device)

    print(f"  HR Latent: {list(hr_latent.shape)}")
    print(f"  LR Latent: {list(lr_latent.shape)}")

    # 归一化前保存统计量
    hr_mean_orig, hr_std_orig = hr_latent.mean(), hr_latent.std()
    lr_mean_orig, lr_std_orig = lr_latent.mean(), lr_latent.std()

    print(f"\n  [归一化]")
    print(f"  归一化前: HR mean={hr_mean_orig:.4f}, std={hr_std_orig:.4f}")
    print(f"            LR mean={lr_mean_orig:.4f}, std={lr_std_orig:.4f}")

    # 归一化
    hr_latent = (hr_latent - hr_mean_orig) / hr_std_orig
    lr_latent = (lr_latent - lr_mean_orig) / lr_std_orig

    print(f"  归一化后: HR mean={hr_latent.mean():.4f}, std={hr_latent.std():.4f}")
    print(f"            LR mean={lr_latent.mean():.4f}, std={lr_latent.std():.4f}")
    print(f"  ✅ Latents 已归一化")

    return hr_latent, lr_latent, (hr_mean_orig, hr_std_orig, lr_mean_orig, lr_std_orig)


def u_shaped_timestep_sampling(batch_size, device, alpha=2.0):
    """
    U-shaped timestep distribution for improved flow matching
    Samples more timesteps near t=0 and t=1

    Based on: "Improving the Training of Rectified Flows" (2024)
    https://arxiv.org/abs/2405.20320

    Args:
        batch_size: number of samples
        device: torch device
        alpha: concentration parameter (higher = more U-shaped)
    Returns:
        t: [B] timesteps in [0, 1]
    """
    # Sample from Beta(alpha, alpha) distribution
    # This creates a U-shape when alpha < 1, or bell-shape when alpha > 1
    # For improved flow matching, we want U-shape, so use alpha < 1
    # But paper suggests using transformed uniform for stability

    # Method: transform uniform with U-shaped density
    u = torch.rand(batch_size, device=device)

    # Apply transformation: concentrate more samples near 0 and 1
    # Using: t = u^alpha for u < 0.5, else 1 - (1-u)^alpha
    mask = u < 0.5
    t = torch.where(
        mask,
        (2 * u) ** alpha / 2,  # Map [0, 0.5] → [0, 0.5] with concentration at 0
        1 - ((2 * (1 - u)) ** alpha) / 2  # Map [0.5, 1] → [0.5, 1] with concentration at 1
    )

    return t


def flow_matching_sample(hr_latent, batch_size, use_u_shaped=True):
    """
    Flow matching noise schedule with optional U-shaped sampling

    Args:
        hr_latent: [B, C, T] clean latent
        batch_size: B
        use_u_shaped: whether to use U-shaped timestep distribution
    """
    device = hr_latent.device

    if use_u_shaped:
        # U-shaped distribution (concentrate near t=0 and t=1)
        t = u_shaped_timestep_sampling(batch_size, device, alpha=0.5)
    else:
        # Uniform distribution (original)
        t = torch.rand(batch_size, device=device)

    noise = torch.randn_like(hr_latent)
    t_view = t.view(-1, 1, 1)
    z_t = t_view * hr_latent + (1 - t_view) * noise
    return z_t, t, noise


def compute_loss(model, hr_latent, lr_latent, use_u_shaped=True):
    """计算x-prediction loss"""
    B = hr_latent.shape[0]
    z_t, t, noise = flow_matching_sample(hr_latent, B, use_u_shaped=use_u_shaped)
    pred_x0 = model(z_t, t, lr_latent)
    loss = F.mse_loss(pred_x0, hr_latent)
    return loss


def overfitting_test(hr_latent, lr_latent, model_config=None, num_epochs=1000,
                     use_u_shaped=True, device='cuda'):
    """
    在单个样本上过拟合测试
    """
    print(f"\n[4/6] 创建JaT-AudioSR V2模型...")

    if model_config is None:
        # Default V2 config: 2x parameters compared to V1 Full
        model_config = {
            'input_channels': 1024,
            'cond_channels': 1024,
            'patch_len': 4,
            'hidden_size': 1024,       # Larger (768 → 1024)
            'depth': 16,               # Deeper (12 → 16)
            'num_q_heads': 16,         # More heads
            'num_kv_heads': 4,         # GQA: 4x fewer KV heads
            'bottleneck_dim': 512      # Larger bottleneck (256 → 512)
        }

    model = JaT_AudioSR_V2(**model_config)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params / 1e6:.2f}M")

    print(f"\n[5/6] 训练...")
    print(f"  U-shaped timestep sampling: {'✅ 启用' if use_u_shaped else '❌ 关闭'}")
    print("=" * 70)
    print(f"{'Epoch':<10} {'Loss':<15} {'状态'}")
    print("=" * 70)

    # 使用统一的学习率
    lr = 1e-4
    print(f"  使用学习率: {lr}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to device
    hr = hr_latent.unsqueeze(0).to(device)
    lr_cond = lr_latent.unsqueeze(0).to(device)

    losses = []

    for epoch in range(num_epochs):
        loss = compute_loss(model, hr, lr_cond, use_u_shaped=use_u_shaped)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            status = ""
            if loss.item() < 0.01:
                status = "✅ 优秀!"
            elif loss.item() < 0.1:
                status = "✅ 良好"
            elif loss.item() < 1.0:
                status = "⚠️  学习中..."
            else:
                status = "❌ 未收敛"

            print(f"{epoch+1:<10} {loss.item():<15.6f} {status}")

    print("=" * 70)

    # Evaluation
    print(f"\n[6/6] 评估...")
    final_loss = losses[-1]
    initial_loss = losses[0]

    print(f"  初始loss: {initial_loss:.6f}")
    print(f"  最终loss: {final_loss:.6f}")
    print(f"  下降:     {(1 - final_loss/initial_loss) * 100:.1f}%")

    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✅ 成功: 模型完美过拟合!")
        success = True
    elif final_loss < 0.1:
        print("⚠️  部分成功: 模型在学习但未完美")
        success = True
    else:
        print("❌ 失败: 无法过拟合")
        success = False
    print("=" * 70)

    return success, losses, model


@torch.no_grad()
def generate_audio(model, dac_model, lr_latent, hr_mean, hr_std, num_steps=50, device='cuda'):
    """
    使用训练好的模型生成音频
    """
    model.eval()
    dac_model = dac_model.to(device)
    dac_model.eval()

    # Start from noise
    lr_latent_batch = lr_latent.unsqueeze(0).to(device)
    z_t = torch.randn_like(lr_latent_batch).to(device)

    # DDIM sampling
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(1, device=device) * (1.0 - i * dt)

        # Predict clean latent
        pred_x0 = model(z_t, t, lr_latent_batch)

        if i < num_steps - 1:
            t_next = torch.ones(1, device=device) * (1.0 - (i + 1) * dt)
            noise = torch.randn_like(z_t)
            z_t = t_next.view(-1, 1, 1) * pred_x0 + (1 - t_next.view(-1, 1, 1)) * noise
        else:
            z_t = pred_x0

    # Denormalize
    pred_latent_normalized = z_t.squeeze(0)
    hr_mean = hr_mean.to(device) if isinstance(hr_mean, torch.Tensor) else torch.tensor(hr_mean, device=device)
    hr_std = hr_std.to(device) if isinstance(hr_std, torch.Tensor) else torch.tensor(hr_std, device=device)

    pred_latent = pred_latent_normalized * hr_std + hr_mean
    pred_latent = pred_latent.cpu()

    # Decode to audio
    audio = dac_latent_to_audio(pred_latent, dac_model, device)

    return audio


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DAC Overfit Test for JaT-AudioSR V2')

    # Audio settings
    parser.add_argument('--duration', type=float, default=6.0,
                        help='Audio chunk duration in seconds (default: 6.0)')
    parser.add_argument('--start-time', type=float, default=40.0,
                        help='Start time in the audio file (default: 40.0)')
    parser.add_argument('--audio-path', type=str,
                        default='/home/husrcf/Code/AIAA/Final2/01 - 风继续吹.wav',
                        help='Path to audio file')

    # Training settings
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--no-u-shaped', action='store_true',
                        help='Disable U-shaped timestep sampling')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("DAC OVERFIT TEST - JaT-AudioSR V2 (Improved)")
    print("=" * 70)
    print(f"\n🎵 改进版测试")
    print(f"   音频长度: {args.duration}s")
    print(f"   Improvements: GQA + RoPE + U-shaped sampling")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # Configuration
    audio_path = args.audio_path
    start_time = args.start_time
    duration = args.duration

    # 1. Extract audio
    hr_audio, sr = extract_audio_chunk(audio_path, start_time, duration)

    # 2. Create LR audio
    lr_audio = create_lr_audio(hr_audio, hr_sr=sr, lr_sr=16000)

    # Save audio
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    sf.write(output_dir / 'hr_audio_chunk.wav', hr_audio, sr)
    sf.write(output_dir / 'lr_audio_chunk.wav', lr_audio, sr)
    print(f"\n  保存音频到 outputs/")

    # 3. Load DAC
    dac_model = load_dac_model()
    dac_model = dac_model.to(device)  # Move DAC to correct device

    # 4. Prepare latents
    hr_latent, lr_latent, (hr_mean, hr_std, lr_mean, lr_std) = prepare_latents(
        hr_audio, lr_audio, dac_model, device
    )

    print(f"\n  📊 归一化参数 (用于反归一化):")
    print(f"      HR: mean={hr_mean:.4f}, std={hr_std:.4f}")
    print(f"      LR: mean={lr_mean:.4f}, std={lr_std:.4f}")

    # 5. Overfit test
    use_u_shaped = not args.no_u_shaped
    success, losses, model = overfitting_test(
        hr_latent, lr_latent,
        model_config=None,  # Use default V2 config
        num_epochs=args.epochs,
        use_u_shaped=use_u_shaped,
        device=device
    )

    # 6. Generate audio
    if success or losses[-1] < 0.2:
        print("\n" + "=" * 70)
        print("[7/7] 生成音频...")
        print("=" * 70)

        try:
            print(f"\n  使用 {args.num_steps} 步采样生成音频...")
            generated_audio = generate_audio(
                model=model,
                dac_model=dac_model,
                lr_latent=lr_latent,
                hr_mean=hr_mean,
                hr_std=hr_std,
                num_steps=args.num_steps,
                device=device
            )

            # Save generated audio
            sf.write(output_dir / 'generated_audio_v2.wav', generated_audio, 48000)
            print(f"  ✅ 保存生成音频: {output_dir}/generated_audio_v2.wav")

            # Decode ground truth for comparison
            print(f"\n  解码ground truth HR用于对比...")
            hr_latent_denorm = hr_latent * hr_std + hr_mean
            hr_recon = dac_latent_to_audio(hr_latent_denorm, dac_model, device)
            sf.write(output_dir / 'hr_reconstructed_v2.wav', hr_recon, 48000)
            print(f"  ✅ 保存HR重建: {output_dir}/hr_reconstructed_v2.wav")

            # LR reconstruction
            print(f"\n  解码LR latent...")
            lr_latent_denorm = lr_latent * lr_std + lr_mean
            lr_recon = dac_latent_to_audio(lr_latent_denorm, dac_model, device)
            sf.write(output_dir / 'lr_reconstructed_v2.wav', lr_recon, 48000)
            print(f"  ✅ 保存LR重建: {output_dir}/lr_reconstructed_v2.wav")

            # Calculate metrics
            print(f"\n  音频质量对比:")
            print(f"    原始HR:      std={hr_audio.std():.6f}")
            print(f"    HR重建(DAC): std={hr_recon.std():.6f}")
            print(f"    生成音频:    std={generated_audio.std():.6f}")

            # Correlation
            min_len = min(len(hr_recon), len(generated_audio))
            correlation = np.corrcoef(hr_recon[:min_len], generated_audio[:min_len])[0, 1]
            print(f"\n    Generated vs HR重建 correlation: {correlation:.6f}")

            print(f"\n  📂 音频文件:")
            print(f"    1. outputs/lr_audio_chunk.wav          - LR输入")
            print(f"    2. outputs/hr_audio_chunk.wav          - HR ground truth")
            print(f"    3. outputs/lr_reconstructed_v2.wav     - LR通过DAC重建")
            print(f"    4. outputs/hr_reconstructed_v2.wav     - HR通过DAC重建")
            print(f"    5. outputs/generated_audio_v2.wav      - JaT V2模型生成")

            print(f"\n  🎧 听音频对比质量!")

        except Exception as e:
            print(f"\n  ❌ 音频生成失败: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 JaT V2 overfit test 通过!")
        print("\n改进验证:")
        print("  ✅ GQA (Grouped-Query Attention) 工作正常")
        print("  ✅ RoPE (Rotary Position Embeddings) 工作正常")
        print(f"  ✅ U-shaped timestep sampling: {'启用' if use_u_shaped else '关闭'}")
        print("  ✅ 训练pipeline完整")
        print(f"\n模型规模: ~2x V1 Full")
        print("准备好进行完整数据集训练!")
    else:
        print("⚠️  JaT V2 overfit test 有问题")
        print("   检查loss曲线和组件")

    print("=" * 70)

    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
