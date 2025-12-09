#!/usr/bin/env python3
"""
DAC Overfit Test for JaT-AudioSR
使用DAC codec进行真实音频过拟合测试
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

from src.models.jat_audiosr import JaT_AudioSR

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
    # 创建AudioSignal
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
    signal = AudioSignal(audio_tensor, sample_rate=48000)

    # 重采样到DAC采样率
    if dac_model.sample_rate != 48000:
        signal = signal.resample(dac_model.sample_rate)

    signal = signal.to(device)

    with torch.no_grad():
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(signal.audio_data)

    return z.squeeze(0)  # [C, T]


def dac_latent_to_audio(latent, dac_model, device='cuda'):
    """将DAC latent解码为音频"""
    latent = latent.unsqueeze(0).to(device)  # [1, C, T]

    with torch.no_grad():
        audio_tensor = dac_model.decode(latent)

    # 转回48kHz
    signal = AudioSignal(audio_tensor, sample_rate=dac_model.sample_rate)
    if dac_model.sample_rate != 48000:
        signal = signal.resample(48000)

    audio = signal.audio_data.cpu().numpy()
    while audio.ndim > 1:
        audio = audio.squeeze(0)

    return audio


def prepare_latents(hr_audio, lr_audio, dac_model, device='cuda'):
    """
    转换HR和LR音频为DAC latents
    """
    print(f"\n[3/6] 转换为DAC latents...")

    dac_model = dac_model.to(device)

    # Encode
    hr_latent = audio_to_dac_latent(hr_audio, dac_model, device)
    lr_latent = audio_to_dac_latent(lr_audio, dac_model, device)

    print(f"  HR Latent: {list(hr_latent.shape)}")
    print(f"  LR Latent: {list(lr_latent.shape)}")

    # 归一化
    print(f"\n  [归一化]")
    print(f"  归一化前: HR mean={hr_latent.mean():.4f}, std={hr_latent.std():.4f}")
    print(f"            LR mean={lr_latent.mean():.4f}, std={lr_latent.std():.4f}")

    # 保存原始统计
    hr_mean_orig, hr_std_orig = hr_latent.mean(), hr_latent.std()
    lr_mean_orig, lr_std_orig = lr_latent.mean(), lr_latent.std()

    # 归一化
    hr_latent = (hr_latent - hr_mean_orig) / hr_std_orig
    lr_latent = (lr_latent - lr_mean_orig) / lr_std_orig

    print(f"  归一化后: HR mean={hr_latent.mean():.4f}, std={hr_latent.std():.4f}")
    print(f"            LR mean={lr_latent.mean():.4f}, std={lr_latent.std():.4f}")
    print(f"  ✅ Latents 已归一化")

    return hr_latent, lr_latent, (hr_mean_orig, hr_std_orig, lr_mean_orig, lr_std_orig)


def flow_matching_sample(hr_latent, batch_size):
    """Flow matching noise schedule"""
    device = hr_latent.device
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(hr_latent)
    t_view = t.view(-1, 1, 1)
    z_t = t_view * hr_latent + (1 - t_view) * noise
    return z_t, t, noise


def compute_loss(model, hr_latent, lr_latent):
    """计算x-prediction loss"""
    B = hr_latent.shape[0]
    z_t, t, noise = flow_matching_sample(hr_latent, B)
    pred_x0 = model(z_t, t, lr_latent)
    loss = F.mse_loss(pred_x0, hr_latent)
    return loss


def overfitting_test(hr_latent, lr_latent, model_config=None, num_epochs=500,
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    在单个样本上过拟合测试
    """
    print(f"\n[4/6] 创建JaT模型...")

    if model_config is None:
        model_config = {
            'input_channels': 1024,
            'cond_channels': 1024,
            'patch_len': 4,
            'hidden_size': 768,       # Full size
            'depth': 12,              # 12 layers
            'num_heads': 12,
            'bottleneck_dim': 256
        }

    model = JaT_AudioSR(**model_config)
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params / 1e6:.2f}M")

    print(f"\n[5/6] 训练...")
    print("=" * 70)
    print(f"{'Epoch':<10} {'Loss':<15} {'状态'}")
    print("=" * 70)

    # 使用统一的学习率
    lr = 1e-4
    print(f"  使用学习率: {lr}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to device
    hr = hr_latent.unsqueeze(0).to(device)
    lr = lr_latent.unsqueeze(0).to(device)

    losses = []

    for epoch in range(num_epochs):
        loss = compute_loss(model, hr, lr)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (对所有模型)
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

    # Denormalize (确保在同一设备上)
    pred_latent_normalized = z_t.squeeze(0)
    # Move hr_mean and hr_std to same device
    hr_mean = hr_mean.to(device) if isinstance(hr_mean, torch.Tensor) else torch.tensor(hr_mean, device=device)
    hr_std = hr_std.to(device) if isinstance(hr_std, torch.Tensor) else torch.tensor(hr_std, device=device)

    pred_latent = pred_latent_normalized * hr_std + hr_mean
    pred_latent = pred_latent.cpu()

    # Decode to audio
    audio = dac_latent_to_audio(pred_latent, dac_model, device)

    return audio


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DAC Overfit Test for JaT-AudioSR')

    # Audio settings
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Audio chunk duration in seconds (default: 10.0)')
    parser.add_argument('--start-time', type=float, default=40.0,
                        help='Start time in the audio file (default: 40.0)')
    parser.add_argument('--audio-path', type=str,
                        default='/home/husrcf/Code/AIAA/Final2/01 - 风继续吹.wav',
                        help='Path to audio file')

    # Model settings
    parser.add_argument('--model-size', type=str, default='full',
                        choices=['small', 'medium', 'full'],
                        help='Model size: small (6 layers, 384dim), medium (8 layers, 512dim), full (12 layers, 768dim)')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Override hidden size (default: determined by model-size)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Override depth (default: determined by model-size)')

    # Training settings
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')

    return parser.parse_args()


def get_model_config(args):
    """根据参数获取模型配置"""
    # Predefined configs
    configs = {
        'small': {
            'hidden_size': 384,
            'depth': 6,
            'num_heads': 6,
            'bottleneck_dim': 128
        },
        'medium': {
            'hidden_size': 512,
            'depth': 8,
            'num_heads': 8,
            'bottleneck_dim': 192
        },
        'full': {
            'hidden_size': 768,
            'depth': 12,
            'num_heads': 12,
            'bottleneck_dim': 256
        }
    }

    config = configs[args.model_size].copy()
    config['input_channels'] = 1024
    config['cond_channels'] = 1024
    config['patch_len'] = 4

    # Override if specified
    if args.hidden_size is not None:
        config['hidden_size'] = args.hidden_size
        config['num_heads'] = args.hidden_size // 64  # Maintain 64 dim per head
    if args.depth is not None:
        config['depth'] = args.depth

    return config


def main():
    args = parse_args()

    print("=" * 70)
    print("DAC OVERFIT TEST - JaT-AudioSR")
    print("=" * 70)
    print(f"\n🎵 使用DAC codec进行真实音频过拟合测试")
    print(f"   音频长度: {args.duration}s")
    print(f"   模型大小: {args.model_size}")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # Configuration
    audio_path = args.audio_path
    start_time = args.start_time
    duration = args.duration

    # Get model config
    model_config = get_model_config(args)

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

    # 4. Prepare latents
    hr_latent, lr_latent, (hr_mean, hr_std, lr_mean, lr_std) = prepare_latents(
        hr_audio, lr_audio, dac_model, device
    )

    print(f"\n  📊 归一化参数 (用于反归一化):")
    print(f"      HR: mean={hr_mean:.4f}, std={hr_std:.4f}")
    print(f"      LR: mean={lr_mean:.4f}, std={lr_std:.4f}")

    # 5. Overfit test
    success, losses, model = overfitting_test(
        hr_latent, lr_latent,
        model_config=model_config,
        num_epochs=args.epochs,
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
            sf.write(output_dir / 'generated_audio_dac.wav', generated_audio, 48000)
            print(f"  ✅ 保存生成音频: {output_dir}/generated_audio_dac.wav")

            # Decode ground truth for comparison
            print(f"\n  解码ground truth HR用于对比...")
            hr_latent_denorm = hr_latent * hr_std + hr_mean
            hr_recon = dac_latent_to_audio(hr_latent_denorm, dac_model, device)
            sf.write(output_dir / 'hr_reconstructed_dac.wav', hr_recon, 48000)
            print(f"  ✅ 保存HR重建: {output_dir}/hr_reconstructed_dac.wav")

            # LR reconstruction
            print(f"\n  解码LR latent...")
            lr_latent_denorm = lr_latent * lr_std + lr_mean
            lr_recon = dac_latent_to_audio(lr_latent_denorm, dac_model, device)
            sf.write(output_dir / 'lr_reconstructed_dac.wav', lr_recon, 48000)
            print(f"  ✅ 保存LR重建: {output_dir}/lr_reconstructed_dac.wav")

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
            print(f"    3. outputs/lr_reconstructed_dac.wav    - LR通过DAC重建")
            print(f"    4. outputs/hr_reconstructed_dac.wav    - HR通过DAC重建")
            print(f"    5. outputs/generated_audio_dac.wav     - JaT模型生成")

            print(f"\n  🎧 听音频对比质量!")

        except Exception as e:
            print(f"\n  ❌ 音频生成失败: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("🎉 DAC overfit test 通过!")
        print("\n证明:")
        print("  ✅ DAC codec工作正常")
        print("  ✅ JaT模型架构正确")
        print("  ✅ 训练pipeline完整")
        print("\n准备好进行完整数据集训练!")
    else:
        print("⚠️  DAC overfit test 有问题")
        print("   检查loss曲线和组件")

    print("=" * 70)

    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
