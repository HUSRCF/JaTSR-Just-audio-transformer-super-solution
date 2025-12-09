#!/usr/bin/env python3
"""
调试训练问题
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.jat_audiosr import JaT_AudioSR

# DAC
import dac
from audiotools import AudioSignal
import soundfile as sf


def load_dac_model(device='cuda'):
    """加载DAC模型"""
    print("加载 DAC 模型...")
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model = model.to(device)
    model.eval()
    return model


def audio_to_dac_latent(audio, dac_model, device='cuda'):
    """将音频编码为DAC latent"""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    signal = AudioSignal(audio_tensor, sample_rate=48000)
    if dac_model.sample_rate != 48000:
        signal = signal.resample(dac_model.sample_rate)
    signal = signal.to(device)

    with torch.no_grad():
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(signal.audio_data)

    return z.squeeze(0)


def prepare_latents(hr_audio, lr_audio, dac_model, device):
    """准备latents"""
    print("\n转换为DAC latents...")
    hr_latent = audio_to_dac_latent(hr_audio, dac_model, device)
    lr_latent = audio_to_dac_latent(lr_audio, dac_model, device)

    print(f"  HR Latent: {list(hr_latent.shape)}")
    print(f"  LR Latent: {list(lr_latent.shape)}")

    # 归一化前保存统计量
    hr_mean_orig, hr_std_orig = hr_latent.mean(), hr_latent.std()
    lr_mean_orig, lr_std_orig = lr_latent.mean(), lr_latent.std()

    print(f"\n  归一化前: HR mean={hr_mean_orig:.4f}, std={hr_std_orig:.4f}")
    print(f"            LR mean={lr_mean_orig:.4f}, std={lr_std_orig:.4f}")

    # 归一化
    hr_latent = (hr_latent - hr_mean_orig) / hr_std_orig
    lr_latent = (lr_latent - lr_mean_orig) / lr_std_orig

    print(f"  归一化后: HR mean={hr_latent.mean():.4f}, std={hr_latent.std():.4f}")
    print(f"            LR mean={lr_latent.mean():.4f}, std={lr_latent.std():.4f}")

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
    return loss, pred_x0, hr_latent, z_t, t


def check_gradients(model):
    """检查梯度"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
    return grad_norms


def main():
    print("=" * 70)
    print("训练调试诊断")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 加载音频
    print("\n[1] 加载音频...")
    import librosa
    audio_path = '/home/husrcf/Code/AIAA/Final2/01 - 风继续吹.wav'
    hr_audio, sr = librosa.load(audio_path, sr=48000, mono=True, offset=40.0, duration=6.0)
    lr_audio = librosa.resample(hr_audio, orig_sr=48000, target_sr=16000)
    lr_audio = librosa.resample(lr_audio, orig_sr=16000, target_sr=48000)

    print(f"  HR audio: {hr_audio.shape}, std={hr_audio.std():.6f}")
    print(f"  LR audio: {lr_audio.shape}, std={lr_audio.std():.6f}")

    # 2. 加载DAC
    dac_model = load_dac_model(device)

    # 3. 准备latents
    hr_latent, lr_latent, stats = prepare_latents(hr_audio, lr_audio, dac_model, device)

    # 4. 创建模型
    print("\n[2] 创建模型...")
    model = JaT_AudioSR(
        input_channels=1024,
        cond_channels=1024,
        patch_len=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=256
    )
    model = model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params / 1e6:.2f}M")

    # 5. 测试前向传播
    print("\n[3] 测试前向传播...")
    hr = hr_latent.unsqueeze(0).to(device)
    lr = lr_latent.unsqueeze(0).to(device)

    with torch.no_grad():
        loss_initial, pred_initial, target, z_t, t = compute_loss(model, hr, lr)

    print(f"  初始loss: {loss_initial.item():.6f}")
    print(f"  预测输出范围: [{pred_initial.min():.4f}, {pred_initial.max():.4f}]")
    print(f"  预测输出均值: {pred_initial.mean():.4f}, 标准差: {pred_initial.std():.4f}")
    print(f"  目标范围: [{target.min():.4f}, {target.max():.4f}]")
    print(f"  目标均值: {target.mean():.4f}, 标准差: {target.std():.4f}")

    # 检查初始预测是否接近零（因为zero init）
    print(f"\n  初始预测是否接近零: {torch.abs(pred_initial).mean():.6f}")

    # 6. 测试不同学习率
    print("\n[4] 测试不同学习率...")
    learning_rates = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4]

    for lr_value in learning_rates:
        # 重新创建模型
        model_test = JaT_AudioSR(
            input_channels=1024,
            cond_channels=1024,
            patch_len=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            bottleneck_dim=256
        )
        model_test = model_test.to(device)
        model_test.train()

        optimizer = torch.optim.Adam(model_test.parameters(), lr=lr_value)

        losses = []
        for step in range(20):
            loss, pred, target, z_t, t = compute_loss(model_test, hr, lr)

            optimizer.zero_grad()
            loss.backward()

            # Check gradients
            if step == 0:
                grad_norms = check_gradients(model_test)
                max_grad = max(grad_norms.values())

            optimizer.step()

            losses.append(loss.item())

        print(f"  LR={lr_value:.0e}: loss {losses[0]:.4f} → {losses[-1]:.4f} "
              f"(下降 {(losses[0]-losses[-1])/losses[0]*100:.1f}%), "
              f"max_grad={max_grad:.2e}")

    # 7. 测试加入gradient clipping
    print("\n[5] 测试gradient clipping...")

    model_clip = JaT_AudioSR(
        input_channels=1024,
        cond_channels=1024,
        patch_len=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        bottleneck_dim=256
    )
    model_clip = model_clip.to(device)
    model_clip.train()

    optimizer = torch.optim.Adam(model_clip.parameters(), lr=1e-3)

    losses_clip = []
    for step in range(50):
        loss, pred, target, z_t, t = compute_loss(model_clip, hr, lr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
        optimizer.step()

        losses_clip.append(loss.item())

        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")

    print(f"\n  最终loss: {losses_clip[-1]:.6f}")
    print(f"  下降: {(losses_clip[0] - losses_clip[-1]) / losses_clip[0] * 100:.1f}%")

    # 8. 诊断结论
    print("\n" + "=" * 70)
    print("诊断结论:")
    print("=" * 70)

    if losses_clip[-1] < 0.1:
        print("✅ 模型可以学习!")
        print("\n建议:")
        print("  1. 使用学习率 1e-3 或 5e-4")
        print("  2. 添加gradient clipping (max_norm=1.0)")
        print("  3. 可能需要更多epochs (1000+)")
    elif losses_clip[-1] < 0.5:
        print("⚠️  模型学习缓慢")
        print("\n建议:")
        print("  1. 降低学习率到 5e-4 或 1e-4")
        print("  2. 添加gradient clipping")
        print("  3. 检查数据归一化")
    else:
        print("❌ 模型无法学习!")
        print("\n可能的问题:")
        print("  1. 模型架构有bug")
        print("  2. 数据有问题")
        print("  3. 初始化有问题")

    print("=" * 70)


if __name__ == '__main__':
    main()
