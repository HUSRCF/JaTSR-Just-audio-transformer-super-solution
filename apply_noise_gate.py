"""
专业级温和Noise Gate - 用于HiFi音频后处理
针对扩散模型生成的微弱噪声底进行抑制
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse


def compute_rms_envelope(audio, sr, window_ms=10):
    """
    计算RMS包络

    Args:
        audio: [channels, samples] tensor
        sr: 采样率
        window_ms: RMS窗口大小（毫秒）
    Returns:
        [channels, samples] RMS包络
    """
    window_samples = int(sr * window_ms / 1000)

    # 确保窗口是奇数（用于对称padding）
    if window_samples % 2 == 0:
        window_samples += 1

    # 计算平方
    audio_squared = audio ** 2

    # 使用平均池化计算RMS（加padding保持长度）
    padding = window_samples // 2
    audio_squared_padded = torch.nn.functional.pad(
        audio_squared.unsqueeze(0),
        (padding, padding),
        mode='reflect'
    ).squeeze(0)

    # 平均池化
    kernel = torch.ones(1, 1, window_samples, device=audio.device) / window_samples
    rms_squared = torch.nn.functional.conv1d(
        audio_squared_padded.unsqueeze(0),
        kernel,
        padding=0
    ).squeeze(0)

    # 开方得到RMS
    rms = torch.sqrt(rms_squared + 1e-10)

    return rms


def smooth_gain_curve(gain, sr, attack_ms=5, release_ms=50):
    """
    平滑增益曲线（实现attack/release）

    Args:
        gain: [channels, samples] 原始增益
        sr: 采样率
        attack_ms: 开启时间（毫秒）
        release_ms: 释放时间（毫秒）
    Returns:
        [channels, samples] 平滑后的增益
    """
    attack_samples = int(sr * attack_ms / 1000)
    release_samples = int(sr * release_ms / 1000)

    # 转为numpy便于逐样本处理
    gain_np = gain.cpu().numpy()
    smoothed = np.zeros_like(gain_np)

    for ch in range(gain_np.shape[0]):
        current_gain = gain_np[ch, 0]

        for i in range(gain_np.shape[1]):
            target_gain = gain_np[ch, i]

            # 根据上升/下降选择时间常数
            if target_gain > current_gain:
                # 信号增强 -> 使用attack
                coeff = 1.0 / max(attack_samples, 1)
            else:
                # 信号减弱 -> 使用release
                coeff = 1.0 / max(release_samples, 1)

            # 一阶低通滤波
            current_gain = current_gain + (target_gain - current_gain) * coeff
            smoothed[ch, i] = current_gain

    return torch.from_numpy(smoothed).to(gain.device)


def gentle_noise_gate(audio, sr, threshold_db=-70, ratio=3.0,
                     attack_ms=5, release_ms=50, window_ms=10):
    """
    温和的噪声门 - 软拐点设计，适合HiFi音乐

    Args:
        audio: [channels, samples] waveform tensor
        sr: 采样率
        threshold_db: 门限（低于此值开始衰减）
        ratio: 衰减比例（1:ratio），3.0表示温和压缩
        attack_ms: 开启时间
        release_ms: 释放时间
        window_ms: RMS窗口大小
    Returns:
        [channels, samples] 处理后的音频
    """
    print(f"\n🔧 Applying gentle noise gate:")
    print(f"   Threshold: {threshold_db} dB")
    print(f"   Ratio: 1:{ratio}")
    print(f"   Attack/Release: {attack_ms}ms / {release_ms}ms")

    # 计算门限线性值
    threshold_linear = 10 ** (threshold_db / 20)

    # 1. 计算RMS包络
    rms = compute_rms_envelope(audio, sr, window_ms)

    # 2. 计算增益（软拐点 - 避免硬切）
    # 使用压缩器风格的平滑过渡，而非硬gate
    gain = torch.ones_like(rms)

    # 低于门限的部分，计算衰减
    mask = rms < threshold_linear

    # 软膝盖（soft knee）公式：
    # gain = (rms / threshold) ^ ((1 - 1/ratio))
    # 当rms=threshold时，gain=1（无衰减）
    # 当rms=0时，gain→0
    # ratio=3时，衰减斜率温和
    gain[mask] = torch.pow(
        rms[mask] / (threshold_linear + 1e-10),
        (1.0 - 1.0 / ratio)
    )

    # 限制最小增益（避免完全静音）
    min_gain = 10 ** (-20 / 20)  # -20dB最大衰减
    gain = torch.clamp(gain, min=min_gain, max=1.0)

    # 3. 平滑增益曲线（attack/release）
    gain_smooth = smooth_gain_curve(gain, sr, attack_ms, release_ms)

    # 4. 应用增益
    output = audio * gain_smooth

    # 统计信息
    avg_attenuation = 20 * torch.log10(gain_smooth.mean() + 1e-10).item()
    print(f"   ✅ Average attenuation: {avg_attenuation:.2f} dB")
    print(f"   ✅ Max attenuation: {20 * torch.log10(gain_smooth.min() + 1e-10).item():.2f} dB")

    return output


def analyze_noise_floor(audio, sr):
    """
    分析音频的噪声底（找最安静的1秒）

    Args:
        audio: [channels, samples]
        sr: 采样率
    Returns:
        noise_floor_db: 噪声底（dB）
    """
    hop = sr  # 1秒窗口
    audio_mono = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]

    # 计算每秒的能量
    energies = []
    for i in range(0, len(audio_mono) - hop, hop):
        segment = audio_mono[i:i+hop]
        energy = torch.sum(segment ** 2).item()
        energies.append(energy)

    if not energies:
        return -80.0

    # 找最安静的段
    quietest_energy = min(energies)
    noise_rms = np.sqrt(quietest_energy / hop)
    noise_db = 20 * np.log10(noise_rms + 1e-10)

    return noise_db


def process_audio_file(input_path, output_path, threshold_db=-70,
                       ratio=3.0, attack_ms=5, release_ms=50):
    """
    处理单个音频文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        threshold_db: noise gate门限
        ratio: 衰减比例
        attack_ms: 开启时间
        release_ms: 释放时间
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*60}")

    # 1. 加载音频
    audio, sr = torchaudio.load(input_path)
    print(f"📊 Loaded audio: {audio.shape[0]} channels, {audio.shape[1]} samples ({audio.shape[1]/sr:.2f}s)")
    print(f"   Sample rate: {sr} Hz")

    # 2. 分析原始噪声底
    original_noise_floor = analyze_noise_floor(audio, sr)
    print(f"   Original noise floor: {original_noise_floor:.1f} dB")

    # 3. 应用noise gate
    audio_processed = gentle_noise_gate(
        audio, sr,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms
    )

    # 4. 分析处理后噪声底
    processed_noise_floor = analyze_noise_floor(audio_processed, sr)
    print(f"   Processed noise floor: {processed_noise_floor:.1f} dB")
    print(f"   Improvement: {processed_noise_floor - original_noise_floor:.1f} dB")

    # 5. 保存
    torchaudio.save(output_path, audio_processed, sr)
    print(f"\n✅ Saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Apply gentle noise gate to audio files')
    parser.add_argument('input', type=str, help='Input audio file path')
    parser.add_argument('--threshold', type=float, default=-70.0,
                       help='Noise gate threshold in dB (default: -70)')
    parser.add_argument('--ratio', type=float, default=3.0,
                       help='Attenuation ratio (default: 3.0 for gentle)')
    parser.add_argument('--attack', type=float, default=5.0,
                       help='Attack time in ms (default: 5)')
    parser.add_argument('--release', type=float, default=50.0,
                       help='Release time in ms (default: 50)')

    args = parser.parse_args()

    # 生成输出文件名
    input_path = Path(args.input)
    output_path = input_path.parent / f"{input_path.stem}_unnoised{input_path.suffix}"

    # 处理
    process_audio_file(
        str(input_path),
        str(output_path),
        threshold_db=args.threshold,
        ratio=args.ratio,
        attack_ms=args.attack,
        release_ms=args.release
    )


if __name__ == '__main__':
    main()
