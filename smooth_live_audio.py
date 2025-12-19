"""
现场录音平滑化处理脚本 - 针对粉丝录制CD等不稳定音频

处理流程：
1. AudioSR超分辨（修复高频、平滑伪影）
2. RMS归一化 + 动态范围压缩（平滑音量波动）
3. De-essing（抑制不稳定的齿音）
4. Gentle Noise Gate（清理底噪）

适用场景：
- 粉丝录制的现场演唱会CD
- 音量不稳定的录音
- 麦克风距离变化导致的音量波动
- 轻微的信号削波修复
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse


def compute_rms_envelope(audio, sr, window_ms=50):
    """
    计算RMS包络（用于检测音量变化）

    Args:
        audio: [channels, samples] tensor
        sr: 采样率
        window_ms: RMS窗口大小（毫秒）
    Returns:
        [channels, samples] RMS包络
    """
    window_samples = int(sr * window_ms / 1000)
    if window_samples % 2 == 0:
        window_samples += 1

    audio_squared = audio ** 2
    padding = window_samples // 2
    audio_squared_padded = torch.nn.functional.pad(
        audio_squared.unsqueeze(0),
        (padding, padding),
        mode='reflect'
    ).squeeze(0)

    kernel = torch.ones(1, 1, window_samples, device=audio.device) / window_samples
    rms_squared = torch.nn.functional.conv1d(
        audio_squared_padded.unsqueeze(0),
        kernel,
        padding=0
    ).squeeze(0)

    rms = torch.sqrt(rms_squared + 1e-10)
    return rms


def smooth_gain_curve(gain, sr, attack_ms=20, release_ms=100):
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

    gain_np = gain.cpu().numpy()
    smoothed = np.zeros_like(gain_np)

    for ch in range(gain_np.shape[0]):
        current_gain = gain_np[ch, 0]

        for i in range(gain_np.shape[1]):
            target_gain = gain_np[ch, i]

            if target_gain > current_gain:
                coeff = 1.0 / max(attack_samples, 1)
            else:
                coeff = 1.0 / max(release_samples, 1)

            current_gain = current_gain + (target_gain - current_gain) * coeff
            smoothed[ch, i] = current_gain

    return torch.from_numpy(smoothed).to(gain.device)


def smooth_dynamics(audio, sr, window_ms=50, target_rms=0.1,
                    max_gain=3.0, min_gain=0.3):
    """
    平滑音量波动（RMS归一化 + 温和压缩）

    原理：
    - 计算RMS包络，检测音量变化
    - 动态调整增益，使音量趋于稳定
    - 限制增益范围，避免过度放大噪音或削波

    适用于：
    - 歌手距离麦克风忽远忽近
    - 录音设备自动增益控制(AGC)不佳

    Args:
        audio: [channels, samples] waveform
        sr: 采样率
        window_ms: RMS窗口大小（毫秒）
        target_rms: 目标RMS值（0.1 = -20dBFS左右）
        max_gain: 最大增益（避免放大噪音）
        min_gain: 最小增益（避免过度压缩）
    Returns:
        [channels, samples] 平滑后的音频
    """
    print(f"\n🎚️ Smoothing dynamics:")
    print(f"   Window: {window_ms}ms")
    print(f"   Target RMS: {target_rms:.3f} ({20*np.log10(target_rms):.1f} dBFS)")
    print(f"   Gain range: {min_gain:.2f}x to {max_gain:.2f}x")

    # 计算RMS包络
    rms = compute_rms_envelope(audio, sr, window_ms)

    # 计算需要的增益
    target_gain = target_rms / (rms + 1e-8)

    # 限制增益范围
    target_gain = torch.clamp(target_gain, min_gain, max_gain)

    # 平滑增益曲线（避免突变）
    smoothed_gain = smooth_gain_curve(target_gain, sr,
                                      attack_ms=20, release_ms=100)

    # 应用增益
    output = audio * smoothed_gain

    # 防止削波
    peak = output.abs().max()
    if peak > 0.99:
        output = output * (0.99 / peak)
        print(f"   ⚠️ Peak limiting applied: {peak:.3f} → 0.99")

    # 统计信息
    original_rms = torch.sqrt(torch.mean(audio ** 2))
    processed_rms = torch.sqrt(torch.mean(output ** 2))
    avg_gain = smoothed_gain.mean()

    print(f"   Original RMS: {original_rms:.4f} ({20*np.log10(original_rms.item()+1e-10):.1f} dBFS)")
    print(f"   Processed RMS: {processed_rms:.4f} ({20*np.log10(processed_rms.item()+1e-10):.1f} dBFS)")
    print(f"   Average gain: {avg_gain:.2f}x")

    return output


def de_ess(audio, sr, threshold_db=-20, reduction_db=6,
           freq_low=5000, freq_high=9000):
    """
    De-essing（抑制过强的齿音）

    原理：
    - 检测高频区域（5-9kHz）的能量峰值
    - 当峰值超过阈值时，选择性衰减
    - 保持其他频段不变

    适用于：
    - 粉丝录音的齿音过重
    - 麦克风过近导致的刺耳高频

    Args:
        audio: [channels, samples]
        sr: 采样率
        threshold_db: 触发阈值（dBFS）
        reduction_db: 衰减量（dB）
        freq_low/high: 齿音频段范围
    Returns:
        [channels, samples] 处理后的音频
    """
    print(f"\n🎙️ De-essing:")
    print(f"   Frequency range: {freq_low}-{freq_high} Hz")
    print(f"   Threshold: {threshold_db} dBFS")
    print(f"   Reduction: {reduction_db} dB")

    # FFT
    n_fft = 2048
    hop_length = 512

    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                     return_complex=True, window=torch.hann_window(n_fft, device=audio.device))

    magnitude = torch.abs(stft)
    phase = torch.angle(stft)

    # 计算频率bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/sr)

    # 找到齿音频段的bin范围
    sibilance_bins = (freqs >= freq_low) & (freqs <= freq_high)

    # 计算齿音区域的能量
    sibilance_energy = magnitude[:, sibilance_bins, :].mean(dim=1, keepdim=True)  # [channels, 1, frames]

    # 转为dB
    sibilance_db = 20 * torch.log10(sibilance_energy + 1e-10)

    # 计算衰减增益
    threshold_linear = 10 ** (threshold_db / 20)
    reduction_factor = 10 ** (-reduction_db / 20)

    # 对于超过阈值的帧，应用衰减
    gain = torch.ones_like(sibilance_energy)
    over_threshold = sibilance_energy > threshold_linear
    gain[over_threshold] = reduction_factor

    # 广播到所有频率bins（仅影响齿音区域）
    freq_gain = torch.ones_like(magnitude)
    freq_gain[:, sibilance_bins, :] = gain.expand(-1, sibilance_bins.sum(), -1)

    # 应用增益
    magnitude_deessed = magnitude * freq_gain

    # 重建
    stft_deessed = magnitude_deessed * torch.exp(1j * phase)
    audio_deessed = torch.istft(stft_deessed, n_fft=n_fft, hop_length=hop_length,
                                window=torch.hann_window(n_fft, device=audio.device),
                                length=audio.shape[-1])

    # 统计
    frames_reduced = over_threshold.sum().item()
    total_frames = over_threshold.numel()
    print(f"   Frames de-essed: {frames_reduced}/{total_frames} ({100*frames_reduced/total_frames:.1f}%)")

    return audio_deessed


def gentle_compressor(audio, sr, threshold_db=-20, ratio=3.0,
                     attack_ms=10, release_ms=100, makeup_gain_db=0):
    """
    温和的动态范围压缩器（类似硬件压缩器）

    原理：
    - 限制动态范围，使大小声更均衡
    - 软拐点（soft knee）设计，平滑过渡
    - 自动makeup gain补偿

    适用于：
    - 现场录音动态范围过大
    - 小声部分听不清，大声部分过响

    Args:
        audio: [channels, samples]
        sr: 采样率
        threshold_db: 压缩阈值
        ratio: 压缩比（3:1表示温和）
        attack_ms: 开启时间
        release_ms: 释放时间
        makeup_gain_db: 补偿增益
    Returns:
        [channels, samples]
    """
    print(f"\n🎛️ Compression:")
    print(f"   Threshold: {threshold_db} dBFS")
    print(f"   Ratio: {ratio}:1")
    print(f"   Attack/Release: {attack_ms}ms / {release_ms}ms")

    # 计算RMS包络
    rms = compute_rms_envelope(audio, sr, window_ms=10)

    # 转为dB
    rms_db = 20 * torch.log10(rms + 1e-10)

    # 计算增益衰减（dB域）
    gain_reduction_db = torch.zeros_like(rms_db)
    over_threshold = rms_db > threshold_db

    # 压缩公式: gain_reduction = (threshold - input) * (1 - 1/ratio)
    gain_reduction_db[over_threshold] = (
        (threshold_db - rms_db[over_threshold]) * (1 - 1/ratio)
    )

    # 转回线性域
    gain_linear = 10 ** (gain_reduction_db / 20)

    # 平滑增益曲线
    gain_smooth = smooth_gain_curve(gain_linear, sr, attack_ms, release_ms)

    # 应用压缩
    output = audio * gain_smooth

    # Makeup gain
    if makeup_gain_db != 0:
        makeup_linear = 10 ** (makeup_gain_db / 20)
        output = output * makeup_linear
        print(f"   Makeup gain: {makeup_gain_db} dB")

    # 统计
    avg_reduction = 20 * torch.log10(gain_smooth.mean() + 1e-10).item()
    print(f"   Average gain reduction: {avg_reduction:.2f} dB")

    return output


def process_live_audio(input_path, output_path,
                      apply_dynamics=True, apply_deess=True,
                      apply_compression=True):
    """
    完整的现场录音平滑化流程

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        apply_dynamics: 是否应用动态平滑
        apply_deess: 是否应用de-essing
        apply_compression: 是否应用压缩
    """
    print(f"\n{'='*60}")
    print(f"Processing Live Audio: {Path(input_path).name}")
    print(f"{'='*60}")

    # 加载音频
    audio, sr = torchaudio.load(input_path)
    print(f"📊 Loaded: {audio.shape[0]} channels, {sr} Hz, {audio.shape[1]/sr:.2f}s")

    original_peak = audio.abs().max()
    print(f"   Original peak: {original_peak:.3f} ({20*np.log10(original_peak.item()):.1f} dBFS)")

    # 处理流程
    if apply_dynamics:
        audio = smooth_dynamics(audio, sr,
                               window_ms=50,
                               target_rms=0.1,
                               max_gain=3.0,
                               min_gain=0.3)

    if apply_compression:
        audio = gentle_compressor(audio, sr,
                                 threshold_db=-20,
                                 ratio=3.0,
                                 attack_ms=10,
                                 release_ms=100,
                                 makeup_gain_db=3.0)

    if apply_deess:
        audio = de_ess(audio, sr,
                      threshold_db=-20,
                      reduction_db=6,
                      freq_low=5000,
                      freq_high=9000)

    # 最终归一化（防止削波）
    peak = audio.abs().max()
    if peak > 0.99:
        audio = audio * (0.99 / peak)
        print(f"\n📊 Final peak limiting: {peak:.3f} → 0.99")

    # 保存
    torchaudio.save(output_path, audio, sr)
    print(f"\n✅ Saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Smooth live audio recordings')
    parser.add_argument('input', type=str, help='Input audio file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: input_smoothed.wav)')
    parser.add_argument('--no-dynamics', action='store_true',
                       help='Skip dynamics smoothing')
    parser.add_argument('--no-deess', action='store_true',
                       help='Skip de-essing')
    parser.add_argument('--no-compression', action='store_true',
                       help='Skip compression')

    args = parser.parse_args()

    # 生成输出文件名
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_smoothed{input_path.suffix}"
    else:
        output_path = args.output

    # 处理
    process_live_audio(
        args.input,
        str(output_path),
        apply_dynamics=not args.no_dynamics,
        apply_deess=not args.no_deess,
        apply_compression=not args.no_compression
    )


if __name__ == '__main__':
    main()
