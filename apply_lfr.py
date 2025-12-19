"""
Low-Frequency Replacement (LFR) - 低频替换后处理

原理：
- 将生成音频的低频部分替换为原始LR输入
- 确保低频一致性，避免模型幻觉
- 适用于AudioSR等超分辨任务

参考文献：
- AudioSR: Versatile Audio Super-resolution at Scale (2024)
  "Lower frequency replacement (LFR) where the lower frequency part
   of the generated mel-spectrogram is replaced with the original inputs"

使用场景：
- 生成音频的低频与输入不一致
- 低频出现幻觉或伪影
- 需要严格保持低频信息

注意事项：
- 需要成对的生成音频和LR输入音频
- 默认替换0-12kHz（可调）
- 包含过渡带（软边界）以避免频谱不连续
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def low_frequency_replacement(generated_audio, lr_audio, sr,
                              cutoff_hz=12000, transition_hz=2000,
                              plot_spectrum=False):
    """
    低频替换（LFR）

    Args:
        generated_audio: [channels, samples] 生成的音频
        lr_audio: [channels, samples] 原始LR输入
        sr: 采样率
        cutoff_hz: 截止频率（Hz），低于此频率完全替换
        transition_hz: 过渡带宽度（Hz），软边界避免突变
        plot_spectrum: 是否绘制频谱对比图
    Returns:
        [channels, samples] LFR处理后的音频
    """
    print(f"\n🔄 Low-Frequency Replacement:")
    print(f"   Cutoff frequency: {cutoff_hz} Hz")
    print(f"   Transition band: {cutoff_hz}-{cutoff_hz+transition_hz} Hz")

    # 确保长度一致
    min_len = min(generated_audio.shape[-1], lr_audio.shape[-1])
    generated_audio = generated_audio[..., :min_len]
    lr_audio = lr_audio[..., :min_len]

    # FFT参数
    n_fft = 4096  # 高分辨率FFT，更精确的频率控制
    hop_length = 1024

    # STFT变换
    gen_stft = torch.stft(generated_audio, n_fft=n_fft, hop_length=hop_length,
                         return_complex=True,
                         window=torch.hann_window(n_fft, device=generated_audio.device))

    lr_stft = torch.stft(lr_audio, n_fft=n_fft, hop_length=hop_length,
                        return_complex=True,
                        window=torch.hann_window(n_fft, device=lr_audio.device))

    # 计算频率bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/sr)

    # 创建混合mask（软边界）
    # 低频区 (< cutoff): mask=0 (完全使用LR)
    # 过渡区 (cutoff ~ cutoff+transition): mask=0→1 (线性过渡)
    # 高频区 (> cutoff+transition): mask=1 (完全使用生成)

    mask = torch.zeros_like(freqs)

    # 低频区：mask=0
    low_freq_bins = freqs <= cutoff_hz
    mask[low_freq_bins] = 0.0

    # 高频区：mask=1
    high_freq_bins = freqs >= (cutoff_hz + transition_hz)
    mask[high_freq_bins] = 1.0

    # 过渡区：线性过渡
    transition_bins = (freqs > cutoff_hz) & (freqs < (cutoff_hz + transition_hz))
    if transition_bins.sum() > 0:
        transition_freqs = freqs[transition_bins]
        # 线性插值 0→1
        mask[transition_bins] = (transition_freqs - cutoff_hz) / transition_hz

    # 重塑mask用于广播 [freq_bins] → [1, freq_bins, 1]
    mask = mask.view(1, -1, 1)

    # 混合：output = lr * (1-mask) + gen * mask
    # 低频用LR，高频用生成，中间平滑过渡
    mixed_stft = lr_stft * (1 - mask) + gen_stft * mask

    # ISTFT重建
    output_audio = torch.istft(mixed_stft, n_fft=n_fft, hop_length=hop_length,
                               window=torch.hann_window(n_fft, device=generated_audio.device),
                               length=min_len)

    # 统计信息
    low_freq_count = low_freq_bins.sum().item()
    transition_count = transition_bins.sum().item()
    high_freq_count = high_freq_bins.sum().item()
    total_bins = len(freqs)

    print(f"   Frequency bins:")
    print(f"     Low (replaced): {low_freq_count}/{total_bins} ({100*low_freq_count/total_bins:.1f}%)")
    print(f"     Transition: {transition_count}/{total_bins} ({100*transition_count/total_bins:.1f}%)")
    print(f"     High (kept): {high_freq_count}/{total_bins} ({100*high_freq_count/total_bins:.1f}%)")

    # 绘制频谱对比（可选）
    if plot_spectrum:
        plot_frequency_comparison(generated_audio, lr_audio, output_audio, sr,
                                 cutoff_hz, transition_hz)

    return output_audio


def plot_frequency_comparison(gen_audio, lr_audio, lfr_audio, sr,
                              cutoff_hz, transition_hz):
    """
    绘制频谱对比图

    Args:
        gen_audio: 原始生成音频
        lr_audio: LR输入
        lfr_audio: LFR处理后
        sr: 采样率
        cutoff_hz: 截止频率
        transition_hz: 过渡带宽度
    """
    print("\n📊 Generating spectrum comparison plot...")

    # 计算频谱（取中间1秒）
    duration = 1.0
    start_sample = gen_audio.shape[-1] // 2 - int(sr * duration / 2)
    end_sample = start_sample + int(sr * duration)

    gen_segment = gen_audio[:, start_sample:end_sample].mean(dim=0)
    lr_segment = lr_audio[:, start_sample:end_sample].mean(dim=0)
    lfr_segment = lfr_audio[:, start_sample:end_sample].mean(dim=0)

    # FFT
    n_fft = 8192
    gen_fft = torch.fft.rfft(gen_segment, n=n_fft)
    lr_fft = torch.fft.rfft(lr_segment, n=n_fft)
    lfr_fft = torch.fft.rfft(lfr_segment, n=n_fft)

    gen_mag = torch.abs(gen_fft).numpy()
    lr_mag = torch.abs(lr_fft).numpy()
    lfr_mag = torch.abs(lfr_fft).numpy()

    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    # 转为dB
    gen_db = 20 * np.log10(gen_mag + 1e-10)
    lr_db = 20 * np.log10(lr_mag + 1e-10)
    lfr_db = 20 * np.log10(lfr_mag + 1e-10)

    # 绘图
    plt.figure(figsize=(12, 6))

    plt.plot(freqs, gen_db, label='Generated (original)', alpha=0.7, linewidth=1)
    plt.plot(freqs, lr_db, label='LR Input', alpha=0.7, linewidth=1)
    plt.plot(freqs, lfr_db, label='After LFR', alpha=0.9, linewidth=2, linestyle='--')

    # 标记截止频率和过渡带
    plt.axvline(cutoff_hz, color='red', linestyle=':', label=f'Cutoff ({cutoff_hz} Hz)')
    plt.axvline(cutoff_hz + transition_hz, color='orange', linestyle=':',
               label=f'Transition end ({cutoff_hz+transition_hz} Hz)')
    plt.axvspan(cutoff_hz, cutoff_hz + transition_hz, alpha=0.1, color='orange')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Spectrum Comparison (LFR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, sr/2)
    plt.ylim(-100, gen_db.max() + 10)

    # 保存图片
    output_path = 'lfr_spectrum_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved spectrum plot: {output_path}")
    plt.close()


def analyze_frequency_content(audio, sr, name="Audio"):
    """
    分析音频的频率内容

    Args:
        audio: [channels, samples]
        sr: 采样率
        name: 名称标识
    """
    # 取1秒中间段
    duration = 1.0
    start_sample = audio.shape[-1] // 2 - int(sr * duration / 2)
    end_sample = start_sample + int(sr * duration)
    segment = audio[:, start_sample:end_sample].mean(dim=0)

    # FFT
    n_fft = 8192
    fft = torch.fft.rfft(segment, n=n_fft)
    magnitude = torch.abs(fft).numpy()
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)

    # 计算不同频段的能量
    bands = [
        ("0-4kHz", 0, 4000),
        ("4-8kHz", 4000, 8000),
        ("8-12kHz", 8000, 12000),
        ("12-16kHz", 12000, 16000),
        ("16-20kHz", 16000, 20000)
    ]

    print(f"\n📊 Frequency content analysis ({name}):")
    for band_name, low, high in bands:
        band_bins = (freqs >= low) & (freqs < high)
        band_energy = np.sum(magnitude[band_bins] ** 2)
        band_db = 10 * np.log10(band_energy + 1e-10)
        print(f"   {band_name}: {band_db:.1f} dB")


def process_audio_pair(generated_path, lr_path, output_path,
                      cutoff_hz=12000, transition_hz=2000,
                      plot_spectrum=False, analyze=True):
    """
    处理音频对（生成 + LR输入）

    Args:
        generated_path: 生成音频路径
        lr_path: LR输入音频路径
        output_path: 输出路径
        cutoff_hz: 截止频率
        transition_hz: 过渡带宽度
        plot_spectrum: 是否绘制频谱图
        analyze: 是否分析频率内容
    """
    print(f"\n{'='*60}")
    print(f"Low-Frequency Replacement (LFR)")
    print(f"{'='*60}")
    print(f"📂 Generated: {Path(generated_path).name}")
    print(f"📂 LR Input:  {Path(lr_path).name}")

    # 加载音频
    gen_audio, gen_sr = torchaudio.load(generated_path)
    lr_audio, lr_sr = torchaudio.load(lr_path)

    if gen_sr != lr_sr:
        raise ValueError(f"Sample rate mismatch: gen={gen_sr}, lr={lr_sr}")

    sr = gen_sr

    print(f"\n📊 Audio info:")
    print(f"   Generated: {gen_audio.shape[0]} channels, {gen_audio.shape[1]/sr:.2f}s")
    print(f"   LR Input:  {lr_audio.shape[0]} channels, {lr_audio.shape[1]/sr:.2f}s")
    print(f"   Sample rate: {sr} Hz")

    # 频率内容分析（可选）
    if analyze:
        analyze_frequency_content(gen_audio, sr, "Generated")
        analyze_frequency_content(lr_audio, sr, "LR Input")

    # 应用LFR
    output_audio = low_frequency_replacement(
        gen_audio, lr_audio, sr,
        cutoff_hz=cutoff_hz,
        transition_hz=transition_hz,
        plot_spectrum=plot_spectrum
    )

    # 频率内容分析（处理后）
    if analyze:
        analyze_frequency_content(output_audio, sr, "After LFR")

    # 保存
    torchaudio.save(output_path, output_audio, sr)
    print(f"\n✅ Saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Apply Low-Frequency Replacement (LFR) to audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (replace 0-12kHz)
  python apply_lfr.py generated.wav lr_input.wav

  # Custom cutoff frequency (16kHz)
  python apply_lfr.py generated.wav lr_input.wav --cutoff 16000

  # With spectrum plot
  python apply_lfr.py generated.wav lr_input.wav --plot

  # Narrow transition band (sharper cutoff)
  python apply_lfr.py generated.wav lr_input.wav --cutoff 12000 --transition 500
        """
    )

    parser.add_argument('generated', type=str,
                       help='Generated audio file path')
    parser.add_argument('lr_input', type=str,
                       help='LR input audio file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: generated_lfr.wav)')
    parser.add_argument('--cutoff', type=float, default=12000,
                       help='Cutoff frequency in Hz (default: 12000)')
    parser.add_argument('--transition', type=float, default=2000,
                       help='Transition band width in Hz (default: 2000)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate spectrum comparison plot')
    parser.add_argument('--no-analyze', action='store_true',
                       help='Skip frequency content analysis')

    args = parser.parse_args()

    # 生成输出文件名
    if args.output is None:
        gen_path = Path(args.generated)
        output_path = gen_path.parent / f"{gen_path.stem}_lfr{gen_path.suffix}"
    else:
        output_path = args.output

    # 处理
    process_audio_pair(
        args.generated,
        args.lr_input,
        str(output_path),
        cutoff_hz=args.cutoff,
        transition_hz=args.transition,
        plot_spectrum=args.plot,
        analyze=not args.no_analyze
    )


if __name__ == '__main__':
    main()
