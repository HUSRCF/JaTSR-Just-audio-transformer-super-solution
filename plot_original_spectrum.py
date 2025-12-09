"""
绘制原始音频的频谱图（用于与解码后的音频对比）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# 配置
SOURCE_AUDIO_DIR = '1_source_audio'
OUTPUT_DIR = 'data_processed_v13_final/validation_samples'
TARGET_SR = 48000  # 与解码后的音频保持一致
SEGMENT_DURATION = 18.0  # 18秒片段

# 需要处理的三个原始音频文件
AUDIO_FILES = [
    '15 - 无胆入情关.flac',
    '张国荣-想你 (Live).flac',
    '01 - 全赖有你.flac'
]

def plot_audio_spectrum(audio_path, output_path, extract_segment=True):
    """绘制单个音频的频谱图"""

    # 加载音频
    print(f"加载: {Path(audio_path).name}")
    audio, sr = librosa.load(audio_path, sr=TARGET_SR)

    # 提取片段（从1/4位置开始的18秒）
    if extract_segment:
        total_duration = len(audio) / sr
        start_time = total_duration / 4.0
        start_sample = int(start_time * sr)
        end_sample = int((start_time + SEGMENT_DURATION) * sr)
        audio = audio[start_sample:end_sample]
        print(f"  提取片段: {start_time:.2f}s - {start_time + SEGMENT_DURATION:.2f}s")

    # 创建图形（3个子图）
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Audio Analysis: {Path(audio_path).stem} [ORIGINAL]',
                 fontsize=14, fontweight='bold')

    # 1. 波形图
    print("  绘制波形...")
    librosa.display.waveshow(audio, sr=sr, ax=axes[0], alpha=0.8)
    axes[0].set_title('Waveform', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. 频谱图（Spectrogram）
    print("  绘制频谱图...")
    D = librosa.stft(audio)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz',
                                    ax=axes[1], cmap='viridis')
    axes[1].set_title('Spectrogram (Linear Frequency)', fontsize=12)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, 24000)  # 显示到 24kHz
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    # 3. Mel频谱图
    print("  绘制 Mel 频谱图...")
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=24000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                     ax=axes[2], cmap='magma')
    axes[2].set_title('Mel Spectrogram', fontsize=12)
    axes[2].set_ylabel('Mel Frequency')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

    # 添加统计信息
    duration = len(audio) / sr
    peak = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio**2))

    info_text = f'Duration: {duration:.2f}s | Peak: {peak:.4f} | RMS: {rms:.4f}'
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存到: {output_path}")
    plt.close()

    return duration, peak, rms

def main():
    print("=" * 70)
    print("🎵 原始音频频谱分析（对比用）")
    print("=" * 70)
    print(f"音频目录: {SOURCE_AUDIO_DIR}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    if not os.path.exists(SOURCE_AUDIO_DIR):
        print(f"❌ 目录不存在: {SOURCE_AUDIO_DIR}")
        return

    # 处理每个音频文件
    results = []
    for i, filename in enumerate(AUDIO_FILES):
        audio_path = os.path.join(SOURCE_AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"[{i+1}/{len(AUDIO_FILES)}] ❌ 文件不存在: {filename}\n")
            continue

        print(f"[{i+1}/{len(AUDIO_FILES)}] 处理: {filename}")

        # 输出文件名：原始文件名_original_18s_spectrum.png
        output_filename = f"{Path(filename).stem}_original_18s_spectrum.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        try:
            duration, peak, rms = plot_audio_spectrum(audio_path, output_path)
            results.append({
                'file': filename,
                'duration': duration,
                'peak': peak,
                'rms': rms,
                'spectrum': output_path
            })
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

        print()

    # 打印汇总
    print("=" * 70)
    print("📊 汇总（原始音频）")
    print("=" * 70)
    for r in results:
        print(f"\n{r['file']}")
        print(f"  时长: {r['duration']:.2f}s")
        print(f"  峰值: {r['peak']:.4f}")
        print(f"  RMS:  {r['rms']:.4f}")
        print(f"  频谱图: {r['spectrum']}")

    print("\n✅ 完成！")
    print("\n现在可以对比原始音频与解码后的音频的频谱图：")
    print("  - *_original_18s_spectrum.png (原始)")
    print("  - *_decoded_18s_spectrum.png (解码后)")

if __name__ == '__main__':
    main()
