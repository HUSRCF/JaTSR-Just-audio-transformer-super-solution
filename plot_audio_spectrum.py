"""
绘制已解码音频的频谱图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# 配置
VALIDATION_SAMPLES_DIR = 'data_processed_v13_final/validation_samples'
OUTPUT_DIR = 'data_processed_v13_final/validation_samples'

def plot_audio_spectrum(audio_path, output_path):
    """绘制单个音频的频谱图"""

    # 加载音频
    print(f"加载: {Path(audio_path).name}")
    audio, sr = librosa.load(audio_path, sr=48000)

    # 创建图形（3个子图）
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Audio Analysis: {Path(audio_path).stem}', fontsize=14, fontweight='bold')

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
    img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title('Spectrogram (Linear Frequency)', fontsize=12)
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, 24000)  # 显示到 24kHz
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    # 3. Mel频谱图
    print("  绘制 Mel 频谱图...")
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=24000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img2 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='magma')
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
    print("🎵 音频频谱分析")
    print("=" * 70)
    print(f"音频目录: {VALIDATION_SAMPLES_DIR}")

    if not os.path.exists(VALIDATION_SAMPLES_DIR):
        print(f"❌ 目录不存在: {VALIDATION_SAMPLES_DIR}")
        return

    # 查找所有音频文件
    audio_files = list(Path(VALIDATION_SAMPLES_DIR).glob('*.wav'))

    if not audio_files:
        print("❌ 没有找到音频文件！")
        return

    print(f"\n找到 {len(audio_files)} 个音频文件\n")

    # 绘制每个音频的频谱
    results = []
    for i, audio_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] 处理: {audio_path.name}")

        output_path = os.path.join(OUTPUT_DIR, f"{audio_path.stem}_spectrum.png")

        try:
            duration, peak, rms = plot_audio_spectrum(str(audio_path), output_path)
            results.append({
                'file': audio_path.name,
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
    print("📊 汇总")
    print("=" * 70)
    for r in results:
        print(f"\n{r['file']}")
        print(f"  时长: {r['duration']:.2f}s")
        print(f"  峰值: {r['peak']:.4f}")
        print(f"  RMS:  {r['rms']:.4f}")
        print(f"  频谱图: {r['spectrum']}")

    print("\n✅ 完成！")

if __name__ == '__main__':
    main()
