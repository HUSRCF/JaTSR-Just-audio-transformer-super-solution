"""
计算音频质量指标：LSD (Log-Spectral Distance) 和 Mel Loss
"""

import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path

def load_audio(path, target_sr=44100):
    """加载音频并重采样到目标采样率"""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # 转为单声道（如果是立体声）
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0).numpy(), target_sr

def calculate_lsd(pred_audio, gt_audio, n_fft=2048, hop_length=512):
    """
    计算 Log-Spectral Distance (LSD)

    LSD = sqrt(mean((log(|pred_spec|) - log(|gt_spec|))^2))

    Args:
        pred_audio: 预测音频 [T]
        gt_audio: Ground truth音频 [T]
        n_fft: FFT窗口大小
        hop_length: 帧移

    Returns:
        lsd_db: LSD值 (dB)
    """
    # 确保长度一致
    min_len = min(len(pred_audio), len(gt_audio))
    pred_audio = pred_audio[:min_len]
    gt_audio = gt_audio[:min_len]

    # 计算STFT
    pred_spec = np.abs(librosa.stft(pred_audio, n_fft=n_fft, hop_length=hop_length))
    gt_spec = np.abs(librosa.stft(gt_audio, n_fft=n_fft, hop_length=hop_length))

    # 避免log(0)
    eps = 1e-8
    pred_spec = np.maximum(pred_spec, eps)
    gt_spec = np.maximum(gt_spec, eps)

    # 计算log差值
    log_diff = np.log10(pred_spec) - np.log10(gt_spec)

    # LSD: 对频率维度求平方和，然后开方，最后对时间维度求平均
    lsd_frames = np.sqrt(np.mean(log_diff ** 2, axis=0))  # 每帧的LSD
    lsd = np.mean(lsd_frames)

    # 转换为dB
    lsd_db = 20 * lsd  # LSD通常用20*log10表示

    return lsd_db, lsd_frames

def calculate_mel_loss(pred_audio, gt_audio, sr=44100, n_mels=80, n_fft=2048, hop_length=512):
    """
    计算 Mel-Spectrogram Loss (Multi-resolution)

    Args:
        pred_audio: 预测音频 [T]
        gt_audio: Ground truth音频 [T]
        sr: 采样率
        n_mels: Mel频带数
        n_fft: FFT窗口大小
        hop_length: 帧移

    Returns:
        mel_l1: Mel L1 Loss
        mel_l2: Mel L2 Loss (MSE)
    """
    # 确保长度一致
    min_len = min(len(pred_audio), len(gt_audio))
    pred_audio = pred_audio[:min_len]
    gt_audio = gt_audio[:min_len]

    # 计算Mel Spectrogram
    pred_mel = librosa.feature.melspectrogram(
        y=pred_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    gt_mel = librosa.feature.melspectrogram(
        y=gt_audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 转换为dB
    pred_mel_db = librosa.power_to_db(pred_mel, ref=np.max)
    gt_mel_db = librosa.power_to_db(gt_mel, ref=np.max)

    # 计算L1和L2损失
    mel_l1 = np.mean(np.abs(pred_mel_db - gt_mel_db))
    mel_l2 = np.sqrt(np.mean((pred_mel_db - gt_mel_db) ** 2))

    return mel_l1, mel_l2, pred_mel_db, gt_mel_db

def calculate_multi_scale_mel_loss(pred_audio, gt_audio, sr=44100):
    """
    计算多尺度 Mel Loss (参考工业界标准)
    使用多个FFT窗口大小
    """
    fft_sizes = [512, 1024, 2048]
    hop_sizes = [128, 256, 512]
    n_mels = [40, 64, 80]

    total_l1 = 0
    total_l2 = 0

    results = {}

    for fft_size, hop_size, n_mel in zip(fft_sizes, hop_sizes, n_mels):
        mel_l1, mel_l2, _, _ = calculate_mel_loss(
            pred_audio, gt_audio, sr, n_mel, fft_size, hop_size
        )
        total_l1 += mel_l1
        total_l2 += mel_l2

        results[f'fft{fft_size}'] = {'l1': mel_l1, 'l2': mel_l2}

    # 平均
    avg_l1 = total_l1 / len(fft_sizes)
    avg_l2 = total_l2 / len(fft_sizes)

    return avg_l1, avg_l2, results

def main():
    # 文件路径
    base_dir = Path("/media/990Evo/Code/JaT/inference_v2_last")

    pred_path = base_dir / "齐秦 - 丝路16_generated.wav"
    gt_path = base_dir / "齐秦 - 丝路16_hr_gt.wav"
    lr_path = base_dir / "齐秦 - 丝路16_lr_input.wav"

    print("=" * 80)
    print("音频质量评估 - 齐秦 - 丝路")
    print("=" * 80)

    # 加载音频
    print("\n📂 加载音频文件...")
    pred_audio, sr = load_audio(pred_path)
    gt_audio, _ = load_audio(gt_path)
    lr_audio, _ = load_audio(lr_path)

    print(f"   Generated: {len(pred_audio)/sr:.2f}s")
    print(f"   GT:        {len(gt_audio)/sr:.2f}s")
    print(f"   LR Input:  {len(lr_audio)/sr:.2f}s")

    # ========== 计算 LSD ==========
    print("\n" + "=" * 80)
    print("📊 Log-Spectral Distance (LSD)")
    print("=" * 80)

    lsd_gen_vs_gt, lsd_frames_gen = calculate_lsd(pred_audio, gt_audio)
    lsd_lr_vs_gt, lsd_frames_lr = calculate_lsd(lr_audio, gt_audio)

    print(f"\n✅ Generated vs GT:  {lsd_gen_vs_gt:.3f} dB")
    print(f"❌ LR Input vs GT:   {lsd_lr_vs_gt:.3f} dB")
    print(f"📈 Improvement:      {lsd_lr_vs_gt - lsd_gen_vs_gt:.3f} dB ({(1 - lsd_gen_vs_gt/lsd_lr_vs_gt)*100:.1f}%)")

    print(f"\n   LSD per frame stats:")
    print(f"   Generated: mean={np.mean(lsd_frames_gen):.3f}, std={np.std(lsd_frames_gen):.3f}")
    print(f"   LR Input:  mean={np.mean(lsd_frames_lr):.3f}, std={np.std(lsd_frames_lr):.3f}")

    # ========== 计算 Mel Loss (单尺度) ==========
    print("\n" + "=" * 80)
    print("🎵 Mel-Spectrogram Loss (Single-Scale)")
    print("=" * 80)

    mel_l1_gen, mel_l2_gen, _, _ = calculate_mel_loss(pred_audio, gt_audio)
    mel_l1_lr, mel_l2_lr, _, _ = calculate_mel_loss(lr_audio, gt_audio)

    print(f"\n✅ Generated vs GT:")
    print(f"   L1 Loss: {mel_l1_gen:.3f} dB")
    print(f"   L2 Loss: {mel_l2_gen:.3f} dB")

    print(f"\n❌ LR Input vs GT:")
    print(f"   L1 Loss: {mel_l1_lr:.3f} dB")
    print(f"   L2 Loss: {mel_l2_lr:.3f} dB")

    print(f"\n📈 Improvement:")
    print(f"   L1: {mel_l1_lr - mel_l1_gen:.3f} dB ({(1 - mel_l1_gen/mel_l1_lr)*100:.1f}%)")
    print(f"   L2: {mel_l2_lr - mel_l2_gen:.3f} dB ({(1 - mel_l2_gen/mel_l2_lr)*100:.1f}%)")

    # ========== 计算 Multi-Scale Mel Loss ==========
    print("\n" + "=" * 80)
    print("🎼 Multi-Scale Mel Loss")
    print("=" * 80)

    ms_l1_gen, ms_l2_gen, details_gen = calculate_multi_scale_mel_loss(pred_audio, gt_audio)
    ms_l1_lr, ms_l2_lr, details_lr = calculate_multi_scale_mel_loss(lr_audio, gt_audio)

    print("\n✅ Generated vs GT (Average):")
    print(f"   L1 Loss: {ms_l1_gen:.3f} dB")
    print(f"   L2 Loss: {ms_l2_gen:.3f} dB")

    print("\n   Details:")
    for key in details_gen:
        print(f"   {key}: L1={details_gen[key]['l1']:.3f}, L2={details_gen[key]['l2']:.3f}")

    print("\n❌ LR Input vs GT (Average):")
    print(f"   L1 Loss: {ms_l1_lr:.3f} dB")
    print(f"   L2 Loss: {ms_l2_lr:.3f} dB")

    print("\n📈 Improvement:")
    print(f"   L1: {ms_l1_lr - ms_l1_gen:.3f} dB ({(1 - ms_l1_gen/ms_l1_lr)*100:.1f}%)")
    print(f"   L2: {ms_l2_lr - ms_l2_gen:.3f} dB ({(1 - ms_l2_gen/ms_l2_lr)*100:.1f}%)")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("📝 Summary")
    print("=" * 80)
    print(f"\n指标                  Generated vs GT    LR vs GT      Improvement")
    print("-" * 80)
    print(f"LSD (dB)              {lsd_gen_vs_gt:8.3f}           {lsd_lr_vs_gt:8.3f}    {lsd_lr_vs_gt - lsd_gen_vs_gt:6.3f} ({(1 - lsd_gen_vs_gt/lsd_lr_vs_gt)*100:4.1f}%)")
    print(f"Mel L1 (dB)           {mel_l1_gen:8.3f}           {mel_l1_lr:8.3f}    {mel_l1_lr - mel_l1_gen:6.3f} ({(1 - mel_l1_gen/mel_l1_lr)*100:4.1f}%)")
    print(f"Mel L2 (dB)           {mel_l2_gen:8.3f}           {mel_l2_lr:8.3f}    {mel_l2_lr - mel_l2_gen:6.3f} ({(1 - mel_l2_gen/mel_l2_lr)*100:4.1f}%)")
    print(f"Multi-Scale L1 (dB)   {ms_l1_gen:8.3f}           {ms_l1_lr:8.3f}    {ms_l1_lr - ms_l1_gen:6.3f} ({(1 - ms_l1_gen/ms_l1_lr)*100:4.1f}%)")
    print(f"Multi-Scale L2 (dB)   {ms_l2_gen:8.3f}           {ms_l2_lr:8.3f}    {ms_l2_lr - ms_l2_gen:6.3f} ({(1 - ms_l2_gen/ms_l2_lr)*100:4.1f}%)")
    print("=" * 80)

    # ========== 质量等级评估 ==========
    print("\n🏆 Quality Assessment:")

    # LSD评估标准 (参考Audio SR论文)
    if lsd_gen_vs_gt < 1.0:
        lsd_grade = "Excellent (近乎完美)"
    elif lsd_gen_vs_gt < 1.5:
        lsd_grade = "Very Good (高质量)"
    elif lsd_gen_vs_gt < 2.0:
        lsd_grade = "Good (良好)"
    elif lsd_gen_vs_gt < 2.5:
        lsd_grade = "Fair (一般)"
    else:
        lsd_grade = "Poor (需改进)"

    print(f"   LSD Grade: {lsd_grade}")

    # Mel Loss评估
    if mel_l1_gen < 3.0:
        mel_grade = "Excellent"
    elif mel_l1_gen < 5.0:
        mel_grade = "Very Good"
    elif mel_l1_gen < 7.0:
        mel_grade = "Good"
    else:
        mel_grade = "Fair"

    print(f"   Mel Loss Grade: {mel_grade}")
    print("=" * 80)

if __name__ == "__main__":
    main()
