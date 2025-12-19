"""
可调参数的 De-essing 脚本
"""
import torch
import torchaudio
import sys

def de_ess_custom(audio, sr, threshold_db=-15, reduction_db=4,
                  freq_low=6000, freq_high=9000):
    """更温和的de-essing设置"""
    print(f"\n🎙️ Custom De-essing:")
    print(f"   Frequency: {freq_low}-{freq_high} Hz (narrower)")
    print(f"   Threshold: {threshold_db} dBFS (higher)")
    print(f"   Reduction: {reduction_db} dB (gentler)")
    
    n_fft = 2048
    hop_length = 512
    
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                     return_complex=True, 
                     window=torch.hann_window(n_fft, device=audio.device))
    
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    freqs = torch.fft.rfftfreq(n_fft, d=1/sr)
    sibilance_bins = (freqs >= freq_low) & (freqs <= freq_high)
    
    sibilance_energy = magnitude[:, sibilance_bins, :].mean(dim=1, keepdim=True)
    threshold_linear = 10 ** (threshold_db / 20)
    reduction_factor = 10 ** (-reduction_db / 20)
    
    gain = torch.ones_like(sibilance_energy)
    over_threshold = sibilance_energy > threshold_linear
    gain[over_threshold] = reduction_factor
    
    freq_gain = torch.ones_like(magnitude)
    freq_gain[:, sibilance_bins, :] = gain.expand(-1, sibilance_bins.sum(), -1)
    
    magnitude_deessed = magnitude * freq_gain
    stft_deessed = magnitude_deessed * torch.exp(1j * phase)
    
    audio_deessed = torch.istft(stft_deessed, n_fft=n_fft, hop_length=hop_length,
                                window=torch.hann_window(n_fft, device=audio.device),
                                length=audio.shape[-1])
    
    frames_reduced = over_threshold.sum().item()
    total_frames = over_threshold.numel()
    print(f"   Frames de-essed: {frames_reduced}/{total_frames} ({100*frames_reduced/total_frames:.1f}%)")
    
    return audio_deessed

if __name__ == '__main__':
    input_path = sys.argv[1]
    
    # 加载
    audio, sr = torchaudio.load(input_path)
    print(f"📊 Processing: {input_path}")
    
    # 更温和的de-essing
    audio_processed = de_ess_custom(audio, sr,
                                    threshold_db=-15,  # 提高阈值（更少触发）
                                    reduction_db=4,     # 减少衰减量（更温和）
                                    freq_low=6000,      # 缩窄频段（只处理最刺耳部分）
                                    freq_high=9000)
    
    # 保存
    output_path = input_path.replace('.wav', '_gentle_deess.wav')
    torchaudio.save(output_path, audio_processed, sr)
    print(f"✅ Saved to: {output_path}\n")
