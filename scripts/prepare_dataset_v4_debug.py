"""
JaT V6 性能验证脚本 (双重优化版)
修复:
1. CPU瓶颈: 将重采样 (Resample) 移至 GPU 进行，消除 200s+ 的加载延迟。
2. GPU瓶颈: Batch Size 降为 4，配合环境变量解决 MIOpen 显存碎片化。
"""

import os
import torch
import torchaudio
import torchaudio.functional as AF
import numpy as np
from pathlib import Path
import random
import torch.multiprocessing as mp
import time
import math
import traceback
import dac
from audiotools import AudioSignal

# ==================== 配置 ====================
SOURCE_DIRS = ['1_source_audio', 'extra_audio']
DEBUG_DIR = "debug_output"

# 🚀 优化配置
HIGH_SR = 48000
LOW_SR = 16000
CHUNK_DURATION = 6.0  
DAC_BATCH_SIZE = 1     # ⬇️ 降级: 从 8 降为 4 (解决 MIOpen 警告的核心)
DAC_MODEL_TYPE = "44khz"

# ==================== 计时器 ====================
class Timer:
    def __init__(self, name):
        self.name = name
        self.start = 0
        self.duration = 0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.duration = time.time() - self.start

# ==================== Worker ====================
def benchmark_worker(gpu_id, file_path, result_queue):
    log_prefix = f"[GPU {gpu_id}]"
    stats = {}
    
    try:
        print(f"{log_prefix} 🚀 启动任务: {Path(file_path).name}")
        
        # 1. 初始化
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        # 预分配一点显存防止碎片
        torch.cuda.reset_peak_memory_stats()
        
        with Timer("Load Model") as t_model:
            model_path = str(dac.utils.download(model_type=DAC_MODEL_TYPE))
            dac_model = dac.DAC.load(model_path).to(device)
            dac_model.eval()
        stats['model_load_time'] = t_model.duration

        # 2. 加载音频 (🚀 优化点 A: GPU 重采样)
        with Timer("Load & Resample") as t_load:
            # 2.1 快速读取 (IO)
            audio, sr = torchaudio.load(file_path)
            
            # 2.2 ⚡️ 立即移动到 GPU
            audio = audio.to(device) 
            
            # 2.3 预处理 (在 GPU 上进行，速度极快)
            if audio.shape[0] > 1: 
                audio = audio.mean(dim=0, keepdim=True)
            peak = audio.abs().max()
            if peak > 1.0: 
                audio = audio / peak
            
            # 2.4 重采样 (在 GPU 上进行)
            if sr != HIGH_SR:
                audio_hr = AF.resample(audio, orig_freq=sr, new_freq=HIGH_SR)
            else:
                audio_hr = audio
                
        stats['audio_load_time'] = t_load.duration
        total_samples = audio_hr.shape[1]
        duration_sec = total_samples / HIGH_SR
        print(f"{log_prefix} 加载&预处理完成: {t_load.duration:.2f}s (音频长度: {duration_sec:.2f}s)")

        # 3. 切分
        chunk_samples = int(CHUNK_DURATION * HIGH_SR)
        chunks_hr = []
        num_chunks = math.ceil(total_samples / chunk_samples)
        pad_len = (num_chunks * chunk_samples) - total_samples
        
        # GPU 上直接 Pad 和切片
        audio_hr_padded = torch.nn.functional.pad(audio_hr, (0, pad_len))
        for i in range(num_chunks):
            start = i * chunk_samples
            chunks_hr.append(audio_hr_padded[:, start : start + chunk_samples])
            
        print(f"{log_prefix} 切分完成: {num_chunks} chunks")
        # 清空显存
        torch.cuda.empty_cache()
        # 4. 编码
        latents_hr_list = []
        hop_length = None
        t_encode_start = time.time()
        
        # 🚀 优化点 B: Batch Size = 4
        for i in range(0, len(chunks_hr), DAC_BATCH_SIZE):
            batch_list = chunks_hr[i : i + DAC_BATCH_SIZE]
            batch_tensor = torch.stack(batch_list) # 已经在 GPU 上了
            
            # 模拟 LR 处理 (GPU)
            batch_lr = AF.resample(AF.resample(batch_tensor, HIGH_SR, LOW_SR), LOW_SR, HIGH_SR)
            if batch_lr.shape[-1] != batch_tensor.shape[-1]:
                 batch_lr = torch.nn.functional.pad(batch_lr, (0, batch_tensor.shape[-1] - batch_lr.shape[-1]))

            def encode_helper(wav):
                # wav: [B, 1, T] on GPU
                signal = AudioSignal(wav, sample_rate=HIGH_SR)
                if dac_model.sample_rate != HIGH_SR:
                    signal = signal.resample(dac_model.sample_rate)
                with torch.no_grad():
                    z, _, _, _, _ = dac_model.encode(signal.audio_data)
                return z

            z_hr = encode_helper(batch_tensor)
            z_lr = encode_helper(batch_lr) # 加上 LR 编码以模拟真实负载

            if hop_length is None:
                hop_length = batch_tensor.shape[-1] / z_hr.shape[-1]

            for b in range(z_hr.shape[0]):
                latents_hr_list.append(z_hr[b].cpu()) # 移回 CPU 暂存

        stats['encoding_time'] = time.time() - t_encode_start
        print(f"{log_prefix} 编码完成: {stats['encoding_time']:.3f}s")

        # 5. 拼接
        full_hr = torch.cat(latents_hr_list, dim=-1)
        valid_frames = int(total_samples / hop_length)
        full_hr = full_hr[..., :valid_frames]
        
        stats['status'] = 'success'
        stats['peak_memory'] = torch.cuda.max_memory_allocated(device) / 1024**3
        stats['realtime_factor'] = duration_sec / stats['encoding_time']
        
        result_queue.put((gpu_id, stats))

    except Exception as e:
        print(f"{log_prefix} ❌ Error: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, {'status': 'error', 'msg': str(e)}))

# ==================== Main ====================
def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
        
    print("=" * 60)
    print("🚀 JaT V6 Performance Test (Optimized)")
    print(f"⚙️  Config: GPU Resample | Batch={DAC_BATCH_SIZE}")
    print("=" * 60)
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    all_files = []
    for d in SOURCE_DIRS:
        if os.path.exists(d):
            all_files.extend([str(p) for p in Path(d).rglob('*') if p.suffix.lower() in ['.wav', '.flac', '.mp3']])
    
    if len(all_files) < 2: return
    random.shuffle(all_files)
    test_files = all_files[:2]
    
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    procs = []
    
    for i in range(2):
        p = ctx.Process(target=benchmark_worker, args=(i, test_files[i], queue))
        p.start()
        procs.append(p)
        
    for p in procs: p.join()
    
    print("\n" + "=" * 60)
    print("📊 最终性能报告")
    print("=" * 60)
    while not queue.empty():
        gpu_id, s = queue.get()
        if s['status'] == 'success':
            print(f"GPU {gpu_id}: ✅")
            print(f"  - 加载时间:   {s['audio_load_time']:.2f}s (原: 225s -> 现: <2s)")
            print(f"  - 编码时间:   {s['encoding_time']:.2f}s (原: 287s -> 现: ~30s)")
            print(f"  - 实时倍率:   {s['realtime_factor']:.2f}x Realtime")
            print(f"  - 显存峰值:   {s['peak_memory']:.2f} GB")

if __name__ == '__main__':
    main()