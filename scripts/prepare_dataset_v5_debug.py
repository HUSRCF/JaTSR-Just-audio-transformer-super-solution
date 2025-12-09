"""
JaT Benchmark V6 Speed Limit (极限速度测试版)
================================================
基于 Benchmark V6 逻辑，叠加以下 Buff:
1. [AMP] 开启 FP16 混合精度 (W7900 强项)。
2. [Compile] 开启 torch.compile(mode="reduce-overhead") (理论极限)。
3. [TF32] 开启 TensorFloat-32。

目标: 测试 W7900 在处理单曲时的理论天花板速度。
注意: 第一次运行时 Compile 会花费约 30-60s 进行编译，这是正常的。
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

# ==================== ⚙️ 配置 ====================
SOURCE_DIRS = ['1_source_audio', 'extra_audio']
DEBUG_DIR = "debug_output"

# 🚀 极限配置
HIGH_SR = 48000
LOW_SR = 16000
CHUNK_DURATION = 15.0  # 15s 切片 (比 30s 更稳，且能跑大 Batch)
DAC_BATCH_SIZE = 8     # 开启 AMP 后显存减半，Batch=8 应该很轻松
DAC_MODEL_TYPE = "44khz"

# ==================== ⏱️ 计时器 ====================
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

# ==================== 🏗️ Worker ====================
def benchmark_worker(gpu_id, file_path, result_queue):
    log_prefix = f"[GPU {gpu_id}]"
    stats = {}
    
    try:
        print(f"{log_prefix} 🚀 启动任务: {Path(file_path).name}")
        
        # 1. 初始化
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        # 开启 TF32
        torch.set_float32_matmul_precision('high')
        torch.cuda.reset_peak_memory_stats()
        
        with Timer("Load Model") as t_model:
            model_path = str(dac.utils.download(model_type=DAC_MODEL_TYPE))
            dac_model = dac.DAC.load(model_path).to(device)
            dac_model.eval()
            
            # 🔥🔥🔥 开启 Compile 🔥🔥🔥
            print(f"{log_prefix} 正在编译模型 (这可能需要几十秒)...")
            t_compile_start = time.time()
            try:
                dac_model = torch.compile(dac_model, mode="reduce-overhead")
            except Exception as e:
                print(f"{log_prefix} ⚠️ 编译失败: {e}")
            print(f"{log_prefix} 编译指令下达完成 (实际编译发生在第一次推理)")
            
        stats['model_load_time'] = t_model.duration

        # 2. 加载音频 (GPU 重采样)
        with Timer("Load & Resample") as t_load:
            # IO
            audio, sr = torchaudio.load(file_path)
            # 立即上 GPU
            audio = audio.to(device) 
            
            if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
            peak = audio.abs().max()
            if peak > 1.0: audio = audio / peak
            
            # GPU 重采样
            if sr != HIGH_SR:
                audio_hr = AF.resample(audio, orig_freq=sr, new_freq=HIGH_SR)
            else:
                audio_hr = audio
                
        stats['audio_load_time'] = t_load.duration
        total_samples = audio_hr.shape[1]
        duration_sec = total_samples / HIGH_SR
        print(f"{log_prefix} 预处理完成: {duration_sec:.2f}s音频")

        # 3. 切分
        chunk_samples = int(CHUNK_DURATION * HIGH_SR)
        chunks_hr = []
        num_chunks = math.ceil(total_samples / chunk_samples)
        pad_len = (num_chunks * chunk_samples) - total_samples
        
        audio_hr_padded = torch.nn.functional.pad(audio_hr, (0, pad_len))
        for i in range(num_chunks):
            start = i * chunk_samples
            chunks_hr.append(audio_hr_padded[:, start : start + chunk_samples])
            
        print(f"{log_prefix} 切分完成: {num_chunks} chunks")

        # 4. 编码 (AMP + Compile)
        latents_hr_list = []
        hop_length = None
        t_encode_start = time.time()
        
        # 强制清理显存
        torch.cuda.empty_cache()
        
        # 预热计数器 (用于剔除编译时间)
        is_first_batch = True
        
        for i in range(0, len(chunks_hr), DAC_BATCH_SIZE):
            batch_list = chunks_hr[i : i + DAC_BATCH_SIZE]
            batch_tensor = torch.stack(batch_list)
            
            # LR 模拟
            batch_lr = AF.resample(AF.resample(batch_tensor, HIGH_SR, LOW_SR), LOW_SR, HIGH_SR)
            if batch_lr.shape[-1] != batch_tensor.shape[-1]:
                 batch_lr = torch.nn.functional.pad(batch_lr, (0, batch_tensor.shape[-1] - batch_lr.shape[-1]))

            # 🔥🔥🔥 AMP 上下文 🔥🔥🔥
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                
                def encode_helper(wav):
                    signal = AudioSignal(wav, sample_rate=HIGH_SR)
                    if dac_model.sample_rate != HIGH_SR:
                        signal = signal.resample(dac_model.sample_rate)
                    with torch.no_grad():
                        z, _, _, _, _ = dac_model.encode(signal.audio_data)
                    return z

                t_batch_start = time.time()
                
                z_hr = encode_helper(batch_tensor)
                z_lr = encode_helper(batch_lr)
                
                # 同步以测量真实时间
                torch.cuda.synchronize()
                t_batch_end = time.time()
                
                if is_first_batch:
                    print(f"{log_prefix} 🔥 第一个 Batch (含编译) 耗时: {t_batch_end - t_batch_start:.3f}s")
                    is_first_batch = False
                else:
                    # 打印后续 Batch 的速度感受
                    print(f"{log_prefix} ⚡️ Batch 耗时: {t_batch_end - t_batch_start:.3f}s")
                    pass

            if hop_length is None:
                hop_length = batch_tensor.shape[-1] / z_hr.shape[-1]

            for b in range(z_hr.shape[0]):
                latents_hr_list.append(z_hr[b].cpu())

        stats['encoding_time'] = time.time() - t_encode_start
        print(f"{log_prefix} 编码总耗时: {stats['encoding_time']:.3f}s")

        # 5. 拼接
        full_hr = torch.cat(latents_hr_list, dim=-1)
        valid_frames = int(total_samples / hop_length)
        full_hr = full_hr[..., :valid_frames]
        
        stats['status'] = 'success'
        stats['peak_memory'] = torch.cuda.max_memory_allocated(device) / 1024**3
        stats['realtime_factor'] = duration_sec / stats['encoding_time']
        stats['duration_sec'] = duration_sec
        
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
    print("🚀 JaT Benchmark V6: Speed Limit Edition")
    print(f"⚙️  Config: AMP=ON | Compile=ON | Batch={DAC_BATCH_SIZE}")
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
    print("📊 极限性能报告")
    print("=" * 60)
    while not queue.empty():
        gpu_id, s = queue.get()
        if s['status'] == 'success':
            print(f"GPU {gpu_id}: ✅")
            print(f"  - 音频长度:   {s['duration_sec']:.2f}s")
            print(f"  - 编码耗时:   {s['encoding_time']:.3f}s (含编译时间)")
            print(f"  - 实时倍率:   {s['realtime_factor']:.2f}x Realtime")
            print(f"  - 显存峰值:   {s['peak_memory']:.2f} GB")
            print(f"  *如果不含首Batch编译，速度会更快*")

if __name__ == '__main__':
    main()