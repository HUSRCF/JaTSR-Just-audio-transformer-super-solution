"""
JaT 数据预处理脚本 V13 (回归基准版)
================================================
深度比对 Benchmark V6 后的修正:
1. [禁用 Compile] 彻底移除 torch.compile。报错日志明确显示 TorchScript/Inductor 是 OOM 的元凶。
2. [保留 AMP] 开启 FP16 混合精度 (Benchmark V6 其实未开 AMP 速度已达标，开启后更稳)。
3. [显存保护] 相比 Benchmark V6 的 "全量加载至GPU"，本脚本保留 "CPU切片->GPU编码" 逻辑，
   这是为了防止遇到 >10分钟的长音频时直接撑爆显存。

核心配置:
- Batch Size: 8 (配合 AMP 很安全)
- 策略: 流式处理 + 显式 GC
"""

import os

# 环境变量配置：解决显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True"

import torch
import torchaudio
import torchaudio.functional as AF
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import json
import torch.multiprocessing as mp
import time
import math
import traceback

# DAC & AudioTools
import dac
from audiotools import AudioSignal

# ==================== ⚙️ 配置参数 ====================
SOURCE_DIRS = [
    '1_source_audio',
    'extra_audio'
]

OUTPUT_DIR = 'data_processed_v13_final' 
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# 状态维护文件
STATS_FILE = os.path.join(OUTPUT_DIR, 'running_stats.pt')
LOG_FILE = os.path.join(OUTPUT_DIR, 'processed_files.jsonl')
FINAL_STATS_JSON = os.path.join(OUTPUT_DIR, 'global_stats.json')

# 音频参数
HIGH_SR = 48000
LOW_SR = 16000

# Overlap 配置 (7s + 0.5s)
CHUNK_DURATION = 7.0    
OVERLAP_DURATION = 0.5  
MIN_DURATION = 1.0     

# 性能参数
# 既然禁用了 compile，Batch=8 是非常安全的
DAC_BATCH_SIZE = 1     

# DAC 模型
DAC_MODEL_TYPE = "44khz"

# 数据划分
VAL_RATIO = 0.1
RANDOM_SEED = 42

# ==================== 🛠️ 工具函数 ====================

def save_jsonl(data, filepath):
    with open(filepath, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_processed_files(filepath):
    processed = set()
    if not os.path.exists(filepath):
        return processed
    print(f"📂 读取处理日志: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                if data.get('status') == 'done':
                    processed.add(data['path'])
            except Exception:
                pass
    return processed

# ==================== 🏗️ GPU Worker (无 Compile 纯净版) ====================

def gpu_worker(gpu_id, input_queue, result_queue, output_dirs):
    try:
        # 1. 初始化
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')
        
        print(f"🚀 [GPU {gpu_id}] Worker 就绪 (Mode: Streaming | No Compile)")

        # 2. 加载模型
        model_path = str(dac.utils.download(model_type=DAC_MODEL_TYPE))
        dac_model = dac.DAC.load(model_path)
        dac_model = dac_model.to(device)
        dac_model.eval()
        
        # ❌ [显式禁用] 绝不使用 torch.compile
        # dac_model = torch.compile(dac_model) 
        
        # 3. 循环处理
        while True:
            task = input_queue.get()
            if task is None: break
            
            filepath, split_type = task
            filename = Path(filepath).stem
            save_dir = output_dirs[split_type]
            save_path = os.path.join(save_dir, f"{filename}.pt")
            
            try:
                # --- A. CPU 加载与预处理 ---
                # 保持音频在 CPU，防止长音频撑爆 GPU
                audio, sr = torchaudio.load(filepath) 
                
                if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
                peak = audio.abs().max()
                if peak > 1.0: audio = audio / peak
                
                total_samples_original = audio.shape[1]
                duration_sec = total_samples_original / sr
                
                if duration_sec < MIN_DURATION:
                    result_queue.put(('skipped', filepath, "too_short"))
                    continue

                # --- B. CPU 切分逻辑 (Overlap) ---
                chunks_raw_cpu = []
                chunk_valid_sec = CHUNK_DURATION
                overlap_sec = OVERLAP_DURATION
                num_chunks = math.ceil(duration_sec / chunk_valid_sec)
                
                for i in range(num_chunks):
                    t_start = i * chunk_valid_sec - overlap_sec
                    t_end = t_start + chunk_valid_sec + (2 * overlap_sec)
                    
                    idx_start = int(t_start * sr)
                    idx_end = int(t_end * sr)
                    
                    pad_left = 0
                    if idx_start < 0:
                        pad_left = -idx_start
                        idx_start = 0
                    
                    pad_right = 0
                    if idx_end > total_samples_original:
                        pad_right = idx_end - total_samples_original
                        idx_end = total_samples_original
                    
                    chunk = audio[:, idx_start:idx_end]
                    
                    if pad_left > 0 or pad_right > 0:
                        chunk = torch.nn.functional.pad(chunk, (pad_left, pad_right))
                    
                    chunks_raw_cpu.append(chunk)

                # --- C. GPU 流式处理 (Batch Loop) ---
                latents_hr_trimmed = []
                latents_lr_trimmed = []
                hop_length_48k = None 
                
                # 显存清理：每首歌开始前清理一次足够了
                torch.cuda.empty_cache()

                for i in range(0, len(chunks_raw_cpu), DAC_BATCH_SIZE):
                    # 1. 准备 Batch (CPU -> GPU)
                    batch_cpu_list = chunks_raw_cpu[i : i + DAC_BATCH_SIZE]
                    
                    # 长度对齐检查
                    target_len = batch_cpu_list[0].shape[-1]
                    clean_batch_list = []
                    for c in batch_cpu_list:
                        if c.shape[-1] != target_len:
                            diff = target_len - c.shape[-1]
                            c = torch.nn.functional.pad(c, (0, diff))
                        clean_batch_list.append(c)
                        
                    batch_raw = torch.stack(clean_batch_list)
                    # ⚡️ 移动到 GPU
                    batch_raw = batch_raw.to(device, non_blocking=True)
                    
                    # 2. GPU 重采样 (Batch Level)
                    if sr != HIGH_SR:
                        batch_hr = AF.resample(batch_raw, orig_freq=sr, new_freq=HIGH_SR)
                    else:
                        batch_hr = batch_raw
                        
                    # 3. LR 模拟
                    batch_lr = AF.resample(AF.resample(batch_hr, HIGH_SR, LOW_SR), LOW_SR, HIGH_SR)
                    if batch_lr.shape[-1] != batch_hr.shape[-1]:
                        batch_lr = torch.nn.functional.pad(batch_lr, (0, batch_hr.shape[-1] - batch_lr.shape[-1]))

                    # 4. AMP 编码 (无 Compile)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        def encode_helper(wav):
                            signal = AudioSignal(wav, sample_rate=HIGH_SR)
                            if dac_model.sample_rate != HIGH_SR:
                                signal = signal.resample(dac_model.sample_rate)
                            # 普通推理
                            with torch.no_grad():
                                z, _, _, _, _ = dac_model.encode(signal.audio_data)
                            return z

                        z_hr = encode_helper(batch_hr)
                        z_lr = encode_helper(batch_lr)
                    
                    # 5. Trim 参数计算
                    if hop_length_48k is None:
                        hop_length_48k = batch_hr.shape[-1] / z_hr.shape[-1]
                        overlap_samples_48k = int(OVERLAP_DURATION * HIGH_SR)
                        valid_samples_48k = int(CHUNK_DURATION * HIGH_SR)
                        trim_frames = int(overlap_samples_48k / hop_length_48k)
                        valid_frames_latent = int(valid_samples_48k / hop_length_48k)

                    # 6. Trim & Collect
                    z_hr_valid = z_hr[..., trim_frames : trim_frames + valid_frames_latent]
                    z_lr_valid = z_lr[..., trim_frames : trim_frames + valid_frames_latent]
                    
                    for b in range(z_hr_valid.shape[0]):
                        latents_hr_trimmed.append(z_hr_valid[b].cpu())
                        latents_lr_trimmed.append(z_lr_valid[b].cpu())
                        
                    # ⚡️ 显式释放显存 (防止循环引用)
                    del batch_raw, batch_hr, batch_lr, z_hr, z_lr

                # --- D. 拼接 ---
                full_hr = torch.cat(latents_hr_trimmed, dim=-1)
                full_lr = torch.cat(latents_lr_trimmed, dim=-1)
                
                total_samples_48k = int(duration_sec * HIGH_SR)
                expected_frames = int(total_samples_48k / hop_length_48k)
                
                full_hr = full_hr[..., :expected_frames]
                full_lr = full_lr[..., :expected_frames]

                # --- E. 统计与保存 ---
                f_sum = full_hr.double().sum(dim=1) + full_lr.double().sum(dim=1)
                f_sq_sum = (full_hr.double()**2).sum(dim=1) + (full_lr.double()**2).sum(dim=1)
                f_count = full_hr.shape[1] + full_lr.shape[1]

                torch.save({
                    'hr_latent': full_hr.half(),
                    'lr_latent': full_lr.half(),
                    'metadata': {
                        'name': filename,
                        'path': filepath,
                        'duration': duration_sec,
                        'sr': sr
                    }
                }, save_path)

                result_queue.put(('success', filepath, (f_sum, f_sq_sum, f_count)))

            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ [GPU {gpu_id}] OOM detected. File: {filename}")
                    # 如果还是 OOM，可能是极个别坏数据，尝试清理后继续
                    torch.cuda.empty_cache()
                result_queue.put(('error', filepath, str(e)))

    except Exception as e:
        print(f"🔥 [GPU {gpu_id}] 崩溃: {e}")
        traceback.print_exc()

# ==================== 🧠 主控流程 ====================

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    print("=" * 60)
    print("🚀 JaT 数据预处理 V13 (回归基准版)")
    print(f"⚙️  Config: Compile=OFF | Streaming=ON | Batch={DAC_BATCH_SIZE}")
    print("=" * 60)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    print("🔍 扫描音频文件...")
    all_files = []
    for d in SOURCE_DIRS:
        if os.path.exists(d):
            all_files.extend([str(p) for p in Path(d).rglob('*') if p.suffix.lower() in ['.wav', '.flac', '.mp3']])
    
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - VAL_RATIO))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    tasks = [(f, 'train') for f in train_files] + [(f, 'val') for f in val_files]
    print(f"📂 总文件数: {len(tasks)}")

    processed_set = load_processed_files(LOG_FILE)
    tasks_to_do = [t for t in tasks if t[0] not in processed_set]
    print(f"⚡️ 待处理: {len(tasks_to_do)}")
    
    if not tasks_to_do:
        print("✅ 完成。")
    
    if os.path.exists(STATS_FILE):
        stats_data = torch.load(STATS_FILE)
        global_sum = stats_data['sum']
        global_sq_sum = stats_data['sq_sum']
        global_count = stats_data['count']
    else:
        global_sum = torch.zeros(1024, dtype=torch.float64)
        global_sq_sum = torch.zeros(1024, dtype=torch.float64)
        global_count = 0

    if tasks_to_do:
        num_gpus = torch.cuda.device_count()
        manager = mp.Manager()
        input_queue = manager.Queue()
        result_queue = manager.Queue()
        
        for t in tasks_to_do: input_queue.put(t)
        for _ in range(num_gpus): input_queue.put(None)

        workers = []
        output_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR}
        
        for i in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(i, input_queue, result_queue, output_dirs))
            p.start()
            workers.append(p)
        
        pbar = tqdm(total=len(tasks_to_do), desc="Processing", unit="song")
        completed_curr = 0
        
        try:
            while completed_curr < len(tasks_to_do):
                if result_queue.empty():
                    if all(not p.is_alive() for p in workers): break
                    time.sleep(0.1)
                    continue
                
                res = result_queue.get()
                status, filepath, data = res
                
                if status == 'success':
                    f_sum, f_sq_sum, f_count = data
                    global_sum += f_sum
                    global_sq_sum += f_sq_sum
                    global_count += f_count
                    save_jsonl({'path': filepath, 'status': 'done'}, LOG_FILE)
                    completed_curr += 1
                    pbar.update(1)
                    if completed_curr % 50 == 0:
                        torch.save({'sum': global_sum, 'sq_sum': global_sq_sum, 'count': global_count}, STATS_FILE)
                elif status == 'skipped':
                    save_jsonl({'path': filepath, 'status': 'skipped', 'reason': data}, LOG_FILE)
                    completed_curr += 1
                    pbar.update(1)
                elif status == 'error':
                    save_jsonl({'path': filepath, 'status': 'error', 'msg': data}, LOG_FILE)
                    completed_curr += 1
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n🛑 用户中断")
            for p in workers: p.terminate()
        finally:
            pbar.close()
            for p in workers: 
                if p.is_alive(): p.join()
            torch.save({'sum': global_sum, 'sq_sum': global_sq_sum, 'count': global_count}, STATS_FILE)

    print("\n📊 计算全局统计量...")
    if global_count > 0:
        mean = global_sum / float(global_count)
        var = (global_sq_sum / float(global_count)) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        
        final_stats = {
            'hr_mean': mean.float().tolist(),
            'hr_std': std.float().tolist(),
            'lr_mean': mean.float().tolist(),
            'lr_std': std.float().tolist(),
            'total_frames': int(global_count)
        }
        with open(FINAL_STATS_JSON, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"✅ 完成: {FINAL_STATS_JSON}")

if __name__ == '__main__':
    main()