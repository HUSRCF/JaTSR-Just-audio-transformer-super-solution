"""
JaT 数据预处理脚本 V8 (Overlap-Stitch 拼接版)
================================================
核心升级:
1. [Overlap机制] 使用 7s 有效切片 + 0.5s 上下文 Padding (总长 8s)。
   编码后切除首尾受污染的 Latent，仅保留中间纯净部分拼接，彻底消除边界效应。
2. [Batch优化] 切片变短 (8s)，Batch Size 提升至 8，进一步压榨 W7900 性能。
3. [继承特性] 保留了 V7 的 GPU 重采样、MIOpen 显存优化、断点续传和全局统计。
"""

import os
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
    '1_source_audio'
]

OUTPUT_DIR = 'data_processed_overlap' # 区分目录
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# 状态维护文件
STATS_FILE = os.path.join(OUTPUT_DIR, 'running_stats.pt')
LOG_FILE = os.path.join(OUTPUT_DIR, 'processed_files.jsonl')
FINAL_STATS_JSON = os.path.join(OUTPUT_DIR, 'global_stats.json')

# 音频参数
HIGH_SR = 48000
LOW_SR = 16000

# 🚀 Overlap 核心配置
CHUNK_DURATION = 7.0    # 有效长度 (保留部分)
OVERLAP_DURATION = 0.5  # 上下文长度 (将被切除部分)
# 实际送入 GPU 长度 = 0.5 + 7.0 + 0.5 = 8.0s

MIN_DURATION = 1.0     

# 🚀 性能参数
# 8s 切片比 15s 小很多，Batch=8 在 W7900 上是安全的
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

# ==================== 🏗️ GPU Worker (核心逻辑) ====================

def gpu_worker(gpu_id, input_queue, result_queue, output_dirs):
    try:
        # 1. 初始化
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        print(f"🚀 [GPU {gpu_id}] Worker 就绪 (Overlap: {OVERLAP_DURATION}s)")

        # 2. 加载模型
        model_path = str(dac.utils.download(model_type=DAC_MODEL_TYPE))
        dac_model = dac.DAC.load(model_path)
        dac_model = dac_model.to(device)
        dac_model.eval()
        
        # 3. 循环处理
        while True:
            task = input_queue.get()
            if task is None: break
            
            filepath, split_type = task
            filename = Path(filepath).stem
            save_dir = output_dirs[split_type]
            save_path = os.path.join(save_dir, f"{filename}.pt")
            
            try:
                # --- A. 高速加载与 GPU 重采样 ---
                audio, sr = torchaudio.load(filepath)
                audio = audio.to(device)
                
                if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
                peak = audio.abs().max()
                if peak > 1.0: audio = audio / peak
                
                if sr != HIGH_SR:
                    audio_hr = AF.resample(audio, orig_freq=sr, new_freq=HIGH_SR)
                else:
                    audio_hr = audio
                
                total_samples = audio_hr.shape[1]
                if total_samples < int(MIN_DURATION * HIGH_SR):
                    result_queue.put(('skipped', filepath, "too_short"))
                    continue

                # --- B. Overlap 切分 (核心修改) ---
                chunk_valid_samples = int(CHUNK_DURATION * HIGH_SR)
                overlap_samples = int(OVERLAP_DURATION * HIGH_SR)
                
                # 计算需要多少个“有效”块
                num_chunks = math.ceil(total_samples / chunk_valid_samples)
                
                chunks_input = []
                
                for i in range(num_chunks):
                    # 理论有效区间: [i*valid, (i+1)*valid]
                    # 实际读取区间: [i*valid - overlap, (i+1)*valid + overlap]
                    
                    start_idx = i * chunk_valid_samples - overlap_samples
                    end_idx = start_idx + chunk_valid_samples + (2 * overlap_samples)
                    
                    # 边界处理 (Padding)
                    pad_left = 0
                    if start_idx < 0:
                        pad_left = -start_idx
                        start_idx = 0
                    
                    pad_right = 0
                    if end_idx > total_samples:
                        pad_right = end_idx - total_samples
                        end_idx = total_samples
                    
                    # 提取片段
                    chunk = audio_hr[:, start_idx:end_idx]
                    
                    # 实施 Padding (在 GPU 上极快)
                    if pad_left > 0 or pad_right > 0:
                        chunk = torch.nn.functional.pad(chunk, (pad_left, pad_right))
                    
                    chunks_input.append(chunk)

                # --- C. 批量编码 ---
                latents_hr_trimmed = []
                latents_lr_trimmed = []
                hop_length = None
                
                # 清理显存防 MIOpen 报错
                torch.cuda.empty_cache()

                for i in range(0, len(chunks_input), DAC_BATCH_SIZE):
                    batch_hr = torch.stack(chunks_input[i : i + DAC_BATCH_SIZE])
                    
                    # LR 模拟
                    batch_lr = AF.resample(AF.resample(batch_hr, HIGH_SR, LOW_SR), LOW_SR, HIGH_SR)
                    if batch_lr.shape[-1] != batch_hr.shape[-1]:
                        batch_lr = torch.nn.functional.pad(batch_lr, (0, batch_hr.shape[-1] - batch_lr.shape[-1]))

                    def encode_helper(wav):
                        signal = AudioSignal(wav, sample_rate=HIGH_SR)
                        if dac_model.sample_rate != HIGH_SR:
                            signal = signal.resample(dac_model.sample_rate)
                        with torch.no_grad():
                            z, _, _, _, _ = dac_model.encode(signal.audio_data)
                        return z

                    z_hr = encode_helper(batch_hr)
                    z_lr = encode_helper(batch_lr)
                    
                    # 计算 Trim 参数 (仅需一次)
                    if hop_length is None:
                        hop_length = batch_hr.shape[-1] / z_hr.shape[-1]
                        # 计算 Latent 层面需要切掉的帧数
                        # overlap_samples / hop_length
                        trim_frames = int(overlap_samples / hop_length)
                        # 计算 Latent 层面需要保留的有效帧数
                        valid_frames_latent = int(chunk_valid_samples / hop_length)

                    # --- D. 修剪与收集 (Trimming) ---
                    # z: [B, 1024, T]
                    # 我们需要切掉 [0:trim] 和 [-trim:]
                    # 保留 [trim : trim + valid]
                    
                    z_hr_valid = z_hr[..., trim_frames : trim_frames + valid_frames_latent]
                    z_lr_valid = z_lr[..., trim_frames : trim_frames + valid_frames_latent]
                    
                    for b in range(z_hr_valid.shape[0]):
                        latents_hr_trimmed.append(z_hr_valid[b].cpu())
                        latents_lr_trimmed.append(z_lr_valid[b].cpu())

                # --- E. 拼接 ---
                full_hr = torch.cat(latents_hr_trimmed, dim=-1)
                full_lr = torch.cat(latents_lr_trimmed, dim=-1)
                
                # 最后一次修剪: 确保总长度不超过原始音频
                assert hop_length is not None
                total_valid_frames = int(total_samples / hop_length)
                
                full_hr = full_hr[..., :total_valid_frames]
                full_lr = full_lr[..., :total_valid_frames]

                # --- F. 统计与保存 ---
                f_sum = full_hr.double().sum(dim=1) + full_lr.double().sum(dim=1)
                f_sq_sum = (full_hr.double()**2).sum(dim=1) + (full_lr.double()**2).sum(dim=1)
                f_count = full_hr.shape[1] + full_lr.shape[1]

                torch.save({
                    'hr_latent': full_hr.half(),
                    'lr_latent': full_lr.half(),
                    'metadata': {
                        'name': filename,
                        'path': filepath,
                        'duration': total_samples / HIGH_SR,
                        'sr': sr
                    }
                }, save_path)

                result_queue.put(('success', filepath, (f_sum, f_sq_sum, f_count)))

            except Exception as e:
                # traceback.print_exc()
                result_queue.put(('error', filepath, str(e)))

    except Exception as e:
        print(f"🔥 [GPU {gpu_id}] 崩溃: {e}")
        traceback.print_exc()

# ==================== 🧠 主控流程 ====================

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    print("=" * 60)
    print("🚀 JaT 数据预处理 V8 (Overlap-Stitch 拼接版)")
    print(f"⚙️  Config: 7s Chunk + 0.5s Overlap | Batch={DAC_BATCH_SIZE}")
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