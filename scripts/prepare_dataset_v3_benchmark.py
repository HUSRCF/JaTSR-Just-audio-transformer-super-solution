"""
音频数据集预处理脚本 V3 (Benchmark 版)
修改内容: 添加了详细的时间戳记录，用于分析 数据加载 vs GPU推理 vs 磁盘IO 的瓶颈。
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
from torch.utils.data import Dataset, DataLoader
import time
import math

# DAC
import dac
from audiotools import AudioSignal

# ==================== 配置参数 ====================
SOURCE_DIRS = [
    '1_source_audio',
    'extra_audio'
]

OUTPUT_DIR = 'data_processed'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# 音频参数
HIGH_SR = 48000
LOW_SR = 16000
CHUNK_DURATION = 15.0
MIN_DURATION = 1.0

# 数据集划分
VAL_RATIO = 0.1
RANDOM_SEED = 42

# DAC 模型
DAC_MODEL_TYPE = "44khz"

# 并行配置
NUM_WORKERS_PER_GPU = 8  
DAC_BATCH_SIZE = 32

# ==================== 核心 Dataset 类 (保持不变) ====================

class AudioLoadingDataset(Dataset):
    def __init__(self, file_list, chunk_duration=CHUNK_DURATION):
        self.files = file_list
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * HIGH_SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        try:
            audio, sr = torchaudio.load(filepath)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()

            segment = audio.squeeze(0)
            if sr != HIGH_SR:
                high_audio = AF.resample(segment, orig_freq=sr, new_freq=HIGH_SR)
            else:
                high_audio = segment
                
            total_samples = high_audio.shape[0]
            if total_samples < int(MIN_DURATION * HIGH_SR):
                return {'valid': False}

            chunks = []
            if total_samples < self.chunk_samples:
                pad_len = self.chunk_samples - total_samples
                high_audio_padded = torch.nn.functional.pad(high_audio, (0, pad_len))
                chunks.append(high_audio_padded)
            else:
                num_chunks = math.ceil(total_samples / self.chunk_samples)
                for i in range(num_chunks):
                    start = i * self.chunk_samples
                    end = min(start + self.chunk_samples, total_samples)
                    chunk = high_audio[start:end]
                    if chunk.shape[0] < self.chunk_samples:
                        pad_len = self.chunk_samples - chunk.shape[0]
                        chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                    chunks.append(chunk)
            
            processed_chunks = []
            filename = Path(filepath).stem
            for i, chunk_hr in enumerate(chunks):
                low_audio = AF.resample(chunk_hr, orig_freq=HIGH_SR, new_freq=LOW_SR)
                low_audio_upsampled = AF.resample(low_audio, orig_freq=LOW_SR, new_freq=HIGH_SR)
                if low_audio_upsampled.shape[0] != chunk_hr.shape[0]:
                    min_len = min(low_audio_upsampled.shape[0], chunk_hr.shape[0])
                    low_audio_upsampled = low_audio_upsampled[:min_len]
                    chunk_hr = chunk_hr[:min_len]

                processed_chunks.append({
                    'hr': chunk_hr,
                    'lr': low_audio_upsampled,
                    'name': f"{filename}_part{i:03d}",
                    'path': str(filepath),
                    'duration': chunk_hr.shape[0] / HIGH_SR,
                    'valid': True
                })
            return processed_chunks
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return [{'valid': False}]

def chunk_collate(batch):
    flat_batch = []
    for chunks in batch:
        for chunk in chunks:
            if chunk['valid']:
                flat_batch.append(chunk)
    if len(flat_batch) == 0:
        return None
    
    max_len = max([b['hr'].shape[0] for b in flat_batch])
    hr_batch = []
    lr_batch = []
    metadata = []
    for b in flat_batch:
        curr_len = b['hr'].shape[0]
        pad_len = max_len - curr_len
        if pad_len > 0:
            hr_padded = torch.nn.functional.pad(b['hr'], (0, pad_len))
            lr_padded = torch.nn.functional.pad(b['lr'], (0, pad_len))
        else:
            hr_padded = b['hr']
            lr_padded = b['lr']
        hr_batch.append(hr_padded)
        lr_batch.append(lr_padded)
        metadata.append({'name': b['name'], 'path': b['path'], 'len': curr_len})
        
    return {'hr': torch.stack(hr_batch), 'lr': torch.stack(lr_batch), 'metadata': metadata}

# ==================== GPU 工作进程 (修改重点) ====================

def gpu_worker(gpu_id, files, output_dir, split_name, stats, mode, return_dict):
    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 [GPU {gpu_id}] 启动 ({split_name})...")
        
        # 加载模型
        model_path = dac.utils.download(model_type=DAC_MODEL_TYPE)
        dac_model = dac.DAC.load(str(model_path)).to(device)
        dac_model.eval()
        
        dataset = AudioLoadingDataset(files, chunk_duration=CHUNK_DURATION)
        file_batch_size = max(1, DAC_BATCH_SIZE // 4) 
        
        loader = DataLoader(
            dataset, 
            batch_size=file_batch_size, 
            shuffle=False, 
            num_workers=NUM_WORKERS_PER_GPU, 
            collate_fn=chunk_collate,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        local_sum = torch.zeros(1024, dtype=torch.float64)
        local_sq_sum = torch.zeros(1024, dtype=torch.float64)
        local_count = 0
        save_count = 0
        
        if mode == 'process' and stats is not None:
            hr_mean = torch.tensor(stats['hr_mean']).to(device).view(-1, 1)
            hr_std = torch.tensor(stats['hr_std']).to(device).view(-1, 1)
            lr_mean = torch.tensor(stats['lr_mean']).to(device).view(-1, 1)
            lr_std = torch.tensor(stats['lr_std']).to(device).view(-1, 1)
        
        desc = f"[GPU {gpu_id}] {split_name}"
        
        # ----------------- Benchmark 初始化 -----------------
        t_start_loop = time.time()
        time_metrics = {
            'data_load': 0.0, # 等待 DataLoader 的时间
            'inference': 0.0, # GPU 前向传播时间
            'io_save': 0.0,   # 保存/统计计算时间
            'total_items': 0
        }
        
        # 记录上一次循环结束的时间点
        last_batch_end_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, position=gpu_id, leave=False)):
            # 1. 记录数据加载耗时 (当前时间 - 上一次循环结束时间)
            t_data_ready = time.time()
            dt_data = t_data_ready - last_batch_end_time
            time_metrics['data_load'] += dt_data
            
            if batch is None: 
                last_batch_end_time = time.time()
                continue
            
            total_chunks = batch['hr'].shape[0]
            time_metrics['total_items'] += total_chunks
            
            # Micro-batch 处理
            micro_batch_size = DAC_BATCH_SIZE
            
            for i in range(0, total_chunks, micro_batch_size):
                # 数据搬运也算在推理准备中
                t_infer_start = time.time()
                
                hr_wav = batch['hr'][i:i+micro_batch_size].unsqueeze(1).to(device)
                lr_wav = batch['lr'][i:i+micro_batch_size].unsqueeze(1).to(device)
                batch_meta = batch['metadata'][i:i+micro_batch_size]
                
                # DAC Encode (核心推理)
                with torch.no_grad():
                    if dac_model.sample_rate != HIGH_SR:
                        hr_wav = AF.resample(hr_wav, HIGH_SR, dac_model.sample_rate)
                        lr_wav = AF.resample(lr_wav, HIGH_SR, dac_model.sample_rate)
                    hr_z = dac_model.encode(hr_wav)[0] 
                    lr_z = dac_model.encode(lr_wav)[0]
                
                t_infer_end = time.time()
                time_metrics['inference'] += (t_infer_end - t_infer_start)
                
                # IO / 统计 / 后处理
                t_io_start = time.time()
                
                if mode == 'stats':
                    B, C, T = hr_z.shape
                    flat_z = hr_z.permute(1, 0, 2).reshape(C, -1).double().cpu()
                    local_sum += flat_z.sum(dim=1)
                    local_sq_sum += (flat_z ** 2).sum(dim=1)
                    local_count += flat_z.shape[1]
                    
                elif mode == 'process':
                    hr_z_norm = (hr_z - hr_mean) / hr_std
                    lr_z_norm = (lr_z - lr_mean) / lr_std
                    
                    for j in range(hr_z.shape[0]):
                        meta = batch_meta[j]
                        save_path = os.path.join(output_dir, f"{meta['name']}.pt")
                        torch.save({
                            'hr_latent': hr_z_norm[j].half().cpu(),
                            'lr_latent': lr_z_norm[j].half().cpu(),
                            'metadata': meta
                        }, save_path)
                        save_count += 1
                        
                t_io_end = time.time()
                time_metrics['io_save'] += (t_io_end - t_io_start)

            # 记录 Batch 结束时间
            last_batch_end_time = time.time()
            
            # 每 10 个 Batch 打印一次详细 Benchmark 信息
            if batch_idx > 0 and batch_idx % 10 == 0:
                elapsed = last_batch_end_time - t_start_loop
                items_per_sec = time_metrics['total_items'] / elapsed
                
                # 计算各部分占比
                total_op_time = time_metrics['data_load'] + time_metrics['inference'] + time_metrics['io_save']
                load_pct = (time_metrics['data_load'] / total_op_time) * 100
                infer_pct = (time_metrics['inference'] / total_op_time) * 100
                io_pct = (time_metrics['io_save'] / total_op_time) * 100
                
                tqdm.write(
                    f"[GPU {gpu_id}] Batch {batch_idx} | Speed: {items_per_sec:.2f} chunks/s | "
                    f"Load: {load_pct:.1f}% | Infer: {infer_pct:.1f}% | IO: {io_pct:.1f}%"
                )

        if mode == 'stats':
            return_dict[gpu_id] = (local_sum, local_sq_sum, local_count)
        else:
            return_dict[gpu_id] = save_count
            
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] 发生异常: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = None

# ==================== 主流程 ====================

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("=" * 60)
    print("🚀 JaT 数据预处理 V3 (Benchmark 版)")
    print("=" * 60)
    
    files = []
    for d in SOURCE_DIRS:
        if os.path.exists(d):
            files.extend([os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(('.wav','.flac'))])
    files.sort()
    random.shuffle(files)
    print(f"找到 {len(files)} 个音频文件")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1
    
    chunk_size = len(files) // num_gpus
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    if len(file_chunks) > num_gpus:
        file_chunks[-2].extend(file_chunks[-1])
        file_chunks.pop()
    
    # ================= Pass 1 =================
    print("\n📊 Pass 1: 全局统计量计算...")
    t_p1_start = time.time() # 计时
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(num_gpus):
        p = mp.Process(
            target=gpu_worker, 
            args=(i, file_chunks[i], None, 'Pass 1', None, 'stats', return_dict)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    t_p1_end = time.time()
    print(f"⏱️ Pass 1 完成，耗时: {t_p1_end - t_p1_start:.2f} 秒")
        
    # (省略部分统计合并代码，保持逻辑一致)
    total_sum = torch.zeros(1024, dtype=torch.float64)
    total_sq_sum = torch.zeros(1024, dtype=torch.float64)
    total_count = 0
    
    for i in range(num_gpus):
        if i in return_dict and return_dict[i] is not None:
            s, sq, c = return_dict[i]
            total_sum += s
            total_sq_sum += sq
            total_count += c
            
    total_sum = total_sum.double()
    total_sq_sum = total_sq_sum.double()
    
    global_mean = total_sum / float(total_count)
    global_var = (total_sq_sum / float(total_count)) - (global_mean ** 2)
    global_var = torch.as_tensor(global_var) 
    global_std = torch.sqrt(torch.clamp(global_var, min=1e-6))
    
    def to_list_safe(t):
        if hasattr(t, 'tolist'): return t.tolist()
        if isinstance(t, (float, int)): return [float(t)]
        if torch.is_tensor(t): return t.float().tolist()
        return float(t)

    stats = {
        'hr_mean': to_list_safe(global_mean),
        'hr_std': to_list_safe(global_std),
        'lr_mean': to_list_safe(global_mean),
        'lr_std': to_list_safe(global_std)
    }
    
    with open('data_processed/global_stats.json', 'w') as f:
        json.dump(stats, f)
    
    # ================= Pass 2 =================
    print("\n💾 Pass 2: 编码并保存...")
    t_p2_start = time.time() # 计时
    
    split_idx = int(len(files) * (1 - VAL_RATIO))
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    train_chunks = np.array_split(train_files, num_gpus)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    return_dict = manager.dict()
    processes = []

    for i in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(i, train_chunks[i], TRAIN_DIR, 'Train Set', stats, 'process', return_dict)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    os.makedirs(VAL_DIR, exist_ok=True)
    val_return_dict = manager.dict()
    p_val = mp.Process(
        target=gpu_worker,
        args=(0, val_files, VAL_DIR, 'Val Set', stats, 'process', val_return_dict)
    )
    p_val.start()
    p_val.join()
    
    t_p2_end = time.time()
    print(f"⏱️ Pass 2 完成，耗时: {t_p2_end - t_p2_start:.2f} 秒")
    print("\n🎉 全部完成！")

if __name__ == '__main__':
    main()