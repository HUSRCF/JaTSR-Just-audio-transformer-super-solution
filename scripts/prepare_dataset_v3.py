"""
音频数据集预处理脚本 V3 (高性能修复版 - 分块处理)
修复:
1. 移除嵌套进程池，改用 PyTorch DataLoader 处理多进程数据加载
2. 移除重复模型加载，每个 GPU 进程只加载一次模型
3. 解决死锁和资源利用率低的问题
4. 修复 Pylance 类型检查错误 (torch.clamp 和 tolist)
5. 修复 "daemonic processes are not allowed to have children" 错误 (替换 mp.Pool 为 mp.Process)
6. 修复 Literal[0] has no attribute 'double' 错误 (正确初始化累加器)
7. 新增：分块处理 (Chunking) 以解决 OOM 问题
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
CHUNK_DURATION = 15.0  # 每个切片 15 秒，避免 OOM
MIN_DURATION = 1.0     # 最小长度，小于此长度丢弃

# 数据集划分
VAL_RATIO = 0.1
RANDOM_SEED = 42

# DAC 模型
DAC_MODEL_TYPE = "44khz"

# 并行配置
# 每个 GPU 分配的 CPU 核心数用于数据加载
# 建议：总 CPU 核心数 / GPU 数量 - 2
NUM_WORKERS_PER_GPU = 8  
DAC_BATCH_SIZE = 32      # 增大 Batch Size 以喂饱 W7900

# ==================== 核心 Dataset 类 ====================

class AudioLoadingDataset(Dataset):
    """
    负责音频的读取、重采样和分块，运行在 CPU Worker 进程中
    """
    def __init__(self, file_list, chunk_duration=CHUNK_DURATION):
        self.files = file_list
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * HIGH_SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        try:
            # 1. 加载音频
            # info = torchaudio.info(filepath) # 可选：先只读元数据判断长度
            
            audio, sr = torchaudio.load(filepath)
            
            # 转单声道
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # 归一化 (防止破音)
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max()

            segment = audio.squeeze(0) # [T]
            
            # 2. 重采样到 48k (HR)
            if sr != HIGH_SR:
                high_audio = AF.resample(segment, orig_freq=sr, new_freq=HIGH_SR)
            else:
                high_audio = segment
                
            # 检查长度
            total_samples = high_audio.shape[0]
            if total_samples < int(MIN_DURATION * HIGH_SR):
                return {'valid': False} # 太短

            # 3. 分块逻辑 (Chunking)
            # 为了统计准确性，我们尽可能多地切分出片段
            # 训练时通常随机切，但这里是预处理，我们要把整首歌存下来
            # 策略：按顺序切分，最后一段如果太短则丢弃或Pad
            
            chunks = []
            
            # 如果音频比 chunk_duration 短，pad 一下
            if total_samples < self.chunk_samples:
                pad_len = self.chunk_samples - total_samples
                high_audio_padded = torch.nn.functional.pad(high_audio, (0, pad_len))
                chunks.append(high_audio_padded)
            else:
                # 正常切分
                num_chunks = math.ceil(total_samples / self.chunk_samples)
                for i in range(num_chunks):
                    start = i * self.chunk_samples
                    end = min(start + self.chunk_samples, total_samples)
                    chunk = high_audio[start:end]
                    
                    # 只有最后一段可能长度不足
                    if chunk.shape[0] < self.chunk_samples:
                        pad_len = self.chunk_samples - chunk.shape[0]
                        chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                    
                    chunks.append(chunk)
            
            # 4. 对每个 Chunk 制作 LR
            processed_chunks = []
            filename = Path(filepath).stem
            
            for i, chunk_hr in enumerate(chunks):
                # 制作 LR (16k -> 48k)
                low_audio = AF.resample(chunk_hr, orig_freq=HIGH_SR, new_freq=LOW_SR)
                low_audio_upsampled = AF.resample(low_audio, orig_freq=LOW_SR, new_freq=HIGH_SR)
                
                # 确保长度一致 (重采样可能导致 1-2 sample 的误差)
                if low_audio_upsampled.shape[0] != chunk_hr.shape[0]:
                    min_len = min(low_audio_upsampled.shape[0], chunk_hr.shape[0])
                    low_audio_upsampled = low_audio_upsampled[:min_len]
                    chunk_hr = chunk_hr[:min_len]

                processed_chunks.append({
                    'hr': chunk_hr,
                    'lr': low_audio_upsampled,
                    'name': f"{filename}_part{i:03d}", # 加上分块后缀
                    'path': str(filepath),
                    'duration': chunk_hr.shape[0] / HIGH_SR,
                    'valid': True
                })
            
            # 返回 list，collate_fn 需要处理 list of lists
            return processed_chunks

        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return [{'valid': False}]

def chunk_collate(batch):
    """
    处理分块后的数据
    batch 是一个 list of lists (每个文件产生多个 chunk)
    我们需要把它展平成一个大的 batch
    """
    flat_batch = []
    for chunks in batch:
        for chunk in chunks:
            if chunk['valid']:
                flat_batch.append(chunk)
                
    if len(flat_batch) == 0:
        return None

    # Stack (现在所有 chunk 长度应该是一样的，或者接近)
    # 因为我们做了 Padding，长度应该是 chunk_samples
    
    # 二次检查长度并 Pad (以防万一重采样误差)
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
        
        metadata.append({
            'name': b['name'],
            'path': b['path'],
            'len': curr_len 
        })
        
    return {
        'hr': torch.stack(hr_batch),
        'lr': torch.stack(lr_batch),
        'metadata': metadata
    }

# ==================== GPU 工作进程 ====================

def gpu_worker(gpu_id, files, output_dir, split_name, stats, mode, return_dict):
    """
    持久化的 GPU 工作进程
    mode: 'stats' (Pass 1) 或 'process' (Pass 2)
    return_dict: 用于返回结果的共享字典
    """
    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 [GPU {gpu_id}] 启动 ({split_name}, 模式: {mode})...")
        
        # 1. 加载模型 (只做一次!)
        print(f"   [GPU {gpu_id}] 加载 DAC 模型...")
        model_path = dac.utils.download(model_type=DAC_MODEL_TYPE)
        dac_model = dac.DAC.load(str(model_path)).to(device)
        dac_model.eval()
        
        # 2. 创建 DataLoader
        dataset = AudioLoadingDataset(files, chunk_duration=CHUNK_DURATION)
        # 注意: batch_size 这里指“文件数”，实际产生的 chunk 数可能是 batch_size * chunks_per_file
        # 为了控制显存，我们稍微减小 DataLoader 的 batch_size
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
        
        # 统计变量
        local_sum = torch.zeros(1024, dtype=torch.float64)
        local_sq_sum = torch.zeros(1024, dtype=torch.float64)
        local_count = 0
        save_count = 0
        
        # 如果是处理模式，准备归一化参数
        if mode == 'process' and stats is not None:
            hr_mean = torch.tensor(stats['hr_mean']).to(device).view(-1, 1)
            hr_std = torch.tensor(stats['hr_std']).to(device).view(-1, 1)
            lr_mean = torch.tensor(stats['lr_mean']).to(device).view(-1, 1)
            lr_std = torch.tensor(stats['lr_std']).to(device).view(-1, 1)
        
        # 3. 主循环
        desc = f"[GPU {gpu_id}] {split_name}"
        for batch in tqdm(loader, desc=desc, position=gpu_id, leave=False):
            if batch is None: continue
            
            # batch['hr'] 包含的是扁平化后的 chunk 列表
            # 如果 chunk 太多导致显存不够，我们需要在这里再次切分 (micro-batch)
            
            total_chunks = batch['hr'].shape[0]
            
            # 使用 Micro-batch 防止 OOM
            micro_batch_size = DAC_BATCH_SIZE
            
            for i in range(0, total_chunks, micro_batch_size):
                hr_wav = batch['hr'][i:i+micro_batch_size].unsqueeze(1).to(device)
                lr_wav = batch['lr'][i:i+micro_batch_size].unsqueeze(1).to(device)
                batch_meta = batch['metadata'][i:i+micro_batch_size]
                
                # DAC Encode
                with torch.no_grad():
                    if dac_model.sample_rate != HIGH_SR:
                        hr_wav = AF.resample(hr_wav, HIGH_SR, dac_model.sample_rate)
                        lr_wav = AF.resample(lr_wav, HIGH_SR, dac_model.sample_rate)
                        
                    hr_z = dac_model.encode(hr_wav)[0] 
                    lr_z = dac_model.encode(lr_wav)[0]
                
                # 处理结果
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
    print("🚀 JaT 数据预处理 V3 (W7900 双卡全速版 - 分块防OOM版)")
    print("=" * 60)
    print(f"Chunk Duration: {CHUNK_DURATION}s")
    
    # 1. 获取文件
    files = []
    for d in SOURCE_DIRS:
        if os.path.exists(d):
            files.extend([os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(('.wav','.flac'))])
    files.sort()
    random.shuffle(files)
    print(f"找到 {len(files)} 个音频文件")
    
    # 划分 GPU 任务
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1 # Fallback CPU
    
    chunk_size = len(files) // num_gpus
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    if len(file_chunks) > num_gpus:
        file_chunks[-2].extend(file_chunks[-1])
        file_chunks.pop()
        
    print(f"任务分配: {num_gpus} GPUs, 每个处理约 {len(file_chunks[0])} 文件")
    
    # ================= Pass 1: 统计 =================
    print("\n📊 Pass 1: 全局统计量计算...")
    
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
        
    total_sum = torch.zeros(1024, dtype=torch.float64)
    total_sq_sum = torch.zeros(1024, dtype=torch.float64)
    total_count = 0
    
    if len(return_dict) != num_gpus or any(v is None for v in return_dict.values()):
        print("❌ Pass 1 失败，部分进程未返回结果")
        return

    for i in range(num_gpus):
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
    
    def to_list_safe(tensor_or_scalar):
        if hasattr(tensor_or_scalar, 'tolist'):
            return tensor_or_scalar.tolist()
        elif isinstance(tensor_or_scalar, (float, int)):
            return [float(tensor_or_scalar)]
        elif torch.is_tensor(tensor_or_scalar):
            return tensor_or_scalar.float().tolist()
        return float(tensor_or_scalar)

    stats = {
        'hr_mean': to_list_safe(global_mean),
        'hr_std': to_list_safe(global_std),
        'lr_mean': to_list_safe(global_mean),
        'lr_std': to_list_safe(global_std)
    }
    
    with open('data_processed/global_stats.json', 'w') as f:
        json.dump(stats, f)
    print("✅ 统计量已保存")
    
    # ================= Pass 2: 转换 =================
    print("\n💾 Pass 2: 编码并保存...")
    
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
    
    print("\n🎉 全部完成！")

if __name__ == '__main__':
    main()