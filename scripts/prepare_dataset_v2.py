"""
音频数据集预处理脚本 V2 (改进版)
改进:
1. 计算全局均值和标准差进行归一化
2. 保存为高效的 .pt (PyTorch) 格式
3. 保留完整歌曲，每首歌一个文件
4. 两阶段处理: Pass 1 计算统计量, Pass 2 转换并归一化
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from threading import Thread, Lock
import queue
import time

# DAC
import dac
from audiotools import AudioSignal


# ==================== 配置参数 ====================
SOURCE_DIRS = [
    '1_source_audio',  # FLAC 歌曲
    'extra_audio'      # WAV 音频
]

OUTPUT_DIR = 'data_processed'  # 新的输出目录
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# 音频参数
HIGH_SR = 48000      # 高分辨率采样率（DAC 需要）
LOW_SR = 16000       # 低分辨率采样率（模拟 MP3 质量）

# 数据集划分
VAL_RATIO = 0.1      # 验证集比例
RANDOM_SEED = 42     # 随机种子

# DAC 模型
DAC_MODEL_TYPE = "44khz"

# 并行处理
NUM_WORKERS = 8      # CPU workers 数量（充分利用64线程）
DAC_BATCH_SIZE = 16    # DAC 编码批处理大小（每个GPU）- 根据显存调整，24GB可尝试32-64
FILE_BATCH_SIZE = 128 # 文件批处理大小（流式处理）
USE_MULTI_GPU = True  # 是否使用多GPU并行

# 恢复机制
PROGRESS_DIR = 'processing_progress'  # 进度文件目录
QUEUE_FILE = 'file_queue.json'        # 固定的文件处理队列
PROGRESS_FILE = 'progress.json'       # 进度记录

# 全局锁（用于多线程进度更新）
progress_lock = Lock()


# ==================== 工具函数 ====================

def get_audio_files():
    """获取所有音频文件"""
    audio_files = []
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"⚠️  跳过不存在的目录: {source_dir}")
            continue

        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.flac', '.mp3')):
                    filepath = os.path.join(root, file)
                    audio_files.append(filepath)

    # 排序确保每次运行顺序一致
    audio_files.sort()
    print(f"✅ 找到 {len(audio_files)} 个音频文件")
    return audio_files


def init_progress_tracking():
    """初始化进度跟踪"""
    os.makedirs(PROGRESS_DIR, exist_ok=True)

    queue_path = os.path.join(PROGRESS_DIR, QUEUE_FILE)
    progress_path = os.path.join(PROGRESS_DIR, PROGRESS_FILE)

    return queue_path, progress_path


def save_file_queue(audio_files, train_files, val_files, queue_path):
    """保存文件处理队列（固定顺序）"""
    queue_data = {
        'all_files': audio_files,
        'train_files': train_files,
        'val_files': val_files,
        'total_count': len(audio_files),
        'train_count': len(train_files),
        'val_count': len(val_files),
        'created_at': str(np.datetime64('now'))
    }

    with open(queue_path, 'w', encoding='utf-8') as f:
        json.dump(queue_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 保存文件队列到: {queue_path}")


def load_file_queue(queue_path):
    """加载已有的文件处理队列"""
    if not os.path.exists(queue_path):
        return None

    with open(queue_path, 'r', encoding='utf-8') as f:
        queue_data = json.load(f)

    print(f"✅ 加载已有文件队列: {queue_data['total_count']} 个文件")
    return queue_data


def load_progress(progress_path):
    """加载处理进度"""
    if not os.path.exists(progress_path):
        return {
            'pass1_completed': False,
            'pass1_processed_files': [],
            'pass1_processed_batches': [],
            'global_stats': None,
            'pass2_train_processed': [],
            'pass2_val_processed': [],
            'pass2_train_completed': False,
            'pass2_val_completed': False,
        }

    with open(progress_path, 'r', encoding='utf-8') as f:
        progress = json.load(f)

    pass1_batches = len(progress.get('pass1_processed_batches', []))
    pass1_status = '完成' if progress['pass1_completed'] else f'{pass1_batches} 批次已处理'

    pass2_train_count = len(progress.get('pass2_train_processed', []))
    pass2_train_status = '完成' if progress.get('pass2_train_completed') else f'{pass2_train_count} 个文件已处理'

    pass2_val_count = len(progress.get('pass2_val_processed', []))
    pass2_val_status = '完成' if progress.get('pass2_val_completed') else f'{pass2_val_count} 个文件已处理'

    print("✅ 加载进度记录:")
    print(f"   Pass 1: {pass1_status}")
    print(f"   Pass 2 训练集: {pass2_train_status}")
    print(f"   Pass 2 验证集: {pass2_val_status}")

    return progress


def save_progress(progress, progress_path):
    """保存处理进度（线程安全）"""
    with progress_lock:
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)


def update_pass1_progress(progress, progress_path, batch_idx, processed_files):
    """更新 Pass 1 进度"""
    progress['pass1_processed_batches'].append(batch_idx)
    progress['pass1_processed_files'].extend(processed_files)
    save_progress(progress, progress_path)


def complete_pass1(progress, progress_path, stats):
    """标记 Pass 1 完成"""
    progress['pass1_completed'] = True
    progress['global_stats'] = stats
    save_progress(progress, progress_path)
    print("✅ Pass 1 已完成并保存")


def update_pass2_progress(progress, progress_path, split, processed_files):
    """更新 Pass 2 进度"""
    if split == 'train':
        progress['pass2_train_processed'].extend(processed_files)
    else:
        progress['pass2_val_processed'].extend(processed_files)
    save_progress(progress, progress_path)


def complete_pass2(progress, progress_path, split):
    """标记 Pass 2 完成"""
    if split == 'train':
        progress['pass2_train_completed'] = True
    else:
        progress['pass2_val_completed'] = True
    save_progress(progress, progress_path)
    print(f"✅ Pass 2 {split} 已完成并保存")


def get_available_gpus():
    """获取可用GPU列表"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return [f'cuda:{i}' for i in range(num_gpus)]
    return ['cpu']


def load_dac_model(device='cuda'):
    """加载DAC模型"""
    model_path = dac.utils.download(model_type=DAC_MODEL_TYPE)
    model = dac.DAC.load(str(model_path))  # Convert Path to str
    model = model.to(device)
    model.eval()
    return model


def audio_to_dac_latent(audio, dac_model, device='cuda'):
    """将音频编码为DAC latent"""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    signal = AudioSignal(audio_tensor, sample_rate=48000)

    if dac_model.sample_rate != 48000:
        signal = signal.resample(dac_model.sample_rate)

    signal = signal.to(device)

    with torch.no_grad():
        z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(signal.audio_data)

    return z.squeeze(0).cpu()  # [C, T] on CPU


def load_and_resample_audio(filepath):
    """
    步骤1: 加载音频并重采样（使用 torchaudio，更快）
    返回: [(high_audio_numpy, low_audio_numpy, metadata), ...]
    """
    try:
        # 使用 torchaudio 加载音频（更快）
        audio, sr = torchaudio.load(filepath)

        # 转换为单声道 [1, T]
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        filename = Path(filepath).stem

        # 所有音频都保持完整，不切分
        duration = audio.shape[1] / sr
        segment = audio.squeeze(0)  # [T]

        # 1. 生成高分辨率版本（48kHz）- 使用 torchaudio.functional.resample
        if sr != HIGH_SR:
            high_audio = AF.resample(segment, orig_freq=sr, new_freq=HIGH_SR)
        else:
            high_audio = segment

        # 2. 生成低分辨率版本（16kHz → 48kHz）
        low_audio = AF.resample(segment, orig_freq=sr, new_freq=LOW_SR)
        low_audio_upsampled = AF.resample(low_audio, orig_freq=LOW_SR, new_freq=HIGH_SR)

        # 3. 元数据
        metadata = {
            'name': filename,
            'source_file': str(filepath),
            'duration': duration,
        }

        # 转换为 numpy（后续 DAC 编码需要）
        audio_pairs = [(high_audio.numpy(), low_audio_upsampled.numpy(), metadata)]

        return audio_pairs

    except Exception as e:
        print(f"  ❌ 加载失败: {filepath} - {str(e)}")
        return []


def batch_encode_dac(audio_list, dac_model, device):
    """
    步骤2: 真正的批量 DAC 编码（GPU 加速）
    将一批音频 pad 并 stack 成 [B, 1, T]，一次性编码
    """
    if len(audio_list) == 0:
        return []

    # 1. 转换为 tensor 并找到最大长度
    audio_tensors = [torch.from_numpy(audio).float() for audio in audio_list]
    max_length = max(a.shape[0] for a in audio_tensors)

    # 2. Pad 到相同长度并 stack 成 [B, 1, T]
    padded_audios = []
    original_lengths = []
    for audio_tensor in audio_tensors:
        original_lengths.append(audio_tensor.shape[0])
        if audio_tensor.shape[0] < max_length:
            pad_length = max_length - audio_tensor.shape[0]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
        padded_audios.append(audio_tensor.unsqueeze(0))  # [1, T]

    # Stack: [B, 1, T]
    batch_audio = torch.stack(padded_audios, dim=0)  # [B, 1, T]

    # 3. 创建 AudioSignal 并移到 GPU
    signal = AudioSignal(batch_audio, sample_rate=48000)
    if dac_model.sample_rate != 48000:
        signal = signal.resample(dac_model.sample_rate)
    signal = signal.to(device)

    # 4. 批量编码（一次性处理整个 batch）
    with torch.no_grad():
        z, _, _, _, _ = dac_model.encode(signal.audio_data)  # [B, C, T']

    # 5. 拆分回单个样本（移回 CPU）
    latents = []
    for i in range(z.shape[0]):
        latent = z[i].cpu()  # [C, T']
        # 根据原始长度裁剪（DAC 压缩后的对应长度）
        # DAC 的压缩率约为 512x，所以不需要特别处理
        latents.append(latent)

    return latents


def process_file_batch_on_gpu(file_batch, gpu_device, batch_idx, total_batches, progress, progress_path):
    """
    在指定GPU上处理一批文件（带进度保存）
    """
    import time
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[{gpu_device}] 开始处理批次 {batch_idx+1}/{total_batches}")
    print(f"[{gpu_device}] 文件数: {len(file_batch)}")
    print(f"{'='*60}")

    # 每个进程加载自己的DAC模型
    print(f"[{gpu_device}] 加载 DAC 模型...")
    dac_model = load_dac_model(gpu_device)
    print(f"[{gpu_device}] ✅ DAC 模型已加载")

    # Step 1: 并行加载和重采样音频
    audio_pairs_batch = []

    print(f"[{gpu_device}] 加载并重采样音频文件...")
    load_start = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(load_and_resample_audio, file_batch),
            total=len(file_batch),
            desc=f"  [{gpu_device}] 加载音频",
            ncols=100
        ))

        for result in results:
            if result:
                audio_pairs_batch.extend(result)

    load_time = time.time() - load_start
    print(f"  ✅ [{gpu_device}] 加载完成: {len(audio_pairs_batch)} 个音频片段 (耗时: {load_time:.1f}s)")

    # Step 2: DAC 编码
    hr_latents_list = []
    lr_latents_list = []

    hr_audios = [pair[0] for pair in audio_pairs_batch]
    lr_audios = [pair[1] for pair in audio_pairs_batch]

    print(f"[{gpu_device}] 开始 DAC 编码 (批次大小: {DAC_BATCH_SIZE})...")
    encode_start = time.time()

    for i in tqdm(range(0, len(hr_audios), DAC_BATCH_SIZE),
                  desc=f"  [{gpu_device}] DAC 编码",
                  ncols=100):
        batch_hr = hr_audios[i:i+DAC_BATCH_SIZE]
        batch_lr = lr_audios[i:i+DAC_BATCH_SIZE]

        hr_latents = batch_encode_dac(batch_hr, dac_model, gpu_device)
        lr_latents = batch_encode_dac(batch_lr, dac_model, gpu_device)

        hr_latents_list.extend(hr_latents)
        lr_latents_list.extend(lr_latents)

    encode_time = time.time() - encode_start
    print(f"  ✅ [{gpu_device}] DAC 编码完成: {len(hr_latents_list)} 个 latent (耗时: {encode_time:.1f}s)")

    # 显示 GPU 显存使用情况
    if torch.cuda.is_available() and 'cuda' in gpu_device:
        gpu_id = int(gpu_device.split(':')[1]) if ':' in gpu_device else 0
        mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"  📊 [{gpu_device}] 显存: {mem_allocated:.2f}GB 已分配 / {mem_reserved:.2f}GB 已保留")

    # 保存进度
    print(f"[{gpu_device}] 保存进度...")
    update_pass1_progress(progress, progress_path, batch_idx, file_batch)

    # 清理内存
    del audio_pairs_batch, hr_audios, lr_audios, dac_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"✅ [{gpu_device}] 批次 {batch_idx+1}/{total_batches} 完成!")
    print(f"   总耗时: {total_time:.1f}s | 速度: {len(file_batch)/total_time:.2f} 文件/秒")
    print(f"{'='*60}\n")

    return hr_latents_list, lr_latents_list


def compute_global_statistics_multi_gpu(all_files, devices, progress, progress_path):
    """
    Pass 1: 使用多GPU计算全局统计量（带断点恢复）
    """
    print("\n" + "="*70)
    print("📊 Pass 1: 计算全局统计量（多GPU并行 + 断点恢复）")
    print("="*70)
    print(f"   使用 {len(devices)} 个 GPU: {devices}")
    print(f"   CPU workers: {NUM_WORKERS}")
    print(f"   文件批次大小: {FILE_BATCH_SIZE}")

    # 检查是否已完成
    if progress.get('pass1_completed'):
        print("✅ Pass 1 已在之前完成，使用已保存的统计量")
        return progress['global_stats']

    hr_latents_list = []
    lr_latents_list = []

    # 已处理的批次
    processed_batches = set(progress.get('pass1_processed_batches', []))

    # 分批处理文件
    num_batches = (len(all_files) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE

    # 将批次分配给不同的GPU
    batch_idx = 0

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []

        for batch_start in range(0, len(all_files), FILE_BATCH_SIZE):
            # 跳过已处理的批次
            if batch_idx in processed_batches:
                print(f"⏭️  跳过已处理批次 {batch_idx+1}/{num_batches}")
                batch_idx += 1
                continue

            batch_end = min(batch_start + FILE_BATCH_SIZE, len(all_files))
            file_batch = all_files[batch_start:batch_end]

            # 轮流分配给不同的GPU
            gpu_device = devices[batch_idx % len(devices)]

            print(f"\n提交批次 {batch_idx+1}/{num_batches} 到 {gpu_device} ({len(file_batch)} 个文件)...")

            # 提交任务
            future = executor.submit(
                process_file_batch_on_gpu,
                file_batch,
                gpu_device,
                batch_idx,
                num_batches,
                progress,
                progress_path
            )
            futures.append(future)
            batch_idx += 1

        # 收集结果
        print(f"\n等待所有 GPU 完成...")
        for future in as_completed(futures):
            hr_lats, lr_lats = future.result()
            hr_latents_list.extend(hr_lats)
            lr_latents_list.extend(lr_lats)

    # 计算全局统计量
    total_segments = len(hr_latents_list)
    print(f"\n计算全局均值和标准差 ({total_segments} 个片段)...")

    all_hr = torch.cat([lat.flatten() for lat in hr_latents_list])
    all_lr = torch.cat([lat.flatten() for lat in lr_latents_list])

    hr_mean = all_hr.mean().item()
    hr_std = all_hr.std().item()
    lr_mean = all_lr.mean().item()
    lr_std = all_lr.std().item()

    print(f"\n✅ 全局统计量:")
    print(f"   HR: mean={hr_mean:.6f}, std={hr_std:.6f}")
    print(f"   LR: mean={lr_mean:.6f}, std={lr_std:.6f}")
    print(f"   总片段数: {total_segments}")

    stats = {
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'lr_mean': lr_mean,
        'lr_std': lr_std,
        'num_samples': total_segments
    }

    # 保存完成状态
    complete_pass1(progress, progress_path, stats)

    return stats


def compute_global_statistics(all_files, dac_model, device, progress, progress_path):
    """
    Pass 1: 计算全局统计量（单GPU流式处理 + 断点恢复）
    """
    print("\n" + "="*70)
    print("📊 Pass 1: 计算全局统计量（单GPU流式处理 + 断点恢复）")
    print("="*70)
    print(f"   使用设备: {device}")
    print(f"   CPU workers: {NUM_WORKERS}")
    print(f"   文件批次大小: {FILE_BATCH_SIZE}")

    # 检查是否已完成
    if progress.get('pass1_completed'):
        print("✅ Pass 1 已在之前完成，使用已保存的统计量")
        return progress['global_stats']

    hr_latents_list = []
    lr_latents_list = []
    total_segments = 0

    # 已处理的批次
    processed_batches = set(progress.get('pass1_processed_batches', []))

    # 分批处理文件（降低内存占用）
    num_batches = (len(all_files) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE

    for batch_idx in range(num_batches):
        # 跳过已处理的批次
        if batch_idx in processed_batches:
            print(f"⏭️  跳过已处理批次 {batch_idx+1}/{num_batches}")
            continue

        start_idx = batch_idx * FILE_BATCH_SIZE
        end_idx = min(start_idx + FILE_BATCH_SIZE, len(all_files))
        file_batch = all_files[start_idx:end_idx]

        print(f"\n处理文件批次 {batch_idx+1}/{num_batches} ({len(file_batch)} 个文件)...")

        # Step 1: 并行加载和重采样音频
        audio_pairs_batch = []

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(load_and_resample_audio, file_batch),
                total=len(file_batch),
                desc=f"  加载批次 {batch_idx+1}"
            ))

            for result in results:
                if result:
                    audio_pairs_batch.extend(result)

        print(f"  ✅ 批次 {batch_idx+1}: {len(audio_pairs_batch)} 个片段")

        # Step 2: DAC 编码（本批次）
        hr_audios = [pair[0] for pair in audio_pairs_batch]
        lr_audios = [pair[1] for pair in audio_pairs_batch]

        for i in tqdm(range(0, len(hr_audios), DAC_BATCH_SIZE), desc=f"  DAC 编码批次 {batch_idx+1}"):
            batch_hr = hr_audios[i:i+DAC_BATCH_SIZE]
            batch_lr = lr_audios[i:i+DAC_BATCH_SIZE]

            hr_latents = batch_encode_dac(batch_hr, dac_model, device)
            lr_latents = batch_encode_dac(batch_lr, dac_model, device)

            hr_latents_list.extend(hr_latents)
            lr_latents_list.extend(lr_latents)

        total_segments += len(audio_pairs_batch)

        # 保存进度
        update_pass1_progress(progress, progress_path, batch_idx, file_batch)

        # 清理内存
        del audio_pairs_batch, hr_audios, lr_audios
        import gc
        gc.collect()

    # 计算全局统计量
    print(f"\n计算全局均值和标准差 ({total_segments} 个片段)...")

    all_hr = torch.cat([lat.flatten() for lat in hr_latents_list])
    all_lr = torch.cat([lat.flatten() for lat in lr_latents_list])

    hr_mean = all_hr.mean().item()
    hr_std = all_hr.std().item()
    lr_mean = all_lr.mean().item()
    lr_std = all_lr.std().item()

    print(f"\n✅ 全局统计量:")
    print(f"   HR: mean={hr_mean:.6f}, std={hr_std:.6f}")
    print(f"   LR: mean={lr_mean:.6f}, std={lr_std:.6f}")
    print(f"   总片段数: {total_segments}")

    stats = {
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'lr_mean': lr_mean,
        'lr_std': lr_std,
        'num_samples': total_segments
    }

    # 保存完成状态
    complete_pass1(progress, progress_path, stats)

    return stats


def process_and_save_dataset(train_files, val_files, dac_model, stats, device, progress, progress_path):
    """
    Pass 2: 处理并保存为 .pt 格式（多线程 + 断点恢复）
    """
    print("\n" + "="*70)
    print("💾 Pass 2: 转换并保存数据")
    print("="*70)
    print(f"   使用 {NUM_WORKERS} 个 CPU workers 加载音频")

    hr_mean = stats['hr_mean']
    hr_std = stats['hr_std']
    lr_mean = stats['lr_mean']
    lr_std = stats['lr_std']

    # 创建输出目录
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    def save_samples(files, output_dir, split_name):
        """保存样本到指定目录（流式处理 + 断点恢复）"""

        # 检查是否已完成
        if (split_name == '训练集' and progress.get('pass2_train_completed')) or \
           (split_name == '验证集' and progress.get('pass2_val_completed')):
            print(f"✅ {split_name} 已在之前完成，跳过")
            # 统计已有文件
            existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])
            return existing_files

        sample_count = 0
        processed_files_set = set(progress.get(f'pass2_{"train" if split_name == "训练集" else "val"}_processed', []))

        # 过滤掉已处理的文件
        remaining_files = [f for f in files if f not in processed_files_set]

        if len(remaining_files) < len(files):
            print(f"⏭️  {split_name}: 跳过 {len(files) - len(remaining_files)} 个已处理文件")

        if len(remaining_files) == 0:
            print(f"✅ {split_name} 所有文件已处理完成")
            complete_pass2(progress, progress_path, 'train' if split_name == '训练集' else 'val')
            return len([f for f in os.listdir(output_dir) if f.endswith('.pt')])

        # 分批处理
        num_batches = (len(remaining_files) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"开始处理 {split_name}: {len(remaining_files)} 个文件，分 {num_batches} 批")
        print(f"{'='*60}")

        for batch_idx in range(num_batches):
            import time
            batch_start = time.time()

            start_idx = batch_idx * FILE_BATCH_SIZE
            end_idx = min(start_idx + FILE_BATCH_SIZE, len(remaining_files))
            file_batch = remaining_files[start_idx:end_idx]

            print(f"\n📦 [{split_name}] 批次 {batch_idx+1}/{num_batches}")
            print(f"   文件数: {len(file_batch)}")
            print(f"   进度: {end_idx}/{len(remaining_files)} ({100*end_idx/len(remaining_files):.1f}%)")

            # Step 1: 并行加载音频
            audio_pairs_batch = []

            print(f"   🔄 加载音频...")
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(load_and_resample_audio, file_batch),
                    total=len(file_batch),
                    desc=f"  加载 {split_name}"
                ))

                for result in results:
                    if result:
                        audio_pairs_batch.extend(result)

            print(f"   ✅ 加载完成: {len(audio_pairs_batch)} 个音频片段")

            # Step 2: 批量 DAC 编码并保存
            batch_processed_files = []

            print(f"   🔄 DAC 编码并保存...")
            saved_count = 0
            for i in tqdm(range(0, len(audio_pairs_batch), DAC_BATCH_SIZE),
                         desc=f"  编码&保存",
                         ncols=100):
                batch_pairs = audio_pairs_batch[i:i+DAC_BATCH_SIZE]

                batch_hr = [pair[0] for pair in batch_pairs]
                batch_lr = [pair[1] for pair in batch_pairs]
                batch_meta = [pair[2] for pair in batch_pairs]

                # DAC 编码
                hr_latents = batch_encode_dac(batch_hr, dac_model, device)
                lr_latents = batch_encode_dac(batch_lr, dac_model, device)

                # 归一化并保存
                for hr_lat, lr_lat, metadata in zip(hr_latents, lr_latents, batch_meta):
                    hr_latent_norm = (hr_lat - hr_mean) / hr_std
                    lr_latent_norm = (lr_lat - lr_mean) / lr_std

                    sample_name = metadata['name']
                    save_path = os.path.join(output_dir, f"{sample_name}.pt")

                    torch.save({
                        'hr_latent': hr_latent_norm,
                        'lr_latent': lr_latent_norm,
                        'metadata': metadata,
                        'global_stats': {
                            'hr_mean': hr_mean,
                            'hr_std': hr_std,
                            'lr_mean': lr_mean,
                            'lr_std': lr_std
                        }
                    }, save_path)

                    sample_count += 1
                    saved_count += 1

            # 记录已处理的源文件
            batch_processed_files.extend(file_batch)

            # 更新进度
            update_pass2_progress(
                progress,
                progress_path,
                'train' if split_name == '训练集' else 'val',
                file_batch
            )

            # 清理内存
            del audio_pairs_batch
            import gc
            gc.collect()

            batch_time = time.time() - batch_start
            print(f"   ✅ 批次完成: 保存了 {saved_count} 个 .pt 文件")
            print(f"   ⏱️  耗时: {batch_time:.1f}s | 速度: {len(file_batch)/batch_time:.2f} 文件/秒")

        # 标记完成
        complete_pass2(progress, progress_path, 'train' if split_name == '训练集' else 'val')

        return sample_count

    # 保存训练集
    train_count = save_samples(train_files, TRAIN_DIR, "训练集")

    # 保存验证集
    val_count = save_samples(val_files, VAL_DIR, "验证集")

    print(f"\n✅ 保存完成:")
    print(f"   训练集: {train_count} 个样本")
    print(f"   验证集: {val_count} 个样本")

    return train_count, val_count


def save_dataset_info(stats, train_count, val_count):
    """保存数据集信息"""
    info = {
        'version': '2.0',
        'dataset': {
            'train_samples': train_count,
            'val_samples': val_count,
            'total_samples': train_count + val_count
        },
        'audio_config': {
            'high_sr': HIGH_SR,
            'low_sr': LOW_SR
        },
        'dac_config': {
            'model_type': DAC_MODEL_TYPE,
            'latent_dim': 1024,
            'compression_rate': 512
        },
        'global_stats': {
            'hr_mean': stats['hr_mean'],
            'hr_std': stats['hr_std'],
            'lr_mean': stats['lr_mean'],
            'lr_std': stats['lr_std']
        }
    }

    info_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 数据集信息保存到: {info_path}")


def main():
    print("=" * 70)
    print("🎵 音频数据集预处理 V2 (改进版 - 流式处理 + 断点恢复)")
    print("=" * 70)
    print("\n改进:")
    print("  1. ✅ 全局均值/标准差归一化")
    print("  2. ✅ 保存为高效 .pt 格式")
    print("  3. ✅ 每首歌一个文件")
    print("  4. ✅ 两阶段处理 (统计 → 转换)")
    print(f"  5. ✅ 多线程处理 ({NUM_WORKERS} workers)")
    print(f"  6. ✅ 流式处理 (批次大小: {FILE_BATCH_SIZE} 文件)")
    print(f"  7. ✅ 断点恢复 (可随时中断，下次继续)")
    print("=" * 70)

    # 0. 初始化进度跟踪
    queue_path, progress_path = init_progress_tracking()
    progress = load_progress(progress_path)

    # 确保进度文件存在（首次运行时创建）
    if not os.path.exists(progress_path):
        save_progress(progress, progress_path)

    # 检测可用GPU
    available_gpus = get_available_gpus()
    print(f"\n可用设备: {available_gpus}")

    if USE_MULTI_GPU and len(available_gpus) > 1:
        print(f"✅ 启用多GPU模式: {len(available_gpus)} 个 GPU")
        use_multi_gpu = True
        devices = available_gpus
    else:
        print(f"使用单GPU模式: {available_gpus[0]}")
        use_multi_gpu = False
        device = available_gpus[0]

    # 1. 获取或加载文件队列
    queue_data = load_file_queue(queue_path)

    if queue_data is None:
        # 首次运行：创建文件队列
        print("\n📝 首次运行，创建文件处理队列...")
        audio_files = get_audio_files()

        if len(audio_files) == 0:
            print("❌ 没有找到音频文件！")
            return

        # 划分训练/验证集（固定随机种子保证一致性）
        random.seed(RANDOM_SEED)
        random.shuffle(audio_files)

        split_idx = int(len(audio_files) * (1 - VAL_RATIO))
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]

        # 保存队列
        save_file_queue(audio_files, train_files, val_files, queue_path)
    else:
        # 从队列恢复
        print("\n📂 从已有队列恢复...")
        audio_files = queue_data['all_files']
        train_files = queue_data['train_files']
        val_files = queue_data['val_files']

    print(f"\n📊 数据集划分:")
    print(f"   总文件数: {len(audio_files)}")
    print(f"   训练集: {len(train_files)} 个文件")
    print(f"   验证集: {len(val_files)} 个文件")

    # 2. Pass 1: 计算全局统计量
    if use_multi_gpu:
        stats = compute_global_statistics_multi_gpu(audio_files, devices, progress, progress_path)
    else:
        if not progress.get('pass1_completed'):
            dac_model = load_dac_model(device)
            print(f"\n✅ DAC 加载完成 on {device}")
        else:
            dac_model = None
        stats = compute_global_statistics(audio_files, dac_model, device, progress, progress_path)

    # 3. Pass 2: 处理并保存
    if use_multi_gpu:
        # TODO: 实现多GPU的Pass 2
        print("\n⚠️  多GPU模式下，Pass 2 暂时使用单GPU")
        dac_model = load_dac_model(devices[0])
        train_count, val_count = process_and_save_dataset(
            train_files, val_files, dac_model, stats, devices[0], progress, progress_path
        )
    else:
        if dac_model is None:
            dac_model = load_dac_model(device)
            print(f"\n✅ DAC 加载完成 on {device}")
        train_count, val_count = process_and_save_dataset(
            train_files, val_files, dac_model, stats, device, progress, progress_path
        )

    # 4. 保存数据集信息
    save_dataset_info(stats, train_count, val_count)

    # 7. 总结
    print("\n" + "=" * 70)
    print("🎉 预处理完成！")
    print("=" * 70)
    print(f"\n📦 输出目录: {OUTPUT_DIR}/")
    print(f"   ├── train/          ({train_count} 个 .pt 文件)")
    print(f"   ├── val/            ({val_count} 个 .pt 文件)")
    print(f"   └── dataset_info.json")
    print(f"\n📊 每个 .pt 文件包含:")
    print(f"   - hr_latent: 归一化的高分辨率 latent [C, T]")
    print(f"   - lr_latent: 归一化的低分辨率 latent [C, T]")
    print(f"   - metadata: 文件元信息")
    print(f"   - global_stats: 全局统计量 (用于反归一化)")
    print("\n💡 使用方法:")
    print("   data = torch.load('data_processed/train/song_001.pt')")
    print("   hr_latent = data['hr_latent']  # 已归一化")
    print("   lr_latent = data['lr_latent']  # 已归一化")
    print("=" * 70)


if __name__ == '__main__':
    main()
