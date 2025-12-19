"""
JaT 数据修正脚本 (LR Only Update) - 最终修复版
================================================
功能: 
1. 读取已生成的 .pt 文件
2. 保持 hr_latent 不变 (节省50%时间)
3. 重新读取源音频 -> Resample 32k -> Encode -> 覆盖 lr_latent
4. 使用 "redone" 状态标记避免重复处理

核心配置:
- LOW_SR: 32000 (针对 MP3 修复优化)
- Batch Size: 8
"""

import os
# 环境变量配置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True"

import torch
import torchaudio
import torchaudio.functional as AF
from pathlib import Path
from tqdm import tqdm
import json
import torch.multiprocessing as mp
import time
import math
import traceback
import dac
from audiotools import AudioSignal

# ==================== ⚙️ 配置参数 ====================

# 指向你之前生成的目录 (脚本会原地修改这里面的文件)
OUTPUT_DIR = 'data_processed_v13_final' 
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

# 新的日志文件和统计文件
LOG_FILE = os.path.join(OUTPUT_DIR, 'processed_files_redone.jsonl') 
STATS_FILE = os.path.join(OUTPUT_DIR, 'running_stats_redone.pt')
FINAL_STATS_JSON = os.path.join(OUTPUT_DIR, 'global_stats_redone.json')

# 关键参数：32kHz (16k Bandwidth)
HIGH_SR = 48000
LOW_SR = 32000

# 保持与 V13 一致的切片参数
CHUNK_DURATION = 7.0    
OVERLAP_DURATION = 0.5  
MIN_DURATION = 1.0     

DAC_BATCH_SIZE = 1  
DAC_MODEL_TYPE = "44khz"

# ==================== 🛠️ 工具函数 ====================

def save_jsonl(data, filepath):
    with open(filepath, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def load_redone_files(filepath):
    """只加载状态为 'redone' 的文件"""
    processed = set()
    if not os.path.exists(filepath):
        return processed
    print(f"📂 读取修正日志: {filepath}")
    with open(filepath, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                if data.get('status') == 'redone':
                    processed.add(data['path'])
            except Exception:
                pass
    return processed

# ==================== 🏗️ GPU Worker (修正版) ====================

def gpu_worker(gpu_id, input_queue, result_queue):
    try:
        # 1. 初始化
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(device)
        
        print(f"🚀 [GPU {gpu_id}] Refine Worker 就绪 (Target: {LOW_SR}Hz)")

        # 2. 加载模型
        model_path = str(dac.utils.download(model_type=DAC_MODEL_TYPE))
        dac_model = dac.DAC.load(model_path)
        dac_model = dac_model.to(device)
        dac_model.eval()
        
        # 3. 循环处理
        while True:
            task = input_queue.get()
            if task is None: break
            
            pt_path = task # 这里的 task 是 .pt 文件的路径
            
            try:
                # --- A. 读取现有的 .pt 文件 ---
                existing_data = torch.load(pt_path, map_location='cpu', weights_only=False)
                
                original_src_path = existing_data['metadata']['path']
                hr_latent_ref = existing_data['hr_latent'] # [1024, T]
                sr = existing_data['metadata']['sr']
                
                if not os.path.exists(original_src_path):
                    result_queue.put(('error', pt_path, f"Source not found: {original_src_path}"))
                    continue

                # --- B. 重新加载原始音频 (CPU) ---
                audio, actual_sr = torchaudio.load(original_src_path)

                # 验证采样率
                if actual_sr != sr:
                    result_queue.put(('error', pt_path, f"SR mismatch: metadata={sr}, actual={actual_sr}"))
                    continue

                # 预处理
                if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
                peak = audio.abs().max()
                if peak > 1.0: audio = audio / peak

                total_samples_original = audio.shape[1]
                duration_sec = total_samples_original / sr
                
                # --- C. 切分逻辑 ---
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

                # --- D. GPU 处理 (只做 LR) ---
                latents_lr_trimmed = []
                hop_length_48k = None 
                
                torch.cuda.empty_cache()

                for i in range(0, len(chunks_raw_cpu), DAC_BATCH_SIZE):
                    batch_cpu_list = chunks_raw_cpu[i : i + DAC_BATCH_SIZE]
                    
                    # 对齐 Batch
                    target_len = batch_cpu_list[0].shape[-1]
                    clean_batch_list = []
                    for c in batch_cpu_list:
                        if c.shape[-1] != target_len:
                            c = torch.nn.functional.pad(c, (0, target_len - c.shape[-1]))
                        clean_batch_list.append(c)
                        
                    batch_raw = torch.stack(clean_batch_list).to(device, non_blocking=True)
                    
                    # 1. 重采样 (48k -> 32k -> 48k)
                    if sr != HIGH_SR:
                        batch_hr = AF.resample(batch_raw, orig_freq=sr, new_freq=HIGH_SR)
                    else:
                        batch_hr = batch_raw
                    
                    batch_lr = AF.resample(AF.resample(batch_hr, HIGH_SR, LOW_SR), LOW_SR, HIGH_SR)
                    
                    if batch_lr.shape[-1] != batch_hr.shape[-1]:
                        batch_lr = torch.nn.functional.pad(batch_lr, (0, batch_hr.shape[-1] - batch_lr.shape[-1]))

                    # 2. Encode LR Only
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        signal_lr = AudioSignal(batch_lr, sample_rate=HIGH_SR)
                        if dac_model.sample_rate != HIGH_SR:
                            signal_lr = signal_lr.resample(dac_model.sample_rate)
                        
                        with torch.no_grad():
                            z_lr, _, _, _, _ = dac_model.encode(signal_lr.audio_data)

                    # 3. Trim
                    if hop_length_48k is None:
                        # 用 batch_hr 估算 hop_length 保证对齐
                        hop_length_48k = batch_hr.shape[-1] / z_lr.shape[-1]

                        overlap_samples_48k = int(OVERLAP_DURATION * HIGH_SR)
                        valid_samples_48k = int(CHUNK_DURATION * HIGH_SR)
                        trim_frames = int(overlap_samples_48k / hop_length_48k)
                        valid_frames_latent = int(valid_samples_48k / hop_length_48k)

                    z_lr_valid = z_lr[..., trim_frames : trim_frames + valid_frames_latent]
                    
                    for b in range(z_lr_valid.shape[0]):
                        latents_lr_trimmed.append(z_lr_valid[b].cpu())

                # --- E. 拼接与对齐 ---
                full_lr = torch.cat(latents_lr_trimmed, dim=-1)
                target_frames = hr_latent_ref.shape[-1]
                
                if full_lr.shape[-1] > target_frames:
                    full_lr = full_lr[..., :target_frames]
                elif full_lr.shape[-1] < target_frames:
                    diff = target_frames - full_lr.shape[-1]
                    full_lr = torch.nn.functional.pad(full_lr, (0, diff))
                
                if full_lr.shape != hr_latent_ref.shape:
                    result_queue.put(('error', pt_path, f"Shape mismatch: Old {hr_latent_ref.shape} vs New {full_lr.shape}"))
                    continue

                # --- F. 覆盖写入 ---
                existing_data['lr_latent'] = full_lr.half()
                torch.save(existing_data, pt_path)

                # --- G. 统计量 ---
                f_sum_hr = hr_latent_ref.double().sum(dim=1)
                f_sq_sum_hr = (hr_latent_ref.double()**2).sum(dim=1)
                f_sum_lr = full_lr.double().sum(dim=1)
                f_sq_sum_lr = (full_lr.double()**2).sum(dim=1)
                f_count = full_lr.shape[1]

                # ✅ 关键修正：返回 pt_path 以便去重
                result_queue.put(('success', pt_path, (f_sum_hr, f_sq_sum_hr, f_sum_lr, f_sq_sum_lr, f_count)))

            except Exception as e:
                result_queue.put(('error', pt_path, str(e)))
                traceback.print_exc()

    except Exception as e:
        print(f"🔥 [GPU {gpu_id}] 崩溃: {e}")

# ==================== 🧠 主控流程 ====================

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    print("=" * 60)
    print("🚀 JaT 数据修正脚本 (LR Only Update) - Final Fix")
    print(f"⚙️  New Low SR: {LOW_SR} Hz | Target: .pt files replacement")
    print("=" * 60)

    # 1. 扫描文件
    print("🔍 扫描现有的 .pt 文件...")
    pt_files = []
    for d in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(d):
            pt_files.extend([str(p) for p in Path(d).rglob('*.pt')])
            
    if not pt_files:
        print("❌ 未找到 .pt 文件")
        return

    # 2. 过滤已完成
    redone_set = load_redone_files(LOG_FILE)
    tasks_to_do = [f for f in pt_files if f not in redone_set]
    print(f"📂 总文件: {len(pt_files)} | ✅ 已修正: {len(redone_set)} | ⚡️ 待修正: {len(tasks_to_do)}")

    if not tasks_to_do:
        print("✅ 所有文件已更新完毕。")
        return

    # 3. 初始化统计量
    if os.path.exists(STATS_FILE):
        stats_data = torch.load(STATS_FILE)
        global_sum_hr = stats_data['sum_hr']
        global_sq_sum_hr = stats_data['sq_sum_hr']
        global_sum_lr = stats_data['sum_lr']
        global_sq_sum_lr = stats_data['sq_sum_lr']
        global_count = stats_data['count']
    else:
        global_sum_hr = torch.zeros(1024, dtype=torch.float64)
        global_sq_sum_hr = torch.zeros(1024, dtype=torch.float64)
        global_sum_lr = torch.zeros(1024, dtype=torch.float64)
        global_sq_sum_lr = torch.zeros(1024, dtype=torch.float64)
        global_count = 0

    # 4. 启动进程
    num_gpus = torch.cuda.device_count()
    manager = mp.Manager()
    input_queue = manager.Queue()
    result_queue = manager.Queue()
    
    for t in tasks_to_do: input_queue.put(t)
    for _ in range(num_gpus): input_queue.put(None)

    workers = []
    for i in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(i, input_queue, result_queue))
        p.start()
        workers.append(p)
    
    pbar = tqdm(total=len(tasks_to_do), desc="Refining", unit="file")
    completed_curr = 0
    
    try:
        while completed_curr < len(tasks_to_do):
            # 检查是否有结果
            if result_queue.empty():
                # 如果所有 worker 都挂了，退出
                if all(not p.is_alive() for p in workers): 
                    print("\n⚠️ 所有 Worker 已退出")
                    break
                time.sleep(0.1)
                continue
            
            # 获取结果
            res = result_queue.get()
            status, pt_path, data = res
            
            if status == 'success':
                # 解包统计数据
                s_sum_hr, s_sq_hr, s_sum_lr, s_sq_lr, s_count = data
                
                # 更新全局统计
                global_sum_hr += s_sum_hr
                global_sq_sum_hr += s_sq_hr
                global_sum_lr += s_sum_lr
                global_sq_sum_lr += s_sq_lr
                global_count += s_count
                
                # 写入日志 (状态: redone)
                save_jsonl({'path': pt_path, 'status': 'redone'}, LOG_FILE)
                
                completed_curr += 1
                pbar.update(1)
                
                # 定期保存统计量 Checkpoint
                if completed_curr % 50 == 0:
                    torch.save({
                        'sum_hr': global_sum_hr, 'sq_sum_hr': global_sq_sum_hr,
                        'sum_lr': global_sum_lr, 'sq_sum_lr': global_sq_sum_lr,
                        'count': global_count
                    }, STATS_FILE)

            elif status == 'error':
                # 记录错误
                save_jsonl({'path': pt_path, 'status': 'error', 'msg': data}, LOG_FILE)
                completed_curr += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        pbar.close()
        print("🧹 清理 Worker 进程...")
        for p in workers:
            if p.is_alive():
                p.terminate()
            p.join()
        
        # 无论如何保存一次当前的统计量
        torch.save({
            'sum_hr': global_sum_hr, 'sq_sum_hr': global_sq_sum_hr,
            'sum_lr': global_sum_lr, 'sq_sum_lr': global_sq_sum_lr,
            'count': global_count
        }, STATS_FILE)

    # 5. 计算最终统计 JSON
    print("\n📊 计算最终全局统计量...")
    if global_count > 0:
        def calc_stats(g_sum, g_sq):
            mean = g_sum / float(global_count)
            var = (g_sq / float(global_count)) - (mean ** 2)
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            return mean.float().tolist(), std.float().tolist()
        
        hr_mean, hr_std = calc_stats(global_sum_hr, global_sq_sum_hr)
        lr_mean, lr_std = calc_stats(global_sum_lr, global_sq_sum_lr)
        
        final_stats = {
            'hr_mean': hr_mean,
            'hr_std': hr_std,
            'lr_mean': lr_mean,
            'lr_std': lr_std,
            'total_frames': int(global_count)
        }
        with open(FINAL_STATS_JSON, 'w') as f:
            json.dump(final_stats, f, indent=4)
        print(f"✅ 统计量已保存: {FINAL_STATS_JSON}")
    else:
        print("⚠️ 未处理任何文件，无法计算统计量。")

if __name__ == '__main__':
    main()