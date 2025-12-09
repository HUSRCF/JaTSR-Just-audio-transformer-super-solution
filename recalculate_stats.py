"""
重新计算分离的 HR 和 LR 统计量
从已有的 .pt 文件读取，不重新编码音频
使用多线程加速 I/O
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置
DATA_DIR = 'data_processed_v13_final'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
OUTPUT_JSON = os.path.join(DATA_DIR, 'global_stats_separated.json')
NUM_WORKERS = 64  # 多线程数量

def process_single_file(filepath):
    """处理单个文件，返回统计量"""
    try:
        # 加载数据
        data = torch.load(filepath, map_location='cpu')

        hr = data['hr_latent'].float()  # [1024, T]
        lr = data['lr_latent'].float()  # [1024, T]

        # 计算统计
        hr_sum = hr.double().sum(dim=1)
        hr_sq_sum = (hr.double() ** 2).sum(dim=1)
        hr_count = hr.shape[1]

        lr_sum = lr.double().sum(dim=1)
        lr_sq_sum = (lr.double() ** 2).sum(dim=1)
        lr_count = lr.shape[1]

        return ('success', hr_sum, hr_sq_sum, hr_count, lr_sum, lr_sq_sum, lr_count)

    except Exception as e:
        return ('error', str(filepath), str(e))

def main():
    print("=" * 60)
    print("🔄 重新计算分离的 HR/LR 统计量 (多线程)")
    print("=" * 60)

    # 收集所有 .pt 文件
    all_files = []
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(split_dir):
            files = list(Path(split_dir).glob("*.pt"))
            all_files.extend(files)
            print(f"📂 {split_dir}: {len(files)} files")

    print(f"📊 总文件数: {len(all_files)}")
    print(f"🔧 线程数: {NUM_WORKERS}")

    # 初始化累加器（分离的 HR 和 LR）
    hr_sum = torch.zeros(1024, dtype=torch.float64)
    hr_sq_sum = torch.zeros(1024, dtype=torch.float64)
    hr_count = 0

    lr_sum = torch.zeros(1024, dtype=torch.float64)
    lr_sq_sum = torch.zeros(1024, dtype=torch.float64)
    lr_count = 0

    # 线程锁（用于累加）
    lock = threading.Lock()

    print("\n🔢 开始计算...")

    # 多线程处理
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, f): f for f in all_files}

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Processing"):
            result = future.result()

            if result[0] == 'success':
                _, hr_s, hr_sq_s, hr_c, lr_s, lr_sq_s, lr_c = result

                # 线程安全累加
                with lock:
                    hr_sum += hr_s
                    hr_sq_sum += hr_sq_s
                    hr_count += hr_c

                    lr_sum += lr_s
                    lr_sq_sum += lr_sq_s
                    lr_count += lr_c
            else:
                _, filepath, error = result
                print(f"\n⚠️ 跳过文件 {Path(filepath).name}: {error}")

    print("\n📐 计算均值和标准差...")

    # HR 统计量
    hr_mean = hr_sum / float(hr_count)
    hr_var = (hr_sq_sum / float(hr_count)) - (hr_mean ** 2)
    hr_std = torch.sqrt(torch.clamp(hr_var, min=1e-6))

    # LR 统计量
    lr_mean = lr_sum / float(lr_count)
    lr_var = (lr_sq_sum / float(lr_count)) - (lr_mean ** 2)
    lr_std = torch.sqrt(torch.clamp(lr_var, min=1e-6))

    # 保存
    stats = {
        'hr_mean': hr_mean.float().tolist(),
        'hr_std': hr_std.float().tolist(),
        'lr_mean': lr_mean.float().tolist(),
        'lr_std': lr_std.float().tolist(),
        'hr_total_frames': int(hr_count),
        'lr_total_frames': int(lr_count),
        'note': 'HR and LR statistics are now SEPARATED'
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\n✅ 保存到: {OUTPUT_JSON}")

    # 打印对比
    print("\n" + "=" * 60)
    print("📊 统计量对比")
    print("=" * 60)
    print(f"HR: 总帧数={hr_count}, mean[0:3]={hr_mean[:3].tolist()}, std[0:3]={hr_std[:3].tolist()}")
    print(f"LR: 总帧数={lr_count}, mean[0:3]={lr_mean[:3].tolist()}, std[0:3]={lr_std[:3].tolist()}")
    print(f"\n是否相同? mean={torch.allclose(hr_mean, lr_mean)}, std={torch.allclose(hr_std, lr_std)}")

    # 加载旧的统计量对比
    old_stats_path = os.path.join(DATA_DIR, 'global_stats.json')
    if os.path.exists(old_stats_path):
        with open(old_stats_path, 'r') as f:
            old_stats = json.load(f)
        old_mean = torch.tensor(old_stats['hr_mean'])
        old_std = torch.tensor(old_stats['hr_std'])
        print("\n旧统计量 (混合):")
        print(f"  mean[0:3]={old_mean[:3].tolist()}, std[0:3]={old_std[:3].tolist()}")

if __name__ == '__main__':
    main()
