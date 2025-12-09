"""
计算 train_ddp_v2.py 使用的模型参数量
"""

import torch
import sys
sys.path.append('/media/990Evo/Code/JaT')

from src.models.jat_audiosr_v2 import JaT_AudioSR_V2

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num):
    """格式化参数量显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def main():
    print("=" * 80)
    print("模型配置对比分析")
    print("=" * 80)

    # train_ddp_v2.py 配置
    config_v2 = {
        'input_channels': 1024,
        'cond_channels': 1024,
        'patch_len': 4,
        'hidden_size': 1024,      # ← 更小
        'depth': 16,              # ← 更少层
        'num_q_heads': 16,        # ← 更少头
        'num_kv_heads': 4,
        'bottleneck_dim': 512,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'drop_path_rate': 0.05
    }

    # train_ddp_v3mod2.py 配置
    config_v3 = {
        'input_channels': 1024,
        'cond_channels': 1024,
        'patch_len': 4,
        'hidden_size': 1280,      # ← 更大
        'depth': 28,              # ← 更多层
        'num_q_heads': 20,        # ← 更多头
        'num_kv_heads': 4,
        'bottleneck_dim': 512,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'drop_path_rate': 0.05
    }

    print("\n📋 配置对比:")
    print("-" * 80)
    print(f"{'参数':<20} {'train_ddp_v2':<20} {'train_ddp_v3mod2':<20} {'变化':<20}")
    print("-" * 80)

    keys = ['hidden_size', 'depth', 'num_q_heads', 'num_kv_heads']
    for key in keys:
        v2_val = config_v2.get(key, '-')
        v3_val = config_v3.get(key, '-')
        if isinstance(v2_val, (int, float)) and isinstance(v3_val, (int, float)):
            change = f"+{v3_val - v2_val}" if v3_val > v2_val else f"{v3_val - v2_val}"
            if v3_val > v2_val:
                change += f" (+{(v3_val/v2_val - 1)*100:.0f}%)"
        else:
            change = "相同"
        print(f"{key:<20} {str(v2_val):<20} {str(v3_val):<20} {change:<20}")

    # 创建两个模型
    print("\n" + "=" * 80)
    print("🔨 创建模型并计算参数量...")
    print("=" * 80)

    print("\n📦 train_ddp_v2.py 模型:")
    print("-" * 80)
    model_v2 = JaT_AudioSR_V2(**config_v2)
    total_v2, trainable_v2 = count_parameters(model_v2)
    print(f"总参数量: {format_params(total_v2)} ({total_v2:,})")

    # 单Block参数量
    block_v2 = model_v2.blocks[0]
    block_params_v2 = sum(p.numel() for p in block_v2.parameters())
    print(f"单Block参数: {format_params(block_params_v2)} × {config_v2['depth']} = {format_params(block_params_v2 * config_v2['depth'])}")

    print("\n📦 train_ddp_v3mod2.py 模型:")
    print("-" * 80)
    model_v3 = JaT_AudioSR_V2(**config_v3)
    total_v3, trainable_v3 = count_parameters(model_v3)
    print(f"总参数量: {format_params(total_v3)} ({total_v3:,})")

    # 单Block参数量
    block_v3 = model_v3.blocks[0]
    block_params_v3 = sum(p.numel() for p in block_v3.parameters())
    print(f"单Block参数: {format_params(block_params_v3)} × {config_v3['depth']} = {format_params(block_params_v3 * config_v3['depth'])}")

    # 对比
    print("\n" + "=" * 80)
    print("📊 参数量对比")
    print("=" * 80)
    diff = total_v3 - total_v2
    ratio = total_v3 / total_v2
    print(f"\ntrain_ddp_v2:      {format_params(total_v2):<15} ({total_v2:,})")
    print(f"train_ddp_v3mod2:  {format_params(total_v3):<15} ({total_v3:,})")
    print(f"\n差异:              {format_params(diff):<15} ({diff:,})")
    print(f"倍数:              {ratio:.2f}x")
    print(f"增长比例:          +{(ratio - 1)*100:.1f}%")

    # 显存对比
    print("\n" + "=" * 80)
    print("💾 显存占用对比 (FP16 混合精度)")
    print("=" * 80)

    # V2
    fp16_v2_base = (total_v2 * 2 + trainable_v2 * 2 + trainable_v2 * 8) / (1024**3)
    fp16_v2_total = fp16_v2_base * 3.5  # 包含激活值

    # V3
    fp16_v3_base = (total_v3 * 2 + trainable_v3 * 2 + trainable_v3 * 8) / (1024**3)
    fp16_v3_total = fp16_v3_base * 3.5  # 包含激活值

    print(f"\ntrain_ddp_v2 (batch=28):")
    print(f"  基础显存:     {fp16_v2_base:.2f} GB")
    print(f"  估算总显存:   {fp16_v2_total:.2f} GB")

    print(f"\ntrain_ddp_v3mod2 (batch=28):")
    print(f"  基础显存:     {fp16_v3_base:.2f} GB")
    print(f"  估算总显存:   {fp16_v3_total:.2f} GB")

    print(f"\n显存增加:       +{fp16_v3_total - fp16_v2_total:.2f} GB")

    # 建议
    print("\n" + "=" * 80)
    print("💡 使用建议")
    print("=" * 80)

    print(f"""
train_ddp_v2.py (小模型):
  ✅ 参数量: {format_params(total_v2)} (较小)
  ✅ 显存需求: ~{fp16_v2_total:.0f}GB (更低)
  ✅ 训练速度: 更快 (约1.5x)
  ✅ 推理速度: 更快 (约1.5x)
  ⚠️ 表达能力: 中等
  📌 适用场景: 快速实验、资源受限、实时应用

train_ddp_v3mod2.py (大模型):
  ✅ 参数量: {format_params(total_v3)} (更大)
  ✅ 表达能力: 更强
  ✅ 拟合能力: 更好
  ⚠️ 显存需求: ~{fp16_v3_total:.0f}GB (更高)
  ⚠️ 训练速度: 较慢
  📌 适用场景: 追求最佳质量、充足算力

当前硬件建议:
  - 如果显卡 < 30GB: 使用 train_ddp_v2.py
  - 如果显卡 ≥ 40GB: 可以使用 train_ddp_v3mod2.py
  - 2×RTX 4090 (24GB): train_ddp_v2.py 最佳
  - 2×A100 (40GB):     train_ddp_v3mod2.py 推荐
    """)

    # 训练时间估算
    print("=" * 80)
    print("⏱️ 训练时间对比 (估算)")
    print("=" * 80)

    # 假设 V2 是基准
    v2_time_per_step = 1.0
    v3_time_per_step = ratio * 0.7  # 不完全线性，因为还有数据加载等开销

    print(f"\n假设训练 10000 steps:")
    print(f"  train_ddp_v2:      ~{v2_time_per_step * 10000 / 3600:.1f} 小时 (基准)")
    print(f"  train_ddp_v3mod2:  ~{v3_time_per_step * 10000 / 3600:.1f} 小时 (慢 {(v3_time_per_step/v2_time_per_step - 1)*100:.0f}%)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
