"""
计算 JaT-AudioSR V2 模型的参数量
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
    print("JaT-AudioSR V2 模型参数量分析")
    print("=" * 80)

    # 模型配置 (从 train_ddp_v3mod2.py)
    config = {
        'input_channels': 1024,      # DAC latent channels
        'cond_channels': 1024,       # LR condition channels
        'patch_len': 4,
        'hidden_size': 1280,
        'depth': 28,                 # 28 DiT blocks
        'num_q_heads': 20,
        'num_kv_heads': 4,           # GQA: 4 KV heads shared by 20 Q heads
        'bottleneck_dim': 512,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'drop_path_rate': 0.05
    }

    print("\n📋 模型配置:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:<20}: {value}")

    # 创建模型
    print("\n🔨 创建模型...")
    model = JaT_AudioSR_V2(**config)

    # 计算参数量
    total_params, trainable_params = count_parameters(model)

    print("\n" + "=" * 80)
    print("📊 参数量统计")
    print("=" * 80)
    print(f"\n总参数量:       {format_params(total_params):<10} ({total_params:,})")
    print(f"可训练参数:     {format_params(trainable_params):<10} ({trainable_params:,})")
    print(f"固定参数:       {format_params(total_params - trainable_params):<10} ({total_params - trainable_params:,})")

    # 分模块统计
    print("\n" + "=" * 80)
    print("🔍 分模块参数量")
    print("=" * 80)

    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
        print(f"  {name:<20}: {format_params(params):<10} ({params:,})")

    # DiT Block 详细分析
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        print("\n" + "=" * 80)
        print("🧩 单个 DiT Block 分析")
        print("=" * 80)

        block = model.blocks[0]
        block_total = sum(p.numel() for p in block.parameters())
        print(f"\n单个 Block 参数量: {format_params(block_total)} ({block_total:,})")
        print(f"28个 Block 总参数量: {format_params(block_total * 28)} ({block_total * 28:,})")

        # Block 子模块
        for name, module in block.named_children():
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name:<20}: {format_params(params):<10} ({params:,})")

    # 与其他模型对比
    print("\n" + "=" * 80)
    print("📈 与其他 SOTA 模型对比")
    print("=" * 80)

    comparisons = [
        ("JaT-AudioSR V2 (Ours)", total_params),
        ("DiT-XL/2 (Image)", 675e6),  # 675M
        ("Stable Diffusion 1.5", 860e6),  # 860M
        ("AudioLDM 2", 712e6),  # 712M
        ("Vocos", 13e6),  # 13M (vocoder)
        ("DAC", 73e6),  # 73M (codec)
    ]

    print()
    for name, params in comparisons:
        bar_length = int(params / 1e7)
        bar = "█" * min(bar_length, 80)
        print(f"  {name:<30}: {format_params(params):<10} {bar}")

    # 显存估算
    print("\n" + "=" * 80)
    print("💾 显存估算 (单卡)")
    print("=" * 80)

    # FP32
    fp32_model_memory = total_params * 4 / (1024**3)  # 4 bytes per param
    fp32_grad_memory = trainable_params * 4 / (1024**3)
    fp32_optimizer_memory = trainable_params * 8 / (1024**3)  # AdamW: 2x params
    fp32_total = fp32_model_memory + fp32_grad_memory + fp32_optimizer_memory

    # FP16 (Mixed Precision)
    fp16_model_memory = total_params * 2 / (1024**3)
    fp16_grad_memory = trainable_params * 2 / (1024**3)
    fp16_optimizer_memory = trainable_params * 8 / (1024**3)  # AdamW状态仍是FP32
    fp16_total = fp16_model_memory + fp16_grad_memory + fp16_optimizer_memory

    print(f"\n🔸 FP32 训练:")
    print(f"  模型权重:         {fp32_model_memory:.2f} GB")
    print(f"  梯度:             {fp32_grad_memory:.2f} GB")
    print(f"  优化器状态:       {fp32_optimizer_memory:.2f} GB")
    print(f"  总计 (不含激活值): {fp32_total:.2f} GB")

    print(f"\n🔸 FP16 混合精度训练 (当前使用):")
    print(f"  模型权重:         {fp16_model_memory:.2f} GB")
    print(f"  梯度:             {fp16_grad_memory:.2f} GB")
    print(f"  优化器状态:       {fp16_optimizer_memory:.2f} GB")
    print(f"  总计 (不含激活值): {fp16_total:.2f} GB")

    print(f"\n💡 实际训练显存 (batch_size=28, T=1378):")
    # 粗略估算：激活值 ≈ 模型权重的 2-3倍
    activation_memory = fp16_total * 2.5
    total_memory = fp16_total + activation_memory
    print(f"  估算总显存:       {total_memory:.2f} GB")
    print(f"  建议显卡:         RTX 4090 (24GB) 或 A100 (40GB/80GB)")

    # GQA 效率分析
    print("\n" + "=" * 80)
    print("⚡ GQA (Grouped-Query Attention) 效率分析")
    print("=" * 80)

    # 如果用 MHA (Multi-Head Attention)
    mha_kv_heads = config['num_q_heads']  # 20
    gqa_kv_heads = config['num_kv_heads']  # 4

    # KV projection 参数量
    hidden_size = config['hidden_size']
    head_dim = hidden_size // config['num_q_heads']  # 1280 / 20 = 64

    kv_params_per_block_mha = 2 * hidden_size * hidden_size  # K + V projection
    kv_params_per_block_gqa = 2 * hidden_size * (gqa_kv_heads * head_dim)

    kv_saved = (kv_params_per_block_mha - kv_params_per_block_gqa) * config['depth']

    print(f"\n如果使用标准 MHA (20 KV heads):")
    print(f"  KV projection 参数: {format_params(kv_params_per_block_mha * config['depth'])}")

    print(f"\n当前 GQA (4 KV heads):")
    print(f"  KV projection 参数: {format_params(kv_params_per_block_gqa * config['depth'])}")

    print(f"\n节省参数量: {format_params(kv_saved)} ({kv_saved:,})")
    print(f"节省比例: {(1 - kv_params_per_block_gqa/kv_params_per_block_mha)*100:.1f}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
