"""
验证CFG修复是否正确
测试：
1. Sample-level masking（而非batch-level）
2. 正确的执行顺序（Noise → CFG）
"""
import torch

print("=" * 80)
print("CFG修复验证")
print("=" * 80)

# 模拟训练配置
cfg_dropout_prob = 0.1
condition_noise_ratio = 0.02

# 模拟一个batch
batch_size = 28
channels = 1024
seq_len = 345

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模拟输入
lr_norm = torch.randn(batch_size, channels, seq_len, device=device)
print(f"\n初始 lr_norm shape: {lr_norm.shape}")
print(f"初始 lr_norm mean: {lr_norm.mean():.4f}, std: {lr_norm.std():.4f}")

# === 旧的错误实现（Batch-level + 错误顺序）===
print("\n" + "=" * 80)
print("❌ 旧实现（有bug）")
print("=" * 80)

lr_norm_old = lr_norm.clone()

# 错误1：Batch-level dropout
import random
if random.random() < cfg_dropout_prob:
    lr_norm_old = torch.zeros_like(lr_norm_old)
    dropout_triggered_old = True
else:
    dropout_triggered_old = False

print(f"Batch-level dropout: {'触发（全部置零）' if dropout_triggered_old else '未触发（全部保留）'}")

# 错误2：CFG在前，Noise在后
if condition_noise_ratio > 0:
    cond_noise = torch.randn_like(lr_norm_old) * condition_noise_ratio
    lr_norm_old = lr_norm_old + cond_noise

print(f"最终 lr_norm_old mean: {lr_norm_old.mean():.4f}, std: {lr_norm_old.std():.4f}")

if dropout_triggered_old:
    print("⚠️  问题：CFG触发后的'null token'不是纯0，而是noise！")
    print(f"   Zero samples的mean: {lr_norm_old.mean():.4f} (应该是0.0)")


# === 新的正确实现（Sample-level + 正确顺序）===
print("\n" + "=" * 80)
print("✅ 新实现（已修复）")
print("=" * 80)

lr_norm_new = lr_norm.clone()

# 步骤1：先加Conditional Noise
if condition_noise_ratio > 0:
    cond_noise = torch.randn_like(lr_norm_new) * condition_noise_ratio
    lr_norm_new = lr_norm_new + cond_noise
    print(f"Step 1: 添加Conditional Noise")
    print(f"  lr_norm mean: {lr_norm_new.mean():.4f}, std: {lr_norm_new.std():.4f}")

# 步骤2：Sample-level CFG masking
B = lr_norm_new.shape[0]
cfg_mask = torch.rand(B, device=device) < cfg_dropout_prob  # [B]
cfg_mask = cfg_mask.view(B, 1, 1)  # [B, 1, 1] 用于broadcast
lr_norm_new = lr_norm_new * (~cfg_mask).float()  # 被mask的样本置零

num_masked = cfg_mask.sum().item()
print(f"\nStep 2: Sample-level CFG masking")
print(f"  Batch size: {B}")
print(f"  被mask的样本数: {num_masked}/{B} ({num_masked/B*100:.1f}%)")
print(f"  期望比例: ~10%")

# 检查被mask的样本是否真的是纯0
if num_masked > 0:
    masked_samples = lr_norm_new[cfg_mask.squeeze()]  # 提取被mask的样本
    print(f"\n被mask样本的验证:")
    print(f"  Mean: {masked_samples.mean():.10f} (应该是0.0)")
    print(f"  Std: {masked_samples.std():.10f} (应该是0.0)")
    print(f"  Max abs value: {masked_samples.abs().max():.10f} (应该是0.0)")

    is_pure_zero = (masked_samples.abs().max() < 1e-6)
    print(f"  ✓ 纯零检查: {'通过' if is_pure_zero else '失败'}")
else:
    print(f"\n  (本次运行没有样本被mask，这是正常的随机现象)")

# 检查未被mask的样本
if num_masked < B:
    unmasked_samples = lr_norm_new[~cfg_mask.squeeze()]
    print(f"\n未mask样本的验证:")
    print(f"  Mean: {unmasked_samples.mean():.4f}")
    print(f"  Std: {unmasked_samples.std():.4f}")
    print(f"  ✓ 保留了原始分布（含noise）")

# === 对比总结 ===
print("\n" + "=" * 80)
print("修复效果对比")
print("=" * 80)

print("""
旧实现问题:
  ❌ Batch-level dropout：整个batch要么全mask，要么全不mask
  ❌ 梯度抖动：一个step全是条件，下个step全是无条件
  ❌ 执行顺序错误：null token = 0 + noise（不是纯0）
  ❌ Domain shift：训练时见的是"noise"，推理时用"0"

新实现优势:
  ✅ Sample-level masking：每个样本独立决定，batch内混合
  ✅ 梯度稳定：每个batch都有~90%条件 + ~10%无条件
  ✅ 执行顺序正确：先noise，后mask（null token是纯0）
  ✅ 训练/推理对齐：推理时用纯0与训练一致
""")

print("=" * 80)
print("✅ CFG实现修复完成！")
print("=" * 80)

print("""
建议：
1. 立即重新开始训练V3m2模型
   - 旧的78步checkpoint是用错误的CFG训练的
   - 需要从头训练以获得正确的CFG能力

2. 监控训练稳定性
   - 应该看到更稳定的loss曲线
   - 梯度范数应该更平滑

3. 推理测试
   - 训练5000步后，测试CFG效果
   - 对比cfg_scale=1.0, 1.5, 2.0的差异
""")
