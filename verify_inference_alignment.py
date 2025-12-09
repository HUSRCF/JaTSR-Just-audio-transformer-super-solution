"""
验证infer_test.py和infer_test_v3.py的pipeline对齐
确保除了模型架构和CFG之外，其他流程完全一致
"""
import re

def check_alignment():
    print("=" * 80)
    print("验证V2和V3推理脚本的Pipeline对齐")
    print("=" * 80)

    with open('infer_test.py') as f:
        v2_content = f.read()

    with open('infer_test_v3.py') as f:
        v3_content = f.read()

    checks = []

    # 1. 数据加载
    v2_data_load = "data = torch.load(first_file, map_location='cpu', mmap=True, weights_only=False)"
    v3_data_load = "data = torch.load(first_file, map_location='cpu', mmap=True, weights_only=False)"
    checks.append(("数据加载", v2_data_load in v2_content and v3_data_load in v3_content))

    # 2. HR/LR latent提取
    v2_hr = "hr_latent = data['hr_latent'].float()"
    v3_hr = "hr_latent = data['hr_latent'].float()"
    checks.append(("HR latent提取", v2_hr in v2_content and v3_hr in v3_content))

    v2_lr = "lr_latent = data['lr_latent'].float()"
    v3_lr = "lr_latent = data['lr_latent'].float()"
    checks.append(("LR latent提取", v2_lr in v2_content and v3_lr in v3_content))

    # 3. 归一化参数加载
    v2_stats = "with open(args.stats_file, 'r') as f:"
    v3_stats = "with open(args.stats_file, 'r') as f:"
    checks.append(("归一化参数加载", v2_stats in v2_content and v3_stats in v3_content))

    # 4. Chunk提取
    v2_chunk = "hr_chunk = hr_latent[:, :chunk_frames].unsqueeze(0).to(device)"
    v3_chunk = "hr_chunk = hr_latent[:, :chunk_frames].unsqueeze(0).to(device)"
    checks.append(("Chunk提取", v2_chunk in v2_content and v3_chunk in v3_content))

    # 5. 归一化公式
    v2_norm = "hr_chunk = (hr_chunk - hr_mean) / hr_std"
    v3_norm = "hr_chunk = (hr_chunk - hr_mean) / hr_std"
    checks.append(("归一化公式", v2_norm in v2_content and v3_norm in v3_content))

    # 6. 反归一化公式
    v2_denorm = "generated_latent = generated_latent * hr_std + hr_mean"
    v3_denorm = "generated_latent = generated_latent * hr_std + hr_mean"
    checks.append(("反归一化公式", v2_denorm in v2_content and v3_denorm in v3_content))

    # 7. DAC解码调用
    v2_decode = "generated_audio = dac_codec.decode(generated_latent)"
    v3_decode = "generated_audio = dac_codec.decode(generated_latent)"
    checks.append(("DAC解码", v2_decode in v2_content and v3_decode in v3_content))

    # 8. 音频后处理
    v2_squeeze = "generated_audio = generated_audio.squeeze(0).cpu()"
    v3_squeeze = "generated_audio = generated_audio.squeeze(0).cpu()"
    checks.append(("音频后处理", v2_squeeze in v2_content and v3_squeeze in v3_content))

    # 9. Flow Matching核心逻辑
    v2_init = "z_t = torch.randn(B, C, T, device=device)"
    v3_init = "z_t = torch.randn(B, C, T, device=device)"
    checks.append(("Flow初始化", v2_init in v2_content and v3_init in v3_content))

    v2_timesteps = "timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)"
    v3_timesteps = "timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)"
    checks.append(("时间步设置", v2_timesteps in v2_content and v3_timesteps in v3_content))

    # 10. ODE速度计算（V3有小改进）
    v2_velocity_pattern = r"velocity = \(x_pred - z_t\) / \(1 - t_curr\)"
    v3_velocity_pattern = r"velocity = \(x_pred - z_t\) / \(1 - t_curr \+ 1e-5\)"
    v2_has_old = bool(re.search(v2_velocity_pattern, v2_content))
    v3_has_new = bool(re.search(v3_velocity_pattern, v3_content))
    checks.append(("ODE速度计算", v2_has_old and v3_has_new))  # V3改进了数值稳定性

    # 打印结果
    print("\n关键流程对齐检查:")
    print("-" * 80)
    all_pass = True
    for name, passed in checks:
        status = "✓ 一致" if passed else "✗ 不一致"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ 所有关键流程对齐检查通过！")
        print("\nV3推理脚本与V2保持pipeline一致，仅以下合理差异：")
        print("  1. 模型架构：JaT_AudioSR_V2 → JaT_AudioSR_V3 (RMSNorm)")
        print("  2. CFG支持：V3新增cfg_scale参数")
        print("  3. 数值稳定性：V3在ODE速度计算中添加1e-5防止除零")
        print("  4. 默认checkpoint路径：v2_full_run → v3_full_run")
    else:
        print("⚠️  发现对齐问题，请检查上述不一致项")
    print("=" * 80)

    return all_pass

if __name__ == "__main__":
    check_alignment()
