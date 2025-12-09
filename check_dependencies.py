"""
依赖检查脚本 - 验证所有必需包是否正确安装
"""

import sys

def check_dependencies():
    """检查所有必需的依赖包"""
    # ANSI color codes for gradient: Light Blue -> Purple
    colors = [
        '\033[38;5;51m',   # Cyan (light blue)
        '\033[38;5;87m',   # Light cyan
        '\033[38;5;123m',  # Sky blue
        '\033[38;5;117m',  # Light blue
        '\033[38;5;111m',  # Medium blue
        '\033[38;5;105m',  # Light purple
        '\033[38;5;99m',   # Medium purple
        '\033[38;5;93m',   # Purple
        '\033[38;5;201m',  # Magenta (purple)
    ]
    reset = '\033[0m'

    # ASCII art with gradient colors
    print(f"""
{colors[0]}╔═════════════════════════════════════════════╗{reset}
{colors[1]}║                                             ║{reset}
{colors[2]}║       ██╗ █████╗ ████████╗███████╗██████╗   ║{reset}
{colors[3]}║       ██║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗  ║{reset}
{colors[4]}║       ██║███████║   ██║   ███████╗██████╔╝  ║{reset}
{colors[5]}║  ██   ██║██╔══██║   ██║   ╚════██║██╔══██╗  ║{reset}
{colors[6]}║  ╚█████╔╝██║  ██║   ██║   ███████║██║  ██║  ║{reset}
{colors[7]}║   ╚════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝  ║{reset}
{colors[7]}║                                             ║{reset}
{colors[7]}║    Just Audio Transformer Super Solution    ║{reset}
{colors[8]}║                                             ║{reset}
{colors[8]}╚═════════════════════════════════════════════╝{reset}
    """)
    print("=" * 60)
    print("JaT-AudioSR 依赖检查")
    print("=" * 60)

    missing_packages = []
    all_good = True

    # 核心依赖
    dependencies = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'dac': 'Descript Audio Codec',
        'audiotools': 'AudioTools (DAC 依赖)',
        'librosa': 'Librosa (音频处理)',
        'soundfile': 'SoundFile (音频 I/O)',
        'tensorboard': 'TensorBoard (训练监控)',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm (进度条)',
        'einops': 'einops (张量操作)',
        'matplotlib': 'Matplotlib (可视化)',
        'argbind': 'argbind (配置管理)',
    }

    print("\n检查依赖包安装状态:\n")

    for package, description in dependencies.items():
        try:
            if package == 'dac':
                # DAC 包的导入名称不同
                __import__('dac')
            else:
                __import__(package)

            # 获取版本号
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {description:35s} (v{version})")
            except:
                print(f"✅ {description:35s} (已安装)")

        except ImportError:
            print(f"❌ {description:30s} - 未安装")
            missing_packages.append(package)
            all_good = False

    # CUDA 检查
    print("\n" + "=" * 60)
    print("CUDA 状态检查:")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   - CUDA 版本: {torch.version.cuda}")
            print(f"   - GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"     显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA 不可用 (将使用 CPU 训练，速度会很慢)")
    except Exception as e:
        print(f"❌ CUDA 检查失败: {e}")
        all_good = False

    # 总结
    print("\n" + "=" * 60)
    if all_good:
        print("✅ 所有依赖已正确安装！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 准备数据: python prepare_dataset_v5.py")
        print("  2. 开始训练: python train_ddp_v3mod2.py")
        return 0
    else:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print("  pip install -r requirements.txt")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(check_dependencies())
