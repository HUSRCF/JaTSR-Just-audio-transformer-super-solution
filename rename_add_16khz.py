#!/usr/bin/env python3
"""
重命名脚本：在文件名尾部（扩展名前）添加 _16kHz 后缀
用途：为多采样率数据集生成准备文件
"""

import os
import argparse
from pathlib import Path


def rename_files_add_suffix(directory, suffix="_16kHz", dry_run=True, extensions=None):
    """
    在文件扩展名前添加后缀

    Args:
        directory: 目标目录路径
        suffix: 要添加的后缀（默认 "_16kHz"）
        dry_run: 是否为试运行模式（仅显示不执行）
        extensions: 要处理的文件扩展名列表（如 ['.wav', '.flac']），None则处理所有文件

    Returns:
        成功重命名的文件数量
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"目录不存在: {directory}")

    # 获取所有文件
    all_files = [f for f in directory.iterdir() if f.is_file()]

    # 如果指定了扩展名，进行过滤
    if extensions is not None:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        files_to_process = [f for f in all_files if f.suffix.lower() in extensions]
    else:
        files_to_process = all_files

    print(f"\n📁 目录: {directory}")
    print(f"📊 总文件数: {len(all_files)}")
    print(f"✅ 待处理文件数: {len(files_to_process)}")
    print(f"🏷️  后缀: {suffix}")
    print(f"{'🔍 [DRY RUN 模式]' if dry_run else '✏️  [执行模式]'}\n")

    if len(files_to_process) == 0:
        print("⚠️  没有找到符合条件的文件")
        return 0

    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for file_path in sorted(files_to_process):
        # 构建新文件名
        stem = file_path.stem  # 文件名（不含扩展名）
        ext = file_path.suffix  # 扩展名（含点号）

        # 检查是否已经包含后缀
        if stem.endswith(suffix):
            print(f"⏭️  跳过（已有后缀）: {file_path.name}")
            skipped_count += 1
            continue

        new_stem = f"{stem}{suffix}"
        new_filename = f"{new_stem}{ext}"
        new_path = file_path.parent / new_filename

        # 检查新文件名是否已存在
        if new_path.exists():
            print(f"❌ 跳过（目标已存在）: {file_path.name} -> {new_filename}")
            error_count += 1
            continue

        # 显示重命名操作
        print(f"{'[DRY]' if dry_run else '✓'} {file_path.name}")
        print(f"   -> {new_filename}")

        # 执行重命名
        if not dry_run:
            try:
                file_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"   ❌ 错误: {e}")
                error_count += 1
        else:
            renamed_count += 1

    print(f"\n{'='*60}")
    print(f"📊 统计:")
    print(f"   ✅ {'预计重命名' if dry_run else '成功重命名'}: {renamed_count}")
    print(f"   ⏭️  跳过（已有后缀）: {skipped_count}")
    print(f"   ❌ 错误/冲突: {error_count}")
    print(f"{'='*60}\n")

    return renamed_count


def main():
    parser = argparse.ArgumentParser(
        description="在文件名尾部（扩展名前）添加后缀，用于多采样率数据集准备",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 试运行模式（默认）- 仅预览不执行
  python rename_add_16khz.py extraA/

  # 执行重命名
  python rename_add_16khz.py extraA/ --execute

  # 自定义后缀
  python rename_add_16khz.py extraA/ --suffix _48kHz --execute

  # 仅处理特定扩展名
  python rename_add_16khz.py extraA/ --ext .wav --ext .flac --execute
        """
    )

    parser.add_argument('directory', type=str,
                        help='目标目录路径')

    parser.add_argument('--suffix', type=str, default='_16kHz',
                        help='要添加的后缀（默认: _16kHz）')

    parser.add_argument('--execute', action='store_true',
                        help='执行重命名（不加此参数则为试运行模式）')

    parser.add_argument('--ext', action='append', dest='extensions',
                        help='仅处理指定扩展名（可多次使用，如: --ext .wav --ext .flac）')

    args = parser.parse_args()

    try:
        dry_run = not args.execute

        if dry_run:
            print("\n" + "="*60)
            print("🔍 当前为 DRY RUN 模式（试运行）")
            print("   将显示重命名预览，但不会实际修改文件")
            print("   确认无误后，请添加 --execute 参数执行实际重命名")
            print("="*60)

        renamed_count = rename_files_add_suffix(
            directory=args.directory,
            suffix=args.suffix,
            dry_run=dry_run,
            extensions=args.extensions
        )

        if dry_run and renamed_count > 0:
            print("💡 提示: 确认无误后，请添加 --execute 参数执行实际重命名")
            print(f"   命令: python {os.path.basename(__file__)} {args.directory} --execute")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
