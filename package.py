#!/usr/bin/env python3
import os
import re
import sys
import fnmatch

# 移除 from pathlib import Path，后续用 os 替代
# from pathlib import Path

# --- 配置区 ---
# 目标目录，使用 os 获取当前脚本所在目录
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

# 输出的Markdown文件
OUTPUT_FILE = "codebase_prompt.md"

# ==============================================================================
# 新增：仅包含（白名单）模式 - 用于筛选【文件内容】
# ==============================================================================
INCLUDE_ONLY_PATTERNS = [
    "*.py", "*.yaml","*.md"
]

# ==============================================================================
# 排除（黑名单）模式
# ==============================================================================
# 要排除的目录。此规则对【项目结构图】和【文件内容】两部分都生效。
EXCLUDE_DIRS = [
    "*/node_modules/*", "*/.git/*", "*/dist/*", "*/build/*", "*/.vscode/*", "*/.idea/*",
    "*/__pycache__/*", "*/venv/*", "*/.nuxt/*", "*/diaries/*", "*/runs/*", "*/output*/*"
]

# 要排除的文件类型或文件。
EXCLUDE_FILES = [
    "*.log", "*.tmp", "*.lock", "*.map", "package-lock.json", "yarn.lock", "pnpm-lock.yaml","*.mp3",
    "*.DS_Store", "*.sqlite3", "*.db", "*.png", "*.ico", "*.jpg", "*.jpeg", "*.gif", "*.svg","*.flac",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.pth", "*.npy", "tokenizer.json", "alphagenome_pytorch","*.wav"
]

# --- END 配置区 ---

def main():
    # 检查 SOURCE_DIR 是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源目录 '{SOURCE_DIR}' 不存在。")
        sys.exit(1)

    # 清空或创建输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('')

    # --- 1. 生成项目结构图 ---
    generate_project_structure()

    # --- 2. 拼接代码内容 ---
    generate_code_content()

    print(f"Done! 代码已拼接至 '{OUTPUT_FILE}'")

def generate_project_structure():
    """生成项目结构的Markdown文档"""
    # 预处理要排除的目录名
    exclude_names_pattern = []
    for dir_pattern in EXCLUDE_DIRS:
        dir_name_part = re.sub(r'/\*$', '', dir_pattern)
        dir_name_part = os.path.basename(dir_name_part)
        if dir_name_part and dir_name_part != '*' and dir_name_part != '.':
            exclude_names_pattern.append(dir_name_part)
    
    exclude_names_regex = '|'.join(exclude_names_pattern) if exclude_names_pattern else ''

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 项目结构\n\n")
        f.write(f"项目`{SOURCE_DIR}`的目录结构（已排除如`node_modules`等目录）：\n")
        f.write("```\n")
        f.write(f"{SOURCE_DIR}/\n")
        
        # 生成目录树
        generate_tree(SOURCE_DIR, '', exclude_names_regex, f)
        
        f.write("```\n\n")
        f.write("---\n\n")

def generate_tree(dir_path, prefix, exclude_regex, file_obj):
    """递归生成目录树"""
    # 获取当前目录下一级的所有文件和目录，并排序
    items = sorted(os.listdir(dir_path))

    for i, item in enumerate(items):
        # 检查是否需要排除这个文件或目录
        if exclude_regex and re.search(exclude_regex, item):
            continue

        item_path = os.path.join(dir_path, item)

        # 排除 EXCLUDE_FILES 中的文件（音频、图片等）
        if os.path.isfile(item_path):
            if any(fnmatch.fnmatch(item, pattern) for pattern in EXCLUDE_FILES):
                continue

        # 判断连接符
        connector = "├── "
        new_prefix = "│   "
        if i == len(items) - 1:
            connector = "└── "
            new_prefix = "    "

        # 判断是目录还是文件，并输出
        if os.path.isdir(item_path):
            file_obj.write(f"{prefix}{connector}{item}/\n")
            # 递归调用
            generate_tree(item_path, prefix + new_prefix, exclude_regex, file_obj)
        else:
            file_obj.write(f"{prefix}{connector}{item}\n")

def generate_code_content():
    """生成代码内容的Markdown文档"""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("# 代码内容\n\n")
        
        # 构建文件搜索条件
        if INCLUDE_ONLY_PATTERNS:
            print(f"模式：仅拼接匹配 {' '.join(INCLUDE_ONLY_PATTERNS)} 的文件内容")
            include_patterns = INCLUDE_ONLY_PATTERNS
        else:
            print("模式：拼接所有文件，除了黑名单中的文件和目录")
            include_patterns = None
        
        # 遍历所有文件
        for root, dirs, files in os.walk(SOURCE_DIR):
            # 排除目录
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), EXCLUDE_DIRS)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # 排除文件
                if should_exclude(file_path, EXCLUDE_DIRS + [f"*/{pattern}" for pattern in EXCLUDE_FILES]):
                    continue
                
                # 检查是否在包含列表中
                if include_patterns and not any(fnmatch.fnmatch(file_path, pattern) for pattern in include_patterns):
                    continue
                
                # 写入文件内容
                relative_path = os.path.relpath(file_path, SOURCE_DIR)
                f.write(f"## 文件: `{relative_path}`\n\n")
                
                # 获取文件扩展名作为代码块语言
                file_ext = os.path.splitext(file_path)[1].lstrip('.')
                f.write(f"```{'python' if file_ext == 'py' else file_ext}\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"无法读取文件: {file_path} ({str(e)})\n")
                
                f.write("\n```\n\n")

def should_exclude(path, patterns):
    """检查路径是否应被排除"""
    path_str = str(path)
    return any(re.search(pattern.replace('*', '.*'), path_str) for pattern in patterns)

if __name__ == "__main__":
    main()