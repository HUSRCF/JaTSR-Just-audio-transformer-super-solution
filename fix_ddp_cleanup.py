"""
为三个训练脚本添加try...finally块以确保DDP正确清理
"""
import re

def add_try_finally_to_file(filename):
    """为指定文件添加try...finally块"""
    print(f"\n处理 {filename}...")

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份原文件
    with open(f"{filename}.backup", 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  已备份到 {filename}.backup")

    # 找到cleanup_ddp()调用的位置
    cleanup_match = re.search(r'^(\s*)cleanup_ddp\(\)', content, re.MULTILINE)
    if not cleanup_match:
        print(f"  ⚠️  未找到cleanup_ddp()调用")
        return False

    cleanup_indent = cleanup_match.group(1)
    cleanup_pos = cleanup_match.start()

    # 向上查找，找到训练循环开始的位置（查找 "训练循环" 注释）
    training_loop_pattern = r'#\s*={50,}\s*\n\s*#\s*.*训练循环.*\n\s*#\s*={50,}\s*\n'
    training_match = re.search(training_loop_pattern, content[:cleanup_pos])

    if not training_match:
        print(f"  ⚠️  未找到训练循环标记")
        return False

    # 在训练循环注释后插入try:
    try_insert_pos = training_match.end()

    # 查找try:之后的第一个非空行，确定基础缩进
    lines_after = content[try_insert_pos:cleanup_pos].split('\n')
    base_indent = None
    for line in lines_after:
        if line.strip() and not line.strip().startswith('#'):
            base_indent = len(line) - len(line.lstrip())
            break

    if base_indent is None:
        print(f"  ⚠️  无法确定基础缩进")
        return False

    print(f"  基础缩进: {base_indent} 空格")
    print(f"  cleanup缩进: {len(cleanup_indent)} 空格")

    # 检查是否已经有try块
    if content[try_insert_pos:cleanup_pos].strip().startswith('try:'):
        print(f"  ℹ️  已存在try块，将重新处理")
        # 移除现有的try:
        content_before_try = content[:try_insert_pos]
        content_after_try = content[try_insert_pos:]
        # 找到try:那一行并移除
        lines = content_after_try.split('\n')
        if lines[0].strip().startswith('try:') or (len(lines) > 1 and lines[1].strip() == 'try:'):
            # 移除try行
            for i, line in enumerate(lines):
                if line.strip() == 'try:':
                    lines.pop(i)
                    break
            content_after_try = '\n'.join(lines)
        content = content_before_try + content_after_try
        # 重新查找cleanup位置
        cleanup_match = re.search(r'^(\s*)cleanup_ddp\(\)', content, re.MULTILINE)
        cleanup_pos = cleanup_match.start()

    # 构建新内容
    # 1. try:之前的部分
    new_content = content[:try_insert_pos]

    # 2. 添加try:
    new_content += '\n' + ' ' * base_indent + 'try:\n'

    # 3. 训练循环部分（需要增加4个空格缩进）
    training_code = content[try_insert_pos:cleanup_pos]
    training_lines = training_code.split('\n')
    indented_lines = []

    for line in training_lines:
        if line.strip():  # 非空行增加缩进
            indented_lines.append('    ' + line)
        else:  # 空行保持原样
            indented_lines.append(line)

    new_content += '\n'.join(indented_lines)

    # 4. 添加finally:和cleanup_ddp()
    new_content += '\n' + ' ' * base_indent + 'finally:\n'
    new_content += ' ' * (base_indent + 4) + '# 确保无论训练正常结束还是异常退出，都清理DDP进程\n'
    new_content += ' ' * (base_indent + 4) + 'cleanup_ddp()\n'

    # 5. cleanup_ddp()之后的部分
    cleanup_line_end = content.find('\n', cleanup_pos)
    new_content += content[cleanup_line_end + 1:]

    # 写入新文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✅ 成功添加try...finally块")
    return True

# 处理三个文件
files = [
    'train_ddp_v2.py',
    'train_ddp_v3.py',
    'train_ddp_v3m2.py'
]

print("=" * 70)
print("为训练脚本添加DDP清理机制（try...finally）")
print("=" * 70)

success_count = 0
for filename in files:
    try:
        if add_try_finally_to_file(filename):
            success_count += 1
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print(f"完成！成功处理 {success_count}/{len(files)} 个文件")
print("=" * 70)
