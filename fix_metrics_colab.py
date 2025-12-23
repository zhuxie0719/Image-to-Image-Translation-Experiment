"""
快速修复脚本：在 Colab 中修复 metrics.py，确保 evaluate_batch 返回 'l1' 键
在 Colab 中运行：!python fix_metrics_colab.py
"""

from pathlib import Path

metrics_file = Path("src/eval/metrics.py")

if metrics_file.exists():
    with open(metrics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经包含 'l1' 键
    if "'l1':" not in content and '"l1":' not in content:
        print("⚠️  检测到 metrics.py 缺少 'l1' 键，正在修复...")
        
        # 查找 evaluate_batch 函数的返回语句并修复
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            
            # 找到 return { 语句
            if 'return {' in line and i > 0 and 'def evaluate_batch' in '\n'.join(lines[max(0, i-20):i]):
                # 检查接下来的几行
                if i + 1 < len(lines) and "'psnr':" in lines[i+1]:
                    # 在 'psnr' 之前插入 'l1'
                    new_lines.append("        'l1': mae,")
                    print("✅ 已添加 'l1' 键")
                elif i + 1 < len(lines) and "'mae': mae" in lines[i+1]:
                    # 在 'mae' 之前插入 'l1'
                    new_lines.pop()  # 移除刚才添加的 return {
                    new_lines.append("    # L1 损失和 MAE 是一样的（都是平均绝对误差）")
                    new_lines.append("    l1 = mae")
                    new_lines.append("    ")
                    new_lines.append("    return {")
                    new_lines.append("        'l1': l1,")
                    print("✅ 已添加 'l1' 键")
            
            i += 1
        
        # 如果没找到，尝试直接替换
        if "'l1':" not in '\n'.join(new_lines):
            content = content.replace(
                "    return {\n        'psnr': psnr,\n        'ssim': ssim,\n        'mae': mae\n    }",
                "    # L1 损失和 MAE 是一样的（都是平均绝对误差）\n    l1 = mae\n    \n    return {\n        'l1': l1,\n        'psnr': psnr,\n        'ssim': ssim,\n        'mae': mae\n    }"
            )
            new_lines = content.split('\n')
        
        # 写回文件
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("✅ metrics.py 已修复")
    else:
        print("✅ metrics.py 已包含 'l1' 键，无需修复")
else:
    print(f"⚠️  文件不存在：{metrics_file}")

