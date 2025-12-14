"""
检查Cityscapes数据集格式的脚本
用于确认图像是否为左右拼接格式（左侧photo，右侧label）
"""

import os
from PIL import Image
import numpy as np

def check_image_format(image_path):
    """检查单张图像的格式"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        channels = len(img.getbands())
        
        print(f"图像路径: {image_path}")
        print(f"尺寸: {width} x {height}")
        print(f"通道数: {channels}")
        
        # 检查是否为左右拼接（宽度应该是高度的2倍左右）
        if width > height * 1.5:
            print("[OK] 疑似左右拼接格式（宽度明显大于高度）")
            # 分割左右两部分查看
            left_half = img.crop((0, 0, width // 2, height))
            right_half = img.crop((width // 2, 0, width, height))
            
            # 转换为numpy数组检查
            left_arr = np.array(left_half)
            right_arr = np.array(right_half)
            
            print(f"  左侧部分（photo）: {left_arr.shape}, 像素值范围: [{left_arr.min()}, {left_arr.max()}]")
            print(f"  右侧部分（label）: {right_arr.shape}, 像素值范围: [{right_arr.min()}, {right_arr.max()}]")
            
            # 检查右侧是否为标签图（通常颜色较少或为特定颜色映射）
            if channels == 3:
                right_unique_colors = len(np.unique(right_arr.reshape(-1, 3), axis=0))
                print(f"  右侧唯一颜色数: {right_unique_colors}")
                if right_unique_colors < 100:
                    print("  [OK] 右侧疑似语义标签图（颜色种类较少）")
                else:
                    print("  [WARNING] 右侧颜色种类较多，可能不是标准标签图")
            
            return True, left_half, right_half
        else:
            print("[WARNING] 不是明显的左右拼接格式")
            return False, None, None
            
    except Exception as e:
        print(f"错误: {e}")
        return False, None, None

if __name__ == "__main__":
    # 检查train和val目录中的样例图像
    base_dir = "../../data/raw/cityscapes"
    
    print("=" * 60)
    print("检查训练集样例图像")
    print("=" * 60)
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
        if train_files:
            sample_file = os.path.join(train_dir, train_files[0])
            check_image_format(sample_file)
    
    print("\n" + "=" * 60)
    print("检查验证集样例图像")
    print("=" * 60)
    val_dir = os.path.join(base_dir, "val")
    if os.path.exists(val_dir):
        val_files = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
        if val_files:
            sample_file = os.path.join(val_dir, val_files[0])
            check_image_format(sample_file)
    
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    if os.path.exists(train_dir):
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        print(f"训练集图像数量: {train_count}")
    
    if os.path.exists(val_dir):
        val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        print(f"验证集图像数量: {val_count}")
    
    # 检查是否有test目录
    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir):
        test_count = len([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
        print(f"测试集图像数量: {test_count}")
    else:
        print("测试集: 未找到test目录")
        print("\n建议:")
        print("1. 从验证集中分出一部分作为测试集（推荐）")
        print("2. 从训练集中分出一部分作为测试集")
        print("3. 使用验证集作为测试集（但这样就没有验证集了）")

