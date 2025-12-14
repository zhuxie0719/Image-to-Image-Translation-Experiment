"""
数据集划分脚本
用于生成train/val划分索引文件（图像生成任务无需测试集）

由于数据集已经预划分了train和val，此脚本主要用于：
1. 确认现有的划分
2. 生成划分索引文件，便于实验复现
"""

import os
import json
from pathlib import Path

def create_split_index(raw_data_dir, output_file, seed=42):
    """
    创建数据集划分索引文件
    
    Args:
        raw_data_dir: 原始数据目录路径（包含train和val子目录）
        output_file: 输出JSON文件路径
        seed: 随机种子（虽然不进行随机划分，但保留参数以保持一致性）
    """
    train_dir = os.path.join(raw_data_dir, "train")
    val_dir = os.path.join(raw_data_dir, "val")
    
    # 检查目录是否存在
    if not os.path.exists(train_dir):
        raise ValueError(f"训练集目录不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"验证集目录不存在: {val_dir}")
    
    # 获取所有图像文件名
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    val_files = sorted([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"训练集图像数量: {len(train_files)}")
    print(f"验证集图像数量: {len(val_files)}")
    
    # 创建划分索引
    split_info = {
        "train": train_files,
        "val": val_files,
        "note": "图像生成任务无需测试集。train和val已由数据集提供。"
    }
    
    # 保存到JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"划分索引文件已保存到: {output_file}")
    return split_info

if __name__ == "__main__":
    # 配置路径
    raw_data_dir = "../../data/raw/cityscapes"
    output_file = "../../data/splits/cityscapes_split_seed42.json"
    
    # 创建划分索引
    split_info = create_split_index(raw_data_dir, output_file, seed=42)
    
    print("\n划分完成！")
    print("注意：图像生成任务无需测试集，直接使用train/val划分即可。")

