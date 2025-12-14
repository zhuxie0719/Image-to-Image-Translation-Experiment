# src/data/

## 目录说明

此目录包含**数据处理相关脚本**。

## 文件列表

- `dataset.py` - 数据集类定义（CityscapesDataset等）
- `transforms.py` - 数据增强变换函数
- `split_data.py` - 数据划分脚本

## 主要功能

### dataset.py
- 实现PyTorch Dataset类
- 处理图像加载和预处理
- 支持同步变换（label和photo）

### transforms.py
- 数据增强函数（随机翻转、裁剪、颜色抖动等）
- 图像归一化
- 尺寸调整

### split_data.py
- 数据集划分脚本（可选，因为数据集已经预划分）
- 如果使用，只需确认train/val划分，无需划分测试集
- 生成划分索引文件（仅包含train和val）
- 保存到 `data/splits/cityscapes_split_seed42.json`

