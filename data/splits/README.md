# data/splits/

## 目录说明

此目录用于存放**训练/验证划分索引**（图像生成任务无需测试集）。

## 文件格式

- JSON格式：`cityscapes_split_seed42.json`
- 包含训练集、验证集的图像ID列表（仅train和val，无test）

## 划分方案

**对于图像生成任务，推荐直接使用现有的train/val划分，无需划分测试集。**

- **训练集 (train)**: 2975 张
- **验证集 (val)**: 500 张（用于验证、模型选择、最终评估）

验证集已足够用于：
- 训练过程中的模型选择
- 超参数调优
- 生成样例展示
- 最终评估（计算PSNR、SSIM、MAE、FID等指标）

## 索引文件格式

```json
{
    "train": ["1.jpg", "2.jpg", "3.jpg", ...],
    "val": ["1.jpg", "2.jpg", "3.jpg", ...]
}
```

**说明**：
- `train` 列表：包含 `data/raw/cityscapes/train/` 目录中的所有2975个图像文件名
- `val` 列表：包含 `data/raw/cityscapes/val/` 目录中的所有500个图像文件名
- 无需 `test` 字段

## 用途

- 保证双方实验一致
- 便于复现实验结果
- 明确指定哪些图像用于训练，哪些用于验证

## 实现方式

可以创建一个简单的Python脚本，扫描train和val目录，生成索引文件：

```python
import os
import json

train_dir = "../../data/raw/cityscapes/train"
val_dir = "../../data/raw/cityscapes/val"

train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
val_files = sorted([f for f in os.listdir(val_dir) if f.endswith('.jpg')])

split_info = {
    "train": train_files,
    "val": val_files
}

with open("cityscapes_split_seed42.json", "w") as f:
    json.dump(split_info, f, indent=2)
```

## 参考文件

- 示例格式：`cityscapes_split_example.json`
- 详细说明：`DATA_SPLIT_INFO.md`
