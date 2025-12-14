# notebooks/

## 目录说明

此目录包含**Jupyter Notebook实验脚本**。

## 文件列表

- `00_data_exploration.ipynb` - 数据探索（EDA）
- `01_unet_baseline.ipynb` - U-Net基线实验
- `02_pix2pix_training.ipynb` - Pix2Pix训练实验
- `03_cyclegan_training.ipynb` - CycleGAN训练实验

## Notebook说明

### 00_data_exploration.ipynb
- 数据集统计（图像数量、尺寸分布）
- 语义标签类别分布
- 随机样本可视化
- 数据质量检查

### 01_unet_baseline.ipynb
- U-Net基线模型训练
- 仅L1损失
- 结果可视化

### 02_pix2pix_training.ipynb
- Pix2Pix模型训练
- 损失函数消融实验
- 训练过程可视化

### 03_cyclegan_training.ipynb
- CycleGAN模型训练
- 循环一致性损失
- 与Pix2Pix对比

## 使用建议

- 建议在Google Colab或本地Jupyter环境中运行
- 定期保存notebook结果
- 关键实验结果导出为图片保存到 `outputs/figures/`

