# src/eval/

## 目录说明

此目录包含**评估指标和可视化工具**。

## 文件列表

- `metrics.py` - 评估指标实现（PSNR、SSIM、MAE、FID）
- `visualize.py` - 三联图生成、结果可视化

## 评估指标

### metrics.py
- **PSNR**：峰值信噪比，衡量像素误差
- **SSIM**：结构相似性，衡量结构一致性与纹理保真度
- **MAE**：平均绝对误差，衡量像素级平均偏差
- **FID**：Fréchet Inception Distance，衡量生成分布与真实分布的距离

## 可视化功能

### visualize.py
- **三联图生成**：Label / Generated / Ground Truth
- **训练曲线绘制**：损失曲线、指标曲线
- **结果对比图**：不同实验的生成结果对比
- **失败案例分析**：可视化失败样例

## 使用方式

每个epoch在验证集上计算所有指标，记录到CSV文件。

