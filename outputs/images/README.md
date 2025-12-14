# outputs/images/

## 目录说明

此目录用于存放**每个epoch的验证集三联图**。

## 文件格式

- 图像格式：PNG或JPG
- 三联图：Label / Generated / Ground Truth 水平拼接

## 命名规范

建议命名格式：
- `pix2pix_epoch_{epoch}_sample_{sample_id}.png`
- `cyclegan_epoch_{epoch}_sample_{sample_id}.png`

## 保存策略

- 每个epoch保存验证集样例（至少10-20张）
- 用于报告展示和结果分析
- 展示训练过程的质量演变

## 用途

- 训练过程可视化
- 结果对比分析
- 报告图表素材

