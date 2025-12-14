# outputs/logs/

## 目录说明

此目录用于存放**训练日志文件**。

## 文件格式

- CSV格式：记录每个epoch的损失和指标
- JSON格式：实验配置和超参数
- 文本格式：训练过程日志

## 日志内容

- 训练损失（生成器损失、判别器损失、L1损失等）
- 验证集指标（PSNR、SSIM、MAE、FID）
- 训练时间、学习率等元信息

## 命名规范

建议命名格式：
- `pix2pix_train_log_YYYYMMDD_HHMMSS.csv`
- `cyclegan_train_log_YYYYMMDD_HHMMSS.csv`

## 用途

- 追踪训练过程
- 分析训练曲线
- 复现实验结果

