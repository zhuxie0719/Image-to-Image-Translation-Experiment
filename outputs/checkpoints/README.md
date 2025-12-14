# outputs/checkpoints/

## 目录说明

此目录用于存放**模型权重文件**。

## 文件格式

- PyTorch checkpoint格式（`.pth`或`.pt`）
- 包含模型权重、优化器状态、epoch信息等

## 保存策略

- 每个epoch保存一次（可选）
- 最佳模型单独保存（基于验证集指标）
- 最终模型保存

## 命名规范

建议命名格式：
- `pix2pix_epoch_{epoch}_best.pth`
- `cyclegan_epoch_{epoch}_best.pth`
- `unet_baseline_final.pth`

## 加载方式

```python
checkpoint = torch.load('checkpoint_path.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

