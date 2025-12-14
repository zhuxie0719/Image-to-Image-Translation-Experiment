# src/training/

## 目录说明

此目录包含**训练脚本**。

## 文件列表

- `train_pix2pix.py` - Pix2Pix训练脚本
- `train_unet.py` - U-Net基线训练脚本
- `train_cyclegan.py` - CycleGAN训练脚本
- `trainer.py` - 训练器基类（可选，用于代码复用）

## 训练流程

### Pix2Pix训练
1. 加载数据集
2. 初始化生成器和判别器
3. 交替训练生成器和判别器
4. 每个epoch保存checkpoint和验证集三联图

### CycleGAN训练
1. 加载数据集（两个域）
2. 初始化两个生成器和两个判别器
3. 训练四个网络（G, F, D_photo, D_label）
4. 计算循环一致性损失

### U-Net基线训练
1. 仅使用L1损失
2. 无对抗训练
3. 作为对比基线

## 输出

- 模型checkpoints保存到 `outputs/checkpoints/`
- 训练日志保存到 `outputs/logs/`
- 验证集三联图保存到 `outputs/images/`

