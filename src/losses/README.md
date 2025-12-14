# src/losses/

## 目录说明

此目录包含**损失函数实现**。

## 文件列表

- `adversarial_loss.py` - 对抗损失（GAN Loss）
- `perceptual_loss.py` - 感知损失（Perceptual Loss）
- `feature_matching.py` - 特征匹配损失（Feature Matching Loss）

## 损失函数说明

### adversarial_loss.py
- 生成器对抗损失：`L_GAN = log(D(x, G(x)))`
- 判别器对抗损失：`L_D = -[log(D(x, y)) + log(1 - D(x, G(x)))]`

### perceptual_loss.py
- 使用预训练VGG网络提取特征
- 计算特征空间距离
- 提升感知质量

### feature_matching.py
- 使用判别器中间层特征
- 计算真实图像和生成图像的特征距离
- 稳定训练过程

## 组合使用

- **Pix2Pix基础**：L1损失 + 对抗损失
- **扩展版本**：L1损失 + 对抗损失 + Feature Matching + Perceptual Loss

