# src/models/

## 目录说明

此目录包含**模型架构定义**。

## 文件列表

- `generator.py` - U-Net生成器（Pix2Pix）
- `discriminator.py` - PatchGAN判别器（Pix2Pix）
- `unet_baseline.py` - U-Net基线模型（仅L1损失）
- `cyclegan_generator.py` - CycleGAN生成器（ResNet-based）
- `cyclegan_discriminator.py` - CycleGAN判别器

## 模型说明

### Pix2Pix模型
- **生成器**：U-Net架构，带skip connections
- **判别器**：70×70 PatchGAN

### CycleGAN模型
- **生成器**：ResNet-based架构（G: Label→Photo, F: Photo→Label）
- **判别器**：70×70 PatchGAN（D_photo, D_label）

### U-Net基线
- 仅使用L1损失，无GAN组件
- 用于对比实验

