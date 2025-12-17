"""
测试损失函数与Pix2Pix模型的集成。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.losses.pix2pix_losses import (
    pix2pix_generator_loss,
    pix2pix_discriminator_loss
)


def test_losses_with_models():
    """测试损失函数与模型的集成"""
    print("=" * 60)
    print("Testing Loss Functions with Pix2Pix Models")
    print("=" * 60)
    
    # 创建模型
    generator = UNetGenerator(in_channels=3, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=6)
    
    generator.eval()
    discriminator.eval()
    
    # 模拟输入数据
    batch_size = 2
    label = torch.randn(batch_size, 3, 256, 256)
    real_photo = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\nInput shapes:")
    print(f"  Label: {label.shape}")
    print(f"  Real photo: {real_photo.shape}")
    
    with torch.no_grad():
        # 生成器前向传播
        fake_photo = generator(label)
        print(f"\nGenerator output:")
        print(f"  Fake photo: {fake_photo.shape}")
        
        # 判别器前向传播
        d_real = discriminator(label, real_photo)
        d_fake = discriminator(label, fake_photo)
        
        print(f"\nDiscriminator outputs:")
        print(f"  D(real): {d_real.shape}")
        print(f"  D(fake): {d_fake.shape}")
        
        # 计算损失
        lambda_l1 = 100.0
        g_total, g_gan, g_l1 = pix2pix_generator_loss(
            fake_photo,
            real_photo,
            d_fake,
            lambda_l1=lambda_l1
        )
        
        d_loss = pix2pix_discriminator_loss(d_real, d_fake)
        
        print(f"\nLoss values:")
        print(f"  Generator GAN loss: {g_gan.item():.4f}")
        print(f"  Generator L1 loss: {g_l1.item():.4f}")
        print(f"  Generator total loss (GAN + {lambda_l1} * L1): {g_total.item():.4f}")
        print(f"  Discriminator loss: {d_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] Loss functions integration test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_losses_with_models()

