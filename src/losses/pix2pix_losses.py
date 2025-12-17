"""
Pix2Pix 损失函数工具模块。

实现 Guide.md 7.2.1 中的基础损失：
1. L1损失（像素级重建损失）：L_L1 = ||G(x) - y||_1
2. 对抗损失（GAN损失）：见 adversarial_loss.py
3. 组合损失：L_total = L_GAN + λ * L_L1（λ通常为100）
"""

import torch
import torch.nn as nn
from adversarial_loss import (
    generator_adversarial_loss,
    discriminator_adversarial_loss,
    generator_adversarial_loss_with_logits,
    discriminator_adversarial_loss_with_logits
)


def l1_loss(predicted, target):
    """
    计算L1损失（像素级重建损失）。
    
    L_L1 = ||G(x) - y||_1
    
    Args:
        predicted: 预测图像 [B, C, H, W]
        target: 真实图像 [B, C, H, W]
    
    Returns:
        loss: 标量损失值
    """
    criterion = nn.L1Loss()
    return criterion(predicted, target)


def pix2pix_generator_loss(
    fake_image,
    real_image,
    discriminator_output,
    lambda_l1=100.0,
    use_logits=False
):
    """
    计算Pix2Pix生成器的总损失。
    
    L_total = L_GAN + λ * L_L1
    
    Args:
        fake_image: 生成器生成的图像 [B, C, H, W]
        real_image: 真实图像 [B, C, H, W]
        discriminator_output: 判别器对生成图像的输出 [B, 1, H', W']
        lambda_l1: L1损失的权重（默认100）
        use_logits: 是否使用logits版本的对抗损失
    
    Returns:
        total_loss: 总损失
        gan_loss: GAN损失
        l1_loss_value: L1损失
    """
    # L1损失
    l1_loss_value = l1_loss(fake_image, real_image)
    
    # 对抗损失
    if use_logits:
        gan_loss = generator_adversarial_loss_with_logits(discriminator_output, target_is_real=True)
    else:
        gan_loss = generator_adversarial_loss(discriminator_output, target_is_real=True)
    
    # 总损失
    total_loss = gan_loss + lambda_l1 * l1_loss_value
    
    return total_loss, gan_loss, l1_loss_value


def pix2pix_discriminator_loss(
    discriminator_real_output,
    discriminator_fake_output,
    use_logits=False
):
    """
    计算Pix2Pix判别器的损失。
    
    L_D = -[log(D(x, y)) + log(1 - D(x, G(x)))]
    
    Args:
        discriminator_real_output: 判别器对真实图像的输出 [B, 1, H', W']
        discriminator_fake_output: 判别器对生成图像的输出 [B, 1, H', W']
        use_logits: 是否使用logits版本的对抗损失
    
    Returns:
        loss: 判别器损失
    """
    if use_logits:
        loss = discriminator_adversarial_loss_with_logits(
            discriminator_real_output,
            discriminator_fake_output
        )
    else:
        loss = discriminator_adversarial_loss(
            discriminator_real_output,
            discriminator_fake_output
        )
    
    return loss


def test_pix2pix_losses():
    """测试Pix2Pix损失函数"""
    print("=" * 60)
    print("Testing Pix2Pix Loss Functions")
    print("=" * 60)
    
    batch_size = 2
    channels = 3
    height, width = 256, 256
    
    # 模拟数据
    fake_image = torch.randn(batch_size, channels, height, width)
    real_image = torch.randn(batch_size, channels, height, width)
    
    # 模拟判别器输出（已经过Sigmoid）
    disc_output_size = 15  # PatchGAN输出尺寸
    discriminator_real_output = torch.rand(batch_size, 1, disc_output_size, disc_output_size) * 0.3 + 0.7
    discriminator_fake_output = torch.rand(batch_size, 1, disc_output_size, disc_output_size) * 0.3
    
    print(f"\nInput shapes:")
    print(f"  Fake image: {fake_image.shape}")
    print(f"  Real image: {real_image.shape}")
    print(f"  Discriminator real output: {discriminator_real_output.shape}")
    print(f"  Discriminator fake output: {discriminator_fake_output.shape}")
    
    # 测试L1损失
    l1 = l1_loss(fake_image, real_image)
    print(f"\nL1 Loss: {l1.item():.4f}")
    
    # 测试生成器总损失
    lambda_l1 = 100.0
    g_total, g_gan, g_l1 = pix2pix_generator_loss(
        fake_image,
        real_image,
        discriminator_fake_output,
        lambda_l1=lambda_l1
    )
    print(f"\nGenerator Loss:")
    print(f"  GAN loss: {g_gan.item():.4f}")
    print(f"  L1 loss: {g_l1.item():.4f}")
    print(f"  Total loss (GAN + {lambda_l1} * L1): {g_total.item():.4f}")
    
    # 测试判别器损失
    d_loss = pix2pix_discriminator_loss(
        discriminator_real_output,
        discriminator_fake_output
    )
    print(f"\nDiscriminator Loss: {d_loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] Pix2Pix losses test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pix2pix_losses()

