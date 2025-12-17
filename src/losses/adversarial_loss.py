"""
Pix2Pix 对抗损失（GAN Loss）实现。

参考 Guide.md 7.2.1：
- 生成器对抗损失：L_GAN = log(D(x, G(x)))
- 判别器对抗损失：L_D = -[log(D(x, y)) + log(1 - D(x, G(x)))]

在PyTorch中，使用BCEWithLogitsLoss或BCELoss实现。
"""

import torch
import torch.nn as nn


def generator_adversarial_loss(discriminator_output, target_is_real=True):
    """
    计算生成器的对抗损失。
    
    Args:
        discriminator_output: 判别器对生成图像的输出 [B, 1, H, W] 或 [B, 1]
        target_is_real: 目标标签（True表示希望判别器认为生成图像是真实的）
    
    Returns:
        loss: 标量损失值
    """
    if target_is_real:
        # 我们希望判别器输出接近1（真实）
        target = torch.ones_like(discriminator_output)
    else:
        # 我们希望判别器输出接近0（虚假）
        target = torch.zeros_like(discriminator_output)
    
    # 使用BCELoss（因为判别器输出已经经过Sigmoid）
    criterion = nn.BCELoss()
    loss = criterion(discriminator_output, target)
    
    return loss


def discriminator_adversarial_loss(discriminator_real, discriminator_fake):
    """
    计算判别器的对抗损失。
    
    L_D = -[log(D(x, y)) + log(1 - D(x, G(x)))]
      = -log(D(x, y)) - log(1 - D(x, G(x)))
    
    在PyTorch中，这等价于：
    - 真实图像：希望D(x, y)接近1，使用BCELoss(target=1)
    - 生成图像：希望D(x, G(x))接近0，使用BCELoss(target=0)
    
    Args:
        discriminator_real: 判别器对真实图像的输出 [B, 1, H, W] 或 [B, 1]
        discriminator_fake: 判别器对生成图像的输出 [B, 1, H, W] 或 [B, 1]
    
    Returns:
        loss: 标量损失值
    """
    criterion = nn.BCELoss()
    
    # 真实图像：希望输出接近1
    real_target = torch.ones_like(discriminator_real)
    real_loss = criterion(discriminator_real, real_target)
    
    # 生成图像：希望输出接近0
    fake_target = torch.zeros_like(discriminator_fake)
    fake_loss = criterion(discriminator_fake, fake_target)
    
    # 总损失
    total_loss = (real_loss + fake_loss) * 0.5  # 平均化
    
    return total_loss


def generator_adversarial_loss_with_logits(discriminator_logits, target_is_real=True):
    """
    计算生成器的对抗损失（使用BCEWithLogitsLoss，适用于判别器输出未经过Sigmoid的情况）。
    
    Args:
        discriminator_logits: 判别器对生成图像的logits输出 [B, 1, H, W] 或 [B, 1]
        target_is_real: 目标标签（True表示希望判别器认为生成图像是真实的）
    
    Returns:
        loss: 标量损失值
    """
    if target_is_real:
        target = torch.ones_like(discriminator_logits)
    else:
        target = torch.zeros_like(discriminator_logits)
    
    # 使用BCEWithLogitsLoss（内部包含Sigmoid + BCE）
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discriminator_logits, target)
    
    return loss


def discriminator_adversarial_loss_with_logits(discriminator_real_logits, discriminator_fake_logits):
    """
    计算判别器的对抗损失（使用BCEWithLogitsLoss）。
    
    Args:
        discriminator_real_logits: 判别器对真实图像的logits输出 [B, 1, H, W] 或 [B, 1]
        discriminator_fake_logits: 判别器对生成图像的logits输出 [B, 1, H, W] 或 [B, 1]
    
    Returns:
        loss: 标量损失值
    """
    criterion = nn.BCEWithLogitsLoss()
    
    # 真实图像：希望输出接近1
    real_target = torch.ones_like(discriminator_real_logits)
    real_loss = criterion(discriminator_real_logits, real_target)
    
    # 生成图像：希望输出接近0
    fake_target = torch.zeros_like(discriminator_fake_logits)
    fake_loss = criterion(discriminator_fake_logits, fake_target)
    
    # 总损失
    total_loss = (real_loss + fake_loss) * 0.5  # 平均化
    
    return total_loss


def test_adversarial_loss():
    """测试对抗损失函数"""
    print("=" * 60)
    print("Testing Adversarial Loss Functions")
    print("=" * 60)
    
    batch_size = 2
    height, width = 15, 15  # PatchGAN输出尺寸
    
    # 模拟判别器输出（已经过Sigmoid，范围[0, 1]）
    discriminator_real = torch.rand(batch_size, 1, height, width) * 0.3 + 0.7  # 接近1
    discriminator_fake = torch.rand(batch_size, 1, height, width) * 0.3  # 接近0
    
    print(f"\nDiscriminator outputs (after Sigmoid):")
    print(f"  Real: shape={discriminator_real.shape}, range=[{discriminator_real.min():.3f}, {discriminator_real.max():.3f}]")
    print(f"  Fake: shape={discriminator_fake.shape}, range=[{discriminator_fake.min():.3f}, {discriminator_fake.max():.3f}]")
    
    # 测试生成器损失
    g_loss = generator_adversarial_loss(discriminator_fake, target_is_real=True)
    print(f"\nGenerator adversarial loss: {g_loss.item():.4f}")
    
    # 测试判别器损失
    d_loss = discriminator_adversarial_loss(discriminator_real, discriminator_fake)
    print(f"Discriminator adversarial loss: {d_loss.item():.4f}")
    
    # 测试使用logits的版本
    discriminator_real_logits = torch.randn(batch_size, 1, height, width)
    discriminator_fake_logits = torch.randn(batch_size, 1, height, width)
    
    print(f"\nDiscriminator logits (before Sigmoid):")
    print(f"  Real logits: shape={discriminator_real_logits.shape}")
    print(f"  Fake logits: shape={discriminator_fake_logits.shape}")
    
    g_loss_logits = generator_adversarial_loss_with_logits(discriminator_fake_logits, target_is_real=True)
    print(f"\nGenerator adversarial loss (with logits): {g_loss_logits.item():.4f}")
    
    d_loss_logits = discriminator_adversarial_loss_with_logits(discriminator_real_logits, discriminator_fake_logits)
    print(f"Discriminator adversarial loss (with logits): {d_loss_logits.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] Adversarial loss test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_adversarial_loss()

