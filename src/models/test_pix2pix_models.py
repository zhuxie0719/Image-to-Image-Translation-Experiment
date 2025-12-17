"""
测试Pix2Pix生成器和判别器的联合使用。
"""

import torch
from generator import UNetGenerator
from discriminator import PatchGANDiscriminator


def test_pix2pix_models():
    """测试生成器和判别器的联合使用"""
    print("=" * 60)
    print("Testing Pix2Pix Generator and Discriminator")
    print("=" * 60)
    
    # 创建模型
    generator = UNetGenerator(in_channels=3, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=6)
    
    generator.eval()
    discriminator.eval()
    
    # 测试输入
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
        print(f"  Fake photo range: [{fake_photo.min():.3f}, {fake_photo.max():.3f}]")
        
        # 判别器前向传播（真实图像）
        d_real = discriminator(label, real_photo)
        print(f"\nDiscriminator output (real):")
        print(f"  D(real): {d_real.shape}")
        print(f"  D(real) range: [{d_real.min():.3f}, {d_real.max():.3f}]")
        print(f"  D(real) mean: {d_real.mean():.3f}")
        
        # 判别器前向传播（生成图像）
        d_fake = discriminator(label, fake_photo)
        print(f"\nDiscriminator output (fake):")
        print(f"  D(fake): {d_fake.shape}")
        print(f"  D(fake) range: [{d_fake.min():.3f}, {d_fake.max():.3f}]")
        print(f"  D(fake) mean: {d_fake.mean():.3f}")
    
    # 计算参数量
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nModel parameters:")
    print(f"  Generator: {g_params:,}")
    print(f"  Discriminator: {d_params:,}")
    print(f"  Total: {g_params + d_params:,}")
    
    print("\n" + "=" * 60)
    print("[OK] Pix2Pix models test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_pix2pix_models()

