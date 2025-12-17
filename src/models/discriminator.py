"""
Pix2Pix PatchGAN判别器。

架构（参考论文附录6.1.2）：
- 70×70 PatchGAN（论文推荐）：C64-C128-C256-C512
  - 所有层：Convolution(4×4, stride=2) + BatchNorm + LeakyReLU(0.2)
  - 第一层C64不使用BatchNorm
  - 最后层：Convolution映射到1维输出 + Sigmoid
  - 输入：concatenate(label, image)，输出：70×70的patch判别结果

注意：70×70指的是每个输出patch对应输入图像中70×70的感受野区域。
对于256×256输入，经过4层下采样（每层stride=2），输出尺寸为16×16。
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN判别器（70×70）。
    
    输入：concatenate(label, image) [B, 6, 256, 256]
    输出：patch判别结果 [B, 1, 16, 16]（每个patch对应输入中70×70的感受野）
    """
    def __init__(self, in_channels=6):
        super(PatchGANDiscriminator, self).__init__()
        
        # C64 (第一层，无BatchNorm)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # C512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 最后层：映射到1维输出 + Sigmoid
        # 输入：512通道，16×16
        # 输出：1通道，16×16（每个像素对应输入中70×70的感受野）
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, label, image):
        """
        前向传播。
        
        Args:
            label: [B, 3, H, W] - 语义标签图
            image: [B, 3, H, W] - 真实或生成的图像
        
        Returns:
            output: [B, 1, H', W'] - patch判别结果
        """
        # 拼接label和image
        x = torch.cat([label, image], dim=1)  # [B, 6, H, W]
        
        # 通过卷积层
        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.conv2(x)  # [B, 128, H/4, W/4]
        x = self.conv3(x)  # [B, 256, H/8, W/8]
        x = self.conv4(x)  # [B, 512, H/16, W/16]
        
        # 最后层
        output = self.final(x)  # [B, 1, H/16, W/16]
        
        return output


def test_patchgan_discriminator():
    """测试PatchGAN判别器的前向传播"""
    model = PatchGANDiscriminator(in_channels=6)
    model.eval()
    
    # 测试输入
    label = torch.randn(1, 3, 256, 256)
    image = torch.randn(1, 3, 256, 256)
    print(f"Label shape: {label.shape}")
    print(f"Image shape: {image.shape}")
    
    with torch.no_grad():
        output = model(label, image)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Expected range: [0, 1] (Sigmoid activation)")
        print(f"Expected output size: [1, 1, 16, 16] for 256×256 input")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n[OK] PatchGAN discriminator test passed!")


if __name__ == "__main__":
    test_patchgan_discriminator()

