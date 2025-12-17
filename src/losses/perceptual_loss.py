"""
Perceptual Loss（感知损失）实现。

参考 Guide.md 7.2.2：
- 使用预训练 VGG 网络提取特征，计算特征空间距离。

说明：
- 这里使用 torchvision.models.vgg16 作为特征提取 backbone。
- 为避免测试时下载权重失败，PerceptualLoss 支持 `pretrained=False`，
  在单元测试中默认使用随机初始化的 VGG；实际训练时建议设置为 True。
"""

from typing import List

import torch
import torch.nn as nn

try:
    from torchvision import models
except ImportError as e:  # pragma: no cover - 环境缺少 torchvision 时提示
    raise ImportError(
        "torchvision 未安装，无法使用 PerceptualLoss。"
    ) from e


class VGGFeatureExtractor(nn.Module):
    """
    从 VGG16 中提取多层特征，用于感知损失计算。

    默认提取的层（以 conv 为单位，大致对应感知层次）：
    - relu1_2, relu2_2, relu3_3, relu4_3
    """

    def __init__(self, pretrained: bool = False, requires_grad: bool = False):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        features = vgg16.features

        # 根据 VGG16 结构划分若干 stage
        # 0-4:  conv1_x
        # 5-9:  conv2_x
        # 10-16: conv3_x
        # 17-23: conv4_x
        self.stage1 = nn.Sequential(*features[:4])   # relu1_2 之前
        self.stage2 = nn.Sequential(*features[4:9])  # relu2_2 之前
        self.stage3 = nn.Sequential(*features[9:16])  # relu3_3 之前
        self.stage4 = nn.Sequential(*features[16:23])  # relu4_3 之前

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.stage1(x)
        feats.append(x)
        x = self.stage2(x)
        feats.append(x)
        x = self.stage3(x)
        feats.append(x)
        x = self.stage4(x)
        feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    """
    感知损失（Perceptual Loss）。

    步骤：
    1. 将输入从 [-1, 1] 映射到 [0, 1]，再按 ImageNet 统计量归一化。
    2. 使用 VGGFeatureExtractor 提取多层特征。
    3. 对每一层特征使用 L1 损失并求和/平均。
    """

    def __init__(
        self,
        pretrained: bool = False,
        feature_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.vgg = VGGFeatureExtractor(pretrained=pretrained, requires_grad=False)
        self.criterion = nn.L1Loss(reduction=reduction)
        self.feature_weight = feature_weight

        # ImageNet 归一化参数
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入从 [-1, 1] 映射到 [0, 1]，再使用 ImageNet 均值/方差归一化。
        """
        # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        # 标准化
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 预处理
        pred_norm = self._preprocess(pred)
        target_norm = self._preprocess(target)

        # 提取多层特征
        pred_feats = self.vgg(pred_norm)
        target_feats = self.vgg(target_norm)

        # 逐层计算 L1 损失并取平均
        losses = [
            self.criterion(p, t) for p, t in zip(pred_feats, target_feats)
        ]
        loss = sum(losses) / len(losses)
        return loss * self.feature_weight


def test_perceptual_loss() -> None:
    """简单测试 Perceptual Loss。"""
    print("=" * 60)
    print("Testing Perceptual Loss")
    print("=" * 60)

    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256)
    y = x + 0.05 * torch.randn_like(x)

    criterion = PerceptualLoss(pretrained=False)  # 测试时不加载预训练权重
    loss = criterion(x, y)
    print(f"\nPerceptual Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("[OK] Perceptual Loss test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_perceptual_loss()


