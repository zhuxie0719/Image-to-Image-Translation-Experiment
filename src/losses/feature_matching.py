"""
Feature Matching Loss 实现。

参考 Guide.md 7.2.2：
- 使用判别器中间层特征，计算真实图像和生成图像的特征距离。

典型做法：
- 从判别器中提取多层特征 {f_l(real), f_l(fake)}
- 对每一层计算 L1 距离，并在所有层上求和/求平均。
"""

from typing import List, Dict, Union

import torch
import torch.nn as nn


FeaturesType = Union[List[torch.Tensor], Dict[str, torch.Tensor]]


class FeatureMatchingLoss(nn.Module):
    """
    特征匹配损失（Feature Matching Loss）。

    支持两种输入形式：
    - List[Tensor]：按顺序的特征列表
    - Dict[str, Tensor]：以层名为键的特征字典

    每层使用 L1Loss，并在所有层上取平均。
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, real_features: FeaturesType, fake_features: FeaturesType) -> torch.Tensor:
        if isinstance(real_features, dict):
            assert isinstance(fake_features, dict), "real_features 为 dict 时，fake_features 也必须为 dict"
            keys = real_features.keys()
            losses = [
                self.criterion(real_features[k], fake_features[k]) for k in keys
            ]
        else:
            # 视为 List[Tensor]
            assert isinstance(fake_features, (list, tuple)), "real_features 为 list 时，fake_features 也必须为 list/tuple"
            assert len(real_features) == len(fake_features), "real_features 与 fake_features 的层数必须一致"
            losses = [
                self.criterion(r, f) for r, f in zip(real_features, fake_features)
            ]

        # 对所有层的损失取平均
        loss = sum(losses) / len(losses)
        return loss


def test_feature_matching_loss() -> None:
    """简单测试 Feature Matching Loss。"""
    print("=" * 60)
    print("Testing Feature Matching Loss")
    print("=" * 60)

    batch_size = 2
    channels = [64, 128, 256]
    heights = [64, 32, 16]
    widths = [64, 32, 16]

    # 使用 List[Tensor] 形式
    real_feats_list = [
        torch.randn(batch_size, c, h, w) for c, h, w in zip(channels, heights, widths)
    ]
    fake_feats_list = [
        r + 0.1 * torch.randn_like(r) for r in real_feats_list
    ]

    criterion = FeatureMatchingLoss()
    loss_list = criterion(real_feats_list, fake_feats_list)
    print(f"\nFeature Matching Loss (list): {loss_list.item():.4f}")

    # 使用 Dict[str, Tensor] 形式
    real_feats_dict = {f"layer_{i}": feat for i, feat in enumerate(real_feats_list)}
    fake_feats_dict = {f"layer_{i}": feat for i, feat in enumerate(fake_feats_list)}

    loss_dict = criterion(real_feats_dict, fake_feats_dict)
    print(f"Feature Matching Loss (dict): {loss_dict.item():.4f}")

    print("\n" + "=" * 60)
    print("[OK] Feature Matching Loss test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_matching_loss()


