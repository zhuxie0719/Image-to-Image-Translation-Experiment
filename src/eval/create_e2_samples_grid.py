"""
创建E2实验的样例三联图网格图。

将epoch 1、20、36、50的代表性生成样例（Label / Generated / Ground Truth）
组合成一个完整的图片。
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_e2_samples_grid(
    exp_dir: Path,
    output_path: Path,
    epochs: list = [1, 20, 36, 50],
    sample_idx: int = 0,
):
    """
    创建E2实验的样例三联图网格图。
    
    Args:
        exp_dir: 实验输出目录（包含images文件夹）
        output_path: 输出图片路径
        epochs: 要展示的epoch列表
        sample_idx: 使用的样本索引（0或1）
    """
    images_dir = exp_dir / "images"
    
    # 读取所有epoch的图片
    images = []
    for epoch in epochs:
        # 格式化epoch编号（001, 020, 036, 050）
        epoch_str = f"epoch_{epoch:03d}_sample_{sample_idx:02d}.png"
        img_path = images_dir / epoch_str
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = Image.open(img_path)
        images.append(np.array(img))
    
    # 创建图形，2行2列（田字形布局）
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 将axes展平为一维数组，方便索引
    axes_flat = axes.flatten()
    
    # 计算图片尺寸（假设所有图片尺寸相同）
    img_height, img_width = images[0].shape[:2]
    
    # 田字形布局：左上(epoch 1), 右上(epoch 20), 左下(epoch 36), 右下(epoch 50)
    layout_order = [0, 1, 2, 3]  # 对应epochs列表的索引
    
    for layout_idx, epoch_idx in enumerate(layout_order):
        epoch = epochs[epoch_idx]
        img = images[epoch_idx]
        ax = axes_flat[layout_idx]
        
        ax.imshow(img)
        ax.axis('off')
        
        # 添加列标题（只在第一行）
        if layout_idx < 2:  # 第一行的两个子图
            # Label列
            ax.text(
                img_width // 6, -img_height * 0.08,
                'Label',
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )
            # Generated列
            ax.text(
                img_width // 2, -img_height * 0.08,
                'Generated',
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )
            # Ground Truth列
            ax.text(
                5 * img_width // 6, -img_height * 0.08,
                'Ground Truth',
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
            )
        
        # 添加epoch标签（在左上角）
        ax.text(
            img_width * 0.02, img_height * 0.95,
            f'Epoch {epoch}',
            ha='left', va='top', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9)
        )
    
    # 添加总标题
    fig.suptitle(
        'Experiment E2: Pix2Pix (L1 + GAN) - Sample Progression Across Training',
        fontsize=20, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Created E2 samples grid: {output_path}")


def main():
    """主函数"""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # E2实验目录
    exp_dir = PROJECT_ROOT / "outputs" / "pix2pix_l1_gan_strong_e50"
    
    # 输出路径
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "pix2pix_l1_gan_strong_e50"
    output_path = output_dir / "e2_samples_grid.png"
    
    # 创建网格图
    create_e2_samples_grid(
        exp_dir=exp_dir,
        output_path=output_path,
        epochs=[1, 20, 36, 50],
        sample_idx=0
    )


if __name__ == "__main__":
    main()

