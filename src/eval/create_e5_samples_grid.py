"""
创建E5实验的样例对比田字形网格图。

对比U-Net (L1) 和 Pix2Pix (L1+GAN) 的生成样例。
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_e5_samples_grid(
    images_dir: Path,
    output_path: Path,
):
    """
    创建E5实验的样例对比田字形网格图。
    
    Args:
        images_dir: 包含样例图片的目录
        output_path: 输出图片路径
    """
    # 图片文件路径
    unet_sample_00 = images_dir / "unet_sample_00_strong_e040.png"
    unet_sample_01 = images_dir / "unet_sample_01_strong_e040.png"
    pix2pix_sample_00 = images_dir / "pix2pix_epoch_050_sample_00.png"
    pix2pix_sample_01 = images_dir / "pix2pix_epoch_050_sample_01.png"
    
    # 检查文件是否存在
    for img_path in [unet_sample_00, unet_sample_01, pix2pix_sample_00, pix2pix_sample_01]:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
    
    # 读取图片
    unet_00 = np.array(Image.open(unet_sample_00))
    unet_01 = np.array(Image.open(unet_sample_01))
    pix2pix_00 = np.array(Image.open(pix2pix_sample_00))
    pix2pix_01 = np.array(Image.open(pix2pix_sample_01))
    
    # 创建2x2田字形布局，更紧凑的尺寸
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 计算图片尺寸（假设所有图片尺寸相同）
    img_height, img_width = unet_00.shape[:2]
    
    # 布局：
    # 左上：U-Net sample 0
    # 右上：Pix2Pix sample 0
    # 左下：U-Net sample 1
    # 右下：Pix2Pix sample 1
    
    # 左上：U-Net sample 0
    axes[0, 0].imshow(unet_00)
    axes[0, 0].axis('off')
    axes[0, 0].text(
        img_width * 0.02, img_height * 0.96,
        'U-Net (L1 only)\nSample 0',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9)
    )
    
    # 右上：Pix2Pix sample 0
    axes[0, 1].imshow(pix2pix_00)
    axes[0, 1].axis('off')
    axes[0, 1].text(
        img_width * 0.02, img_height * 0.96,
        'Pix2Pix (L1+GAN)\nSample 0',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9)
    )
    
    # 左下：U-Net sample 1
    axes[1, 0].imshow(unet_01)
    axes[1, 0].axis('off')
    axes[1, 0].text(
        img_width * 0.02, img_height * 0.96,
        'U-Net (L1 only)\nSample 1',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9)
    )
    
    # 右下：Pix2Pix sample 1
    axes[1, 1].imshow(pix2pix_01)
    axes[1, 1].axis('off')
    axes[1, 1].text(
        img_width * 0.02, img_height * 0.96,
        'Pix2Pix (L1+GAN)\nSample 1',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9)
    )
    
    # 添加列标题（在第一行上方，更紧凑）
    fig.text(0.25, 0.97, 'U-Net (L1 only)', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
    fig.text(0.75, 0.97, 'Pix2Pix (L1+GAN)', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))
    
    # 添加行标签（在左侧，更紧凑）
    fig.text(0.015, 0.75, 'Sample 0', 
             ha='left', va='center', fontsize=16, fontweight='bold',
             rotation=90, bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    fig.text(0.015, 0.25, 'Sample 1', 
             ha='left', va='center', fontsize=16, fontweight='bold',
             rotation=90, bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    
    # 添加总标题
    fig.suptitle(
        'Experiment E5: L1 vs L1+GAN Loss Comparison',
        fontsize=18, fontweight='bold', y=0.99
    )
    
    # 使用更紧凑的布局参数，进一步减小上下间距
    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.02, 
                        wspace=0.01, hspace=0.005)
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Created E5 samples grid: {output_path}")


def main():
    """主函数"""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # E5样例图片目录
    images_dir = PROJECT_ROOT / "outputs" / "images" / "e5_l1_vs_l1_gan"
    
    # 输出路径
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "e5_l1_vs_l1_gan"
    output_path = output_dir / "e5_samples_grid.png"
    
    # 创建网格图
    create_e5_samples_grid(
        images_dir=images_dir,
        output_path=output_path
    )


if __name__ == "__main__":
    main()

