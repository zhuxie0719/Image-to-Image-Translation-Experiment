"""
创建E7实验的样例对比图。

展示 E7 在最佳 epoch（26）和最终 epoch（50）的生成样例，并与 E2、E6 进行对比。
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_e7_samples_grid(
    e2_images_dir: Path,
    e6_images_dir: Path,
    e7_images_dir: Path,
    output_path: Path,
    sample_idx: int = 1,
):
    """
    创建E7实验的样例对比图。
    
    Args:
        e2_images_dir: E2 实验的图片目录
        e6_images_dir: E6 实验的图片目录
        e7_images_dir: E7 实验的图片目录
        output_path: 输出图片路径
        sample_idx: 要展示的样本索引（默认1）
    """
    # E2: 最佳 epoch 36, 最终 epoch 50
    # E6: 最佳 epoch 31, 最终 epoch 50
    # E7: 最佳 epoch 26, 最终 epoch 50
    
    # 图片文件路径
    e2_best = e2_images_dir / f"epoch_036_sample_{sample_idx:02d}.png"
    e2_final = e2_images_dir / f"epoch_050_sample_{sample_idx:02d}.png"
    e6_best = e6_images_dir / f"epoch_031_sample_{sample_idx:02d}.png"
    e6_final = e6_images_dir / f"epoch_050_sample_{sample_idx:02d}.png"
    e7_best = e7_images_dir / f"epoch_026_sample_{sample_idx:02d}.png"
    e7_final = e7_images_dir / f"epoch_050_sample_{sample_idx:02d}.png"
    
    # 检查文件是否存在
    all_paths = [e2_best, e2_final, e6_best, e6_final, e7_best, e7_final]
    for img_path in all_paths:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
    
    # 读取图片
    e2_best_img = np.array(Image.open(e2_best))
    e2_final_img = np.array(Image.open(e2_final))
    e6_best_img = np.array(Image.open(e6_best))
    e6_final_img = np.array(Image.open(e6_final))
    e7_best_img = np.array(Image.open(e7_best))
    e7_final_img = np.array(Image.open(e7_final))
    
    # 创建2x3布局（两行三列）
    # 第一行：E2最佳, E6最佳, E7最佳
    # 第二行：E2最终, E6最终, E7最终
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    
    # 计算图片尺寸（假设所有图片尺寸相同）
    img_height, img_width = e2_best_img.shape[:2]
    
    # 第一行：最佳 epoch
    # E2 最佳 (epoch 36)
    axes[0, 0].imshow(e2_best_img)
    axes[0, 0].axis('off')
    axes[0, 0].text(
        img_width * 0.02, img_height * 0.96,
        'E2 (L1+GAN)\nBest Epoch (36)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9)
    )
    
    # E6 最佳 (epoch 31)
    axes[0, 1].imshow(e6_best_img)
    axes[0, 1].axis('off')
    axes[0, 1].text(
        img_width * 0.02, img_height * 0.96,
        'E6 (L1+GAN+FM)\nBest Epoch (31)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9)
    )
    
    # E7 最佳 (epoch 26)
    axes[0, 2].imshow(e7_best_img)
    axes[0, 2].axis('off')
    axes[0, 2].text(
        img_width * 0.02, img_height * 0.96,
        'E7 (L1+GAN+Perc)\nBest Epoch (26)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9)
    )
    
    # 第二行：最终 epoch
    # E2 最终 (epoch 50)
    axes[1, 0].imshow(e2_final_img)
    axes[1, 0].axis('off')
    axes[1, 0].text(
        img_width * 0.02, img_height * 0.96,
        'E2 (L1+GAN)\nFinal Epoch (50)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9)
    )
    
    # E6 最终 (epoch 50)
    axes[1, 1].imshow(e6_final_img)
    axes[1, 1].axis('off')
    axes[1, 1].text(
        img_width * 0.02, img_height * 0.96,
        'E6 (L1+GAN+FM)\nFinal Epoch (50)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9)
    )
    
    # E7 最终 (epoch 50)
    axes[1, 2].imshow(e7_final_img)
    axes[1, 2].axis('off')
    axes[1, 2].text(
        img_width * 0.02, img_height * 0.96,
        'E7 (L1+GAN+Perc)\nFinal Epoch (50)',
        ha='left', va='top', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9)
    )
    
    # 添加列标题（在第一行上方）
    fig.text(0.17, 0.97, 'E2 (L1+GAN)', 
             ha='center', va='top', fontsize=15, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    fig.text(0.50, 0.97, 'E6 (L1+GAN+FM)', 
             ha='center', va='top', fontsize=15, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
    fig.text(0.83, 0.97, 'E7 (L1+GAN+Perc)', 
             ha='center', va='top', fontsize=15, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))
    
    # 添加行标签（在左侧）
    fig.text(0.015, 0.75, 'Best Epoch', 
             ha='left', va='center', fontsize=15, fontweight='bold',
             rotation=90, bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    fig.text(0.015, 0.25, 'Final Epoch', 
             ha='left', va='center', fontsize=15, fontweight='bold',
             rotation=90, bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    
    # 添加总标题
    fig.suptitle(
        'Experiment E7: Comparison of E2, E6, and E7 at Best and Final Epochs',
        fontsize=18, fontweight='bold', y=0.99
    )
    
    # 使用紧凑的布局参数
    plt.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.02, 
                        wspace=0.02, hspace=0.02)
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Created E7 samples comparison grid: {output_path}")


def main():
    """主函数"""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # 各实验的图片目录
    e2_images_dir = PROJECT_ROOT / "outputs" / "pix2pix_l1_gan_strong_e50" / "images"
    e6_images_dir = PROJECT_ROOT / "outputs" / "e6_results_images_only" / "images"
    e7_images_dir = PROJECT_ROOT / "outputs" / "e7_results" / "images"
    
    # 输出路径
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "e7"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e7_samples_grid.png"
    
    # 创建对比图（使用样本1）
    create_e7_samples_grid(
        e2_images_dir=e2_images_dir,
        e6_images_dir=e6_images_dir,
        e7_images_dir=e7_images_dir,
        output_path=output_path,
        sample_idx=1
    )


if __name__ == "__main__":
    main()

