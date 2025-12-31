"""
创建E6实验的样例对比图。

对比最佳 epoch（31）和最终 epoch（50）的生成样例（仅显示样本1）。
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_e6_samples_grid(
    images_dir: Path,
    output_path: Path,
    best_epoch: int = 31,
    final_epoch: int = 50,
    sample_idx: int = 1,
):
    """
    创建E6实验的样例对比图（仅显示样本1）。
    
    Args:
        images_dir: 包含样例图片的目录
        output_path: 输出图片路径
        best_epoch: 最佳 epoch 编号
        final_epoch: 最终 epoch 编号
        sample_idx: 要展示的样本索引（默认1）
    """
    # 图片文件路径
    best_sample = images_dir / f"epoch_{best_epoch:03d}_sample_{sample_idx:02d}.png"
    final_sample = images_dir / f"epoch_{final_epoch:03d}_sample_{sample_idx:02d}.png"
    
    # 检查文件是否存在
    for img_path in [best_sample, final_sample]:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
    
    # 读取图片
    best_img = np.array(Image.open(best_sample))
    final_img = np.array(Image.open(final_sample))
    
    # 创建1x2布局（一行两列）
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 计算图片尺寸（假设所有图片尺寸相同）
    img_height, img_width = best_img.shape[:2]
    
    # 左侧：Best epoch sample 1
    axes[0].imshow(best_img)
    axes[0].axis('off')
    axes[0].text(
        img_width * 0.02, img_height * 0.96,
        f'Best Epoch ({best_epoch})\nSample {sample_idx}',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9)
    )
    
    # 右侧：Final epoch sample 1
    axes[1].imshow(final_img)
    axes[1].axis('off')
    axes[1].text(
        img_width * 0.02, img_height * 0.96,
        f'Final Epoch ({final_epoch})\nSample {sample_idx}',
        ha='left', va='top', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9)
    )
    
    # 添加列标题（在图片上方）
    fig.text(0.25, 0.97, f'Best Epoch ({best_epoch})', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))
    fig.text(0.75, 0.97, f'Final Epoch ({final_epoch})', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))
    
    # 添加总标题
    fig.suptitle(
        'Experiment E6: Pix2Pix (L1 + GAN + Feature Matching) - Best vs Final Epoch Comparison',
        fontsize=18, fontweight='bold', y=0.99
    )
    
    # 使用紧凑的布局参数
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, 
                        wspace=0.02)
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Created E6 samples grid: {output_path}")


def main():
    """主函数"""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # E6样例图片目录（从e6_results_images_only目录读取）
    images_dir = PROJECT_ROOT / "outputs" / "e6_results_images_only" / "images"
    
    # 输出路径
    output_dir = PROJECT_ROOT / "outputs" / "figures" / "e6"
    output_path = output_dir / "e6_samples_grid.png"
    
    # 创建网格图（仅显示样本1）
    create_e6_samples_grid(
        images_dir=images_dir,
        output_path=output_path,
        best_epoch=31,
        final_epoch=50,
        sample_idx=1
    )


if __name__ == "__main__":
    main()

