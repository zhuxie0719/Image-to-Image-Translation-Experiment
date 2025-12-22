import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_history(exp_dir: Path):
    history_path = exp_dir / "logs" / "history_pix2pix.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Cannot find history file at: {history_path}")

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    return history


def plot_losses(history: dict, out_dir: Path):
    epochs = np.array(history["epoch"])

    train_g_total = np.array(history["train_g_total"])
    train_g_l1 = np.array(history["train_g_l1"])
    train_g_gan = np.array(history["train_g_gan"])
    train_d = np.array(history["train_d"])
    val_l1 = np.array(history["val_l1"])

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_g_total, label="G total (train)")
    plt.plot(epochs, train_g_l1, label="G L1 (train)")
    plt.plot(epochs, train_g_gan, label="G GAN (train)")
    plt.plot(epochs, train_d, label="D loss (train)")
    plt.plot(epochs, val_l1, label="L1 (val)", linestyle="--", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pix2Pix (L1 + GAN, strong aug) - Training / Validation Losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "e2_pix2pix_losses_e50.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metrics(history: dict, out_dir: Path):
    epochs = np.array(history["epoch"])

    psnr = np.array(history["val_psnr"])
    ssim = np.array(history["val_ssim"])
    mae = np.array(history["val_mae"])

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(epochs, psnr, marker="o")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Validation PSNR")
    axes[0].grid(True, alpha=0.3)

    # 使用 Matplotlib 内置调色板中的橙色
    axes[1].plot(epochs, ssim, marker="o", color="tab:orange")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Validation SSIM")
    axes[1].grid(True, alpha=0.3)

    # 使用 Matplotlib 内置调色板中的绿色
    axes[2].plot(epochs, mae, marker="o", color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE / L1")
    axes[2].set_title("Validation MAE (== L1)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = out_dir / "e2_pix2pix_val_metrics_e50.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_main_results_table(history: dict, out_dir: Path):
    epochs = np.array(history["epoch"])
    val_l1 = np.array(history["val_l1"])
    psnr = np.array(history["val_psnr"])
    ssim = np.array(history["val_ssim"])
    mae = np.array(history["val_mae"])

    # best epoch by minimum val_l1 (same as MAE)
    best_idx = int(np.argmin(val_l1))
    best_epoch = int(epochs[best_idx])

    last_idx = len(epochs) - 1
    last_epoch = int(epochs[last_idx])

    lines = []
    lines.append("| 设置 | Epoch | Val L1/MAE ↓ | PSNR ↑ | SSIM ↑ |")
    lines.append("|------|-------|--------------|--------|--------|")
    lines.append(
        f"| 最佳 (按最小 Val L1) | {best_epoch} | "
        f"{val_l1[best_idx]:.4f} | {psnr[best_idx]:.3f} | {ssim[best_idx]:.3f} |"
    )
    lines.append(
        f"| 最后 (Epoch {last_epoch}) | {last_epoch} | "
        f"{val_l1[last_idx]:.4f} | {psnr[last_idx]:.3f} | {ssim[last_idx]:.3f} |"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "e2_pix2pix_main_results_e50.md"
    with table_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 也打印到终端，方便快速查看
    print("\n".join(lines))


def collect_sample_images(exp_dir: Path, out_images_root: Path):
    """把实验里的样例三联图复制到统一的 outputs/images 目录，方便论文使用。"""
    src_dir = exp_dir / "images"
    if not src_dir.exists():
        print(f"[WARN] images dir not found: {src_dir}")
        return

    import shutil

    dst_dir = out_images_root / "pix2pix_l1_gan_strong_e50"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 直接复制全部 PNG，保持原始文件名
    for img_path in sorted(src_dir.glob("*.png")):
        shutil.copy2(img_path, dst_dir / img_path.name)

    print(f"[OK] Copied sample images to: {dst_dir}")


def main():
    exp_dir = PROJECT_ROOT / "outputs" / "pix2pix_l1_gan_strong_e50"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "pix2pix_l1_gan_strong_e50"
    images_root = PROJECT_ROOT / "outputs" / "images"

    history = load_history(exp_dir)

    plot_losses(history, figures_dir)
    plot_metrics(history, figures_dir)
    make_main_results_table(history, figures_dir)
    collect_sample_images(exp_dir, images_root)

    print("[OK] Pix2Pix E2 (50 epochs) analysis figures and table generated.")
    print(f"  - Loss & metrics figures: {figures_dir}")
    print(f"  - Sample images copied under: {images_root / 'pix2pix_l1_gan_strong_e50'}")


if __name__ == "__main__":
    main()


