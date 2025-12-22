import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_unet_strong_results(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 选择 strong 增强配置
    strong_row = None
    for row in rows:
        if row["Augmentation"] == "strong":
            strong_row = row
            break

    if strong_row is None:
        raise RuntimeError("Cannot find 'strong' row in U-Net ablation results.")

    # 转成 float
    return {
        "model": "U-Net (L1, strong aug)",
        "final_val_l1": float(strong_row["Final Val Loss"]),
        "final_psnr": float(strong_row["Final PSNR (dB)"]),
        "final_ssim": float(strong_row["Final SSIM"]),
        "final_mae": float(strong_row["Final MAE"]),
        "best_psnr": float(strong_row["Best PSNR (dB)"]),
        "best_ssim": float(strong_row["Best SSIM"]),
        "best_epoch_psnr": int(strong_row["Best Epoch (PSNR)"]),
    }


def load_pix2pix_results(md_path: Path):
    # 从简单的 markdown 表中解析出 2 行结果
    lines = md_path.read_text(encoding="utf-8").strip().splitlines()
    # 期望格式：
    # | 设置 | Epoch | Val L1/MAE ↓ | PSNR ↑ | SSIM ↑ |
    # |------| ...  |
    # | 最佳 ... |
    # | 最后 ... |
    best_parts = [p.strip() for p in lines[2].strip("|").split("|")]
    last_parts = [p.strip() for p in lines[3].strip("|").split("|")]

    best_epoch = int(best_parts[1])
    best_l1 = float(best_parts[2])
    best_psnr = float(best_parts[3])
    best_ssim = float(best_parts[4])

    last_epoch = int(last_parts[1])
    last_l1 = float(last_parts[2])
    last_psnr = float(last_parts[3])
    last_ssim = float(last_parts[4])

    return {
        "model": "Pix2Pix (L1+GAN, strong aug)",
        "best_epoch": best_epoch,
        "best_val_l1": best_l1,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "last_epoch": last_epoch,
        "last_val_l1": last_l1,
        "last_psnr": last_psnr,
        "last_ssim": last_ssim,
    }


def make_e5_comparison_table(unet_res: dict, pix2pix_res: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("| 模型 | 增强 | 选取策略 | Epoch | Val L1/MAE ↓ | PSNR ↑ | SSIM ↑ |")
    lines.append("|------|------|----------|-------|--------------|--------|--------|")

    # U-Net：用 strong 配置的 final 指标 + best PSNR 对应 epoch 信息
    lines.append(
        f"| U-Net | strong | Final (40 epoch) | 40 | "
        f"{unet_res['final_mae']:.4f} | {unet_res['final_psnr']:.3f} | {unet_res['final_ssim']:.3f} |"
    )
    lines.append(
        f"| U-Net | strong | Best PSNR | {unet_res['best_epoch_psnr']} | "
        f"{unet_res['final_mae']:.4f} | {unet_res['best_psnr']:.3f} | {unet_res['best_ssim']:.3f} |"
    )

    # Pix2Pix：最佳 L1 和最后一个 epoch
    lines.append(
        f"| Pix2Pix | strong | Best Val L1 | {pix2pix_res['best_epoch']} | "
        f"{pix2pix_res['best_val_l1']:.4f} | {pix2pix_res['best_psnr']:.3f} | {pix2pix_res['best_ssim']:.3f} |"
    )
    lines.append(
        f"| Pix2Pix | strong | Last (50 epoch) | {pix2pix_res['last_epoch']} | "
        f"{pix2pix_res['last_val_l1']:.4f} | {pix2pix_res['last_psnr']:.3f} | {pix2pix_res['last_ssim']:.3f} |"
    )

    table_path = out_dir / "e5_l1_vs_l1_gan_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(table_path.read_text(encoding="utf-8"))


def make_e5_bar_plot(unet_res: dict, pix2pix_res: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["U-Net (L1)", "Pix2Pix (L1+GAN)"]

    # 这里选用：U-Net = strong final；Pix2Pix = best L1
    l1_vals = [unet_res["final_mae"], pix2pix_res["best_val_l1"]]
    psnr_vals = [unet_res["final_psnr"], pix2pix_res["best_psnr"]]
    ssim_vals = [unet_res["final_ssim"], pix2pix_res["best_ssim"]]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    ax[0].bar(x, l1_vals, width, color=["C0", "C1"])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=20)
    ax[0].set_ylabel("Val L1 / MAE")
    ax[0].set_title("Val L1 (↓ 更好)")

    ax[1].bar(x, psnr_vals, width, color=["C0", "C1"])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=20)
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("Val PSNR (↑ 更好)")

    ax[2].bar(x, ssim_vals, width, color=["C0", "C1"])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=20)
    ax[2].set_ylabel("SSIM")
    ax[2].set_title("Val SSIM (↑ 更好)")

    fig.tight_layout()
    out_path = out_dir / "e5_l1_vs_l1_gan_bar.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_e5_samples(out_root: Path):
    """收集 E5 对比用的样例图：U-Net vs Pix2Pix。"""
    import shutil

    unet_dir = PROJECT_ROOT / "outputs" / "images" / "unet_baseline"
    pix2pix_dir = PROJECT_ROOT / "outputs" / "images" / "pix2pix_l1_gan_strong_e50"

    dst_dir = out_root / "e5_l1_vs_l1_gan"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 简单策略：挑前两个样本，U-Net 用 strong_e040，对应 Pix2Pix epoch_050_sample_xx
    mapping = [
        ("sample_00_strong_e040.png", "epoch_050_sample_00.png"),
        ("sample_01_strong_e040.png", "epoch_050_sample_01.png"),
    ]

    for unet_name, pix_name in mapping:
        src_unet = unet_dir / unet_name
        src_pix = pix2pix_dir / pix_name
        if src_unet.exists():
            shutil.copy2(src_unet, dst_dir / f"unet_{unet_name}")
        if src_pix.exists():
            shutil.copy2(src_pix, dst_dir / f"pix2pix_{pix_name}")

    print(f"[OK] Copied E5 sample pairs to: {dst_dir}")


def main():
    unet_csv = PROJECT_ROOT / "outputs" / "logs" / "unet_baseline" / "ablation_results_first40.csv"
    pix_md = PROJECT_ROOT / "outputs" / "figures" / "pix2pix_l1_gan_strong_e50" / "e2_pix2pix_main_results_e50.md"

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "e5_l1_vs_l1_gan"
    out_images_root = PROJECT_ROOT / "outputs" / "images"

    unet_res = load_unet_strong_results(unet_csv)
    pix2pix_res = load_pix2pix_results(pix_md)

    make_e5_comparison_table(unet_res, pix2pix_res, out_dir)
    make_e5_bar_plot(unet_res, pix2pix_res, out_dir)
    collect_e5_samples(out_images_root)

    print("[OK] E5 (L1 vs L1+GAN) comparison artifacts generated.")


if __name__ == "__main__":
    main()


