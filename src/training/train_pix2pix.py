"""
Pix2Pix 训练脚本（L1 + GAN）。

对应 Guide.md:
- 7.1: U-Net 生成器 + PatchGAN 判别器
- 7.2.1: 基础损失（L1 + GAN）
- 7.3: 训练策略（Adam, lr=2e-4, 200 epochs, batch size=1, 线性衰减）
- 7.4: 实现检查清单（前向测试、损失、交替训练、三联图、曲线）
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

# 保证以项目根目录为工作路径运行时，可以通过绝对包名导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.data.dataset import CityscapesDataset
from src.data.transforms import build_transform
from src.losses.pix2pix_losses import (
    pix2pix_generator_loss,
    pix2pix_discriminator_loss,
)


def save_triplet(label, generated, ground_truth, save_path: Path):
    """保存三联图：Label / Generated / Ground Truth"""

    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()

        # 如果是归一化到[-1, 1]的，转回[0, 1]
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        return img

    label_img = tensor_to_image(label)
    gen_img = tensor_to_image(generated)
    gt_img = tensor_to_image(ground_truth)

    triplet = np.hstack([label_img, gen_img, gt_img])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(triplet).save(save_path)


def get_lr_scheduler(optimizer, num_epochs: int, start_decay_epoch: int = 100):
    """线性学习率衰减（从 start_decay_epoch 开始衰减到 0）"""

    def lr_lambda(epoch: int):
        if epoch < start_decay_epoch:
            return 1.0
        return 1.0 - (epoch - start_decay_epoch) / (num_epochs - start_decay_epoch)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # 环境里没有 tensorboard 时，提供一个空壳
    class SummaryWriter:  # type: ignore
        def __init__(self, *_, **__):
            pass

        def add_scalar(self, *_, **__):
            pass

        def close(self):
            pass


def train_epoch(
    generator: UNetGenerator,
    discriminator: PatchGANDiscriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    device,
    epoch: int,
    lambda_l1: float = 100.0,
    writer: SummaryWriter | None = None,
):
    """训练一个 epoch：交替更新 D 和 G。"""
    generator.train()
    discriminator.train()

    total_g_total = 0.0
    total_g_gan = 0.0
    total_g_l1 = 0.0
    total_d = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train Pix2Pix]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)

        # -------------------------
        # 1. 更新判别器 D
        # -------------------------
        for p in generator.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        with torch.no_grad():
            fake_photo_detached = generator(label)

        d_real = discriminator(label, photo)
        d_fake = discriminator(label, fake_photo_detached)

        d_loss = pix2pix_discriminator_loss(d_real, d_fake)
        d_loss.backward()
        optimizer_D.step()

        # -------------------------
        # 2. 更新生成器 G
        # -------------------------
        for p in generator.parameters():
            p.requires_grad = True
        for p in discriminator.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()

        fake_photo = generator(label)
        d_fake_for_G = discriminator(label, fake_photo)

        g_total, g_gan, g_l1 = pix2pix_generator_loss(
            fake_photo,
            photo,
            d_fake_for_G,
            lambda_l1=lambda_l1,
        )
        g_total.backward()
        optimizer_G.step()

        # 统计
        total_g_total += g_total.item()
        total_g_gan += g_gan.item()
        total_g_l1 += g_l1.item()
        total_d += d_loss.item()
        num_batches += 1

        pbar.set_postfix(
            {
                "G_total": f"{g_total.item():.3f}",
                "G_L1": f"{g_l1.item():.3f}",
                "D": f"{d_loss.item():.3f}",
            }
        )

        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/G_total", g_total.item(), global_step)
            writer.add_scalar("Train/G_L1", g_l1.item(), global_step)
            writer.add_scalar("Train/G_GAN", g_gan.item(), global_step)
            writer.add_scalar("Train/D_loss", d_loss.item(), global_step)

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_g_total = total_g_total / num_batches
    avg_g_gan = total_g_gan / num_batches
    avg_g_l1 = total_g_l1 / num_batches
    avg_d = total_d / num_batches
    return avg_g_total, avg_g_gan, avg_g_l1, avg_d


@torch.no_grad()
def validate(
    generator: UNetGenerator,
    dataloader,
    device,
    epoch: int,
    exp_dir: Path,
    writer: SummaryWriter | None = None,
    num_samples: int = 10,
):
    """验证：仅前向 G，保存若干三联图（Label / Generated / Ground Truth）。"""
    from src.eval.metrics import evaluate_batch  # 延迟导入，避免循环依赖

    generator.eval()

    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    num_batches = 0

    saved_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val Pix2Pix]")
    for batch_idx, batch in enumerate(pbar):
        label = batch["label"].to(device)
        photo = batch["photo"].to(device)

        fake_photo = generator(label)

        # 评估指标（与 U-Net 基线保持一致接口）
        metrics = evaluate_batch(fake_photo, photo)
        l1_val = metrics["l1"]
        psnr_val = metrics["psnr"]
        ssim_val = metrics["ssim"]
        mae_val = metrics["mae"]

        total_l1 += l1_val
        total_psnr += psnr_val
        total_ssim += ssim_val
        total_mae += mae_val
        num_batches += 1

        # 保存三联图
        if saved_samples < num_samples:
            save_path = (
                exp_dir
                / "images"
                / f"epoch_{epoch:03d}_sample_{saved_samples:02d}.png"
            )
            save_triplet(label, fake_photo, photo, save_path)
            saved_samples += 1

        pbar.set_postfix(
            {
                "L1": f"{l1_val:.4f}",
                "PSNR": f"{psnr_val:.2f}",
                "SSIM": f"{ssim_val:.3f}",
            }
        )

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_l1 = total_l1 / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_mae = total_mae / num_batches

    if writer is not None:
        writer.add_scalar("Val/L1", avg_l1, epoch)
        writer.add_scalar("Val/PSNR", avg_psnr, epoch)
        writer.add_scalar("Val/SSIM", avg_ssim, epoch)
        writer.add_scalar("Val/MAE", avg_mae, epoch)

    return avg_l1, avg_psnr, avg_ssim, avg_mae


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix (L1 + GAN)")
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Data root directory"
    )
    parser.add_argument(
        "--split-index",
        type=Path,
        default=Path("data/splits/cityscapes_split_seed42.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="pix2pix_l1_gan_strong",
        help="Experiment name",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--start-decay-epoch",
        type=int,
        default=100,
        help="Start decay epoch",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=100.0,
        help="Weight for L1 loss in generator",
    )

    # 数据增强配置（E2 建议默认 strong，用于与最优 U-Net 基线对比）
    parser.add_argument(
        "--aug-mode",
        type=str,
        default="strong",
        choices=["none", "basic", "strong"],
        help="Augmentation mode: none/basic/strong",
    )

    # 其他参数
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=10,
        help="Number of validation samples to save",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # 创建输出目录
    exp_dir = args.output_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "images").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # 保存配置
    # 将 Path 转为 str，避免 json 序列化报错
    config = vars(args).copy()
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    config["timestamp"] = datetime.now().isoformat()
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 设备
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 数据增强配置（与 train_unet 保持一致）
    aug_configs = {
        "none": build_transform(
            image_size=256,
            jitter=False,
            horizontal_flip=False,
            color_jitter=None,
            scale_range=None,
            normalize_mode="tanh",
        ),
        "basic": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=None,
            scale_range=None,
            normalize_mode="tanh",
        ),
        "strong": build_transform(
            image_size=256,
            jitter=True,
            horizontal_flip=True,
            color_jitter=(0.2, 0.2, 0.2, 0.05),
            scale_range=(0.8, 1.2),
            normalize_mode="tanh",
        ),
    }

    train_transform = aug_configs[args.aug_mode]
    val_transform = aug_configs["none"]

    # 数据集
    print("Loading datasets...")
    train_dataset = CityscapesDataset(
        root=args.data_root,
        split="train",
        split_index=args.split_index,
        transform=train_transform,
    )
    val_dataset = CityscapesDataset(
        root=args.data_root,
        split="val",
        split_index=args.split_index,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 模型
    print("Creating Pix2Pix models (G: U-Net, D: PatchGAN)...")
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6).to(device)
    print(
        f"Generator params: {sum(p.numel() for p in generator.parameters()):,}; "
        f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}"
    )

    # 优化器 & 学习率调度
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler_G = get_lr_scheduler(optimizer_G, args.epochs, args.start_decay_epoch)
    scheduler_D = get_lr_scheduler(optimizer_D, args.epochs, args.start_decay_epoch)

    # TensorBoard
    writer = SummaryWriter(log_dir=exp_dir / "logs" / "tensorboard")

    best_val_l1 = float("inf")
    history = {
        "epoch": [],
        "train_g_total": [],
        "train_g_gan": [],
        "train_g_l1": [],
        "train_d": [],
        "val_l1": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_mae": [],
    }

    print(f"\nStarting Pix2Pix training (aug_mode={args.aug_mode})...")
    for epoch in range(1, args.epochs + 1):
        # 训练
        (
            train_g_total,
            train_g_gan,
            train_g_l1,
            train_d,
        ) = train_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_G,
            optimizer_D,
            device,
            epoch,
            lambda_l1=args.lambda_l1,
            writer=writer,
        )

        # 验证
        val_l1, val_psnr, val_ssim, val_mae = validate(
            generator,
            val_loader,
            device,
            epoch,
            exp_dir,
            writer,
            args.num_val_samples,
        )

        # 学习率调度
        scheduler_G.step()
        scheduler_D.step()
        current_lr = optimizer_G.param_groups[0]["lr"]
        writer.add_scalar("Train/LR", current_lr, epoch)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- G_total: {train_g_total:.4f}, G_L1: {train_g_l1:.4f}, "
            f"D_loss: {train_d:.4f}, Val L1: {val_l1:.4f}, PSNR: {val_psnr:.2f}, "
            f"SSIM: {val_ssim:.3f}, LR: {current_lr:.6f}"
        )

        # 记录 history
        history["epoch"].append(epoch)
        history["train_g_total"].append(train_g_total)
        history["train_g_gan"].append(train_g_gan)
        history["train_g_l1"].append(train_g_l1)
        history["train_d"].append(train_d)
        history["val_l1"].append(val_l1)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["val_mae"].append(val_mae)

        with open(exp_dir / "logs" / "history_pix2pix.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # 保存 checkpoint
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "scheduler_G_state_dict": scheduler_G.state_dict(),
                    "scheduler_D_state_dict": scheduler_D.state_dict(),
                    "train_g_total": train_g_total,
                    "train_g_l1": train_g_l1,
                    "train_d": train_d,
                    "val_l1": val_l1,
                },
                exp_dir / "checkpoints" / "best_model.pth",
            )

        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "scheduler_G_state_dict": scheduler_G.state_dict(),
                    "scheduler_D_state_dict": scheduler_D.state_dict(),
                    "train_g_total": train_g_total,
                    "train_g_l1": train_g_l1,
                    "train_d": train_d,
                    "val_l1": val_l1,
                },
                exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch:03d}.pth",
            )

    writer.close()
    print(f"\n✅ Pix2Pix training complete! Best val L1: {best_val_l1:.4f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()


