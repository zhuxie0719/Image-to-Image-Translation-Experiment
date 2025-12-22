| 模型 | 增强 | 选取策略 | Epoch | Val L1/MAE ↓ | PSNR ↑ | SSIM ↑ |
|------|------|----------|-------|--------------|--------|--------|
| U-Net | strong | Final (40 epoch) | 40 | 0.1014 | 17.089 | 0.509 |
| U-Net | strong | Best PSNR | 38 | 0.1014 | 17.323 | 0.512 |
| Pix2Pix | strong | Best Val L1 | 36 | 0.1109 | 16.275 | 0.435 |
| Pix2Pix | strong | Last (50 epoch) | 50 | 0.1122 | 16.150 | 0.449 |