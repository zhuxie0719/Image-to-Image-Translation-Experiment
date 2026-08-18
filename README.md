# 城市街景图像到图像生成

基于 Cityscapes 街景数据，完成语义标签图到真实照片的图像翻译（Label → Photo）。实现 U-Net 基线、Pix2Pix 与 CycleGAN，对比损失函数与数据增强对生成质量的影响，并用 PSNR / SSIM / MAE / FID 评估。

参考论文：Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*（CVPR 2017）。

## 任务设定

Cityscapes 拼接图左侧为真实街景（photo），右侧为语义标签（label）。实验将二者拆成图像对，学习：

```text
语义标签图  →  真实街景照片
```

每个 epoch 在验证集上保存三联图：Label / Generated / Ground Truth。

## 模型

| 模型 | 结构 | 损失 |
| --- | --- | --- |
| U-Net 基线 | 带 skip 的编码器-解码器 | 仅 L1 |
| Pix2Pix | U-Net 生成器 + 70×70 PatchGAN | L1 + 对抗损失，可加 Feature Matching / Perceptual Loss |
| CycleGAN | 双向 ResNet 生成器 + 双判别器 | 循环一致性损失，无需严格成对监督 |

数据增强消融（U-Net）：

- `none`：仅 resize 到 256×256
- `basic`：随机 jitter + 水平翻转
- `strong`：jitter + 翻转 + 颜色抖动 + 随机缩放

## 评价指标

| 指标 | 含义 |
| --- | --- |
| PSNR | 峰值信噪比，越高越好 |
| SSIM | 结构相似性，越高越好 |
| MAE | 像素平均绝对误差，越低越好 |
| FID | 生成分布与真实分布距离，越低越好 |

## 技术栈

- Python ≥ 3.8、PyTorch ≥ 1.12、CUDA
- torchvision、Pillow、scikit-image、matplotlib、tqdm
- 可选：pytorch-fid、lpips、Google Colab

## 项目结构

```text
Image-to-Image-Translation-Experiment/
├── data/
│   ├── raw/                   # 原始左右拼接图
│   ├── processed/             # 拆分后的 label / photo
│   └── splits/                # train / val 索引
├── src/
│   ├── data/                  # Dataset 与增强
│   ├── models/
│   │   ├── generator.py       # Pix2Pix U-Net 生成器
│   │   ├── discriminator.py   # PatchGAN
│   │   ├── unet_baseline.py
│   │   ├── cyclegan_generator.py
│   │   └── cyclegan_discriminator.py
│   ├── losses/                # L1 / 对抗 / 感知 / Feature Matching
│   ├── training/              # 各模型训练脚本
│   └── eval/                  # 指标与三联图
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_unet_baseline.ipynb
│   ├── 03_cyclegan_training.ipynb
│   ├── 04_cyclegan_colab_training.ipynb
│   └── ABLATION_GUIDE.md
├── outputs/                   # 权重、曲线、样例
├── colab_setup.ipynb
└── Guide.md
```

官方划分规模：训练集约 2975 张，验证集约 500 张。图像生成任务以验证集做模型选择与最终评估。

## 快速开始

### 环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install torch torchvision numpy pillow tqdm scikit-image matplotlib
```

### 数据

1. 下载 Cityscapes 拼接图（课程提供的数据包）
2. 将左右图像拆成 `*_photo` 与 `*_label`，放入 `data/processed/`
3. 生成 `data/splits/cityscapes_split_seed42.json`

### 训练

推荐在 Jupyter / Colab 中运行：

1. `notebooks/00_data_exploration.ipynb`：检查尺寸、标签分布与样例
2. `notebooks/01_unet_baseline.ipynb`：L1 基线与增强消融（见 `notebooks/ABLATION_GUIDE.md`）
3. Pix2Pix：`src/models/generator.py` + `discriminator.py`，按 Guide 中的训练流程
4. `notebooks/03_cyclegan_training.ipynb` 或 `04_cyclegan_colab_training.ipynb`

Colab 环境初始化可使用仓库根目录的 `colab_setup.ipynb`。

建议输入分辨率从 256×256 开始，Pix2Pix 常用 `batch size = 1` 或 `4`。

## 文档

- [`Guide.md`](Guide.md)：数据分割、模型实现、损失消融与报告框架
- [`notebooks/ABLATION_GUIDE.md`](notebooks/ABLATION_GUIDE.md)：U-Net 三种增强配置的逐步实验
- [`作业要求.md`](作业要求.md)：课程任务与评分说明
