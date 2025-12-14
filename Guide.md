---
title: 图像到图像生成实验执行指南（Cityscapes街景）
author: 课程小组
date: 2024-12-14
---

# 图像到图像生成实验执行指南

> 适用对象：首次接触图像生成与GAN的两人小组  
> 作业截止：2024 年 12 月 28 日  
> 参考文献：Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR 2017.  
> 官方代码：https://github.com/phillipi/pix2pix

---

## 1. 项目背景与目标

- **核心任务**：基于 Cityscapes 数据集完成从语义标签图到真实街景照片的图像生成任务（Label → Photo），覆盖数据准备、模型实现、训练调优、评估分析与报告撰写的端到端流程。  
- **数据集特点**：  
  - Cityscapes 格式拼接图像：每张图片左侧为真实街景（photo），右侧为语义标签（label）。  
  - 数据集下载地址：https://gitcode.com/open-source-toolkit/04615  
  - 需要将拼接图像分割为独立的 label 和 photo 图像对。  
  - 任务方向：从语义标签图生成对应的真实街景照片。  
- **考核重点**：实验完整性（代码可运行）、结果分析与指标对比、报告质量、创新尝试（新损失/模型/扩散方法）、生成效果质量。  
- **评估指标**：PSNR（峰值信噪比）、SSIM（结构相似性）、MAE（平均绝对误差）、FID（感知质量）、主观质量评价。

---

## 2. 角色分工（两人小组）

| 角色 | 负责人 | 主要职责 |
| --- | --- | --- |
| **Pix2Pix与损失函数负责人（成员 A）** | 成员 A | 数据预处理主导（图像分割、数据加载器）、Pix2Pix模型实现（U-Net生成器、PatchGAN判别器）、损失函数实现与消融实验（L1、Adversarial、Feature Matching、Perceptual Loss）、评估指标实现（PSNR、SSIM、MAE、FID）、训练循环与超参数调优、报告撰写对应章节 |
| **模型对比与可视化负责人（成员 B）** | 成员 B | 数据增强策略实现、U-Net基线模型（仅L1损失）、CycleGAN实现（必做）、训练过程可视化（损失曲线、三联图生成）、结果可视化与分析（生成样例对比、指标对比图表）、失败案例分析、报告撰写对应章节 |
| **公共协作** | 双方 | 数据集维护、日志同步、实验记录、结果汇总表、报告整体排版、提交物整理与审核 |

---

## 3. 时间规划（两周14天安排）

| 时间段 | 关键里程碑 | 负责人 | 优先级 |
| --- | --- | --- | --- |
| **第 1 天** | 环境搭建、数据集下载与检查、Git项目结构初始化、图像分割脚本编写 | 双方 | ⭐⭐⭐ |
| **第 2 天** | 数据探索（EDA）、确认数据集划分（train/val，无需test）、数据加载器实现、样例可视化 | 成员 A 主导，成员 B 协助 | ⭐⭐⭐ |
| **第 3 天** | U-Net基线模型实现（仅L1损失，无GAN），完成首轮训练与评估 | 成员 B | ⭐⭐⭐ |
| **第 4-5 天** | Pix2Pix完整实现（U-Net生成器+PatchGAN判别器），L1+GAN损失训练 | 成员 A | ⭐⭐⭐ |
| **第 6 天** | 损失函数消融实验：L1 vs L1+GAN vs L1+GAN+Feature Matching | 成员 A | ⭐⭐⭐ |
| **第 7 天** | 损失函数扩展：Perceptual Loss实现与对比实验 | 成员 A | ⭐⭐ |
| **第 8 天** | 数据增强策略对比（随机翻转、随机裁剪、颜色抖动等） | 成员 B | ⭐⭐ |
| **第 9 天** | CycleGAN实现与训练（两个生成器+两个判别器，循环一致性损失） | 成员 B | ⭐⭐⭐ |
| **第 10 天** | 评估指标完整实现（PSNR、SSIM、MAE、FID），验证集三联图批量生成 | 双方 | ⭐⭐⭐ |
| **第 11 天** | 失败案例分析、生成质量对比可视化、训练曲线整理 | 成员 B 主导，成员 A 协助 | ⭐⭐⭐ |
| **第 12 天** | 报告撰写初稿（结构 + 所有图表 + 结果汇总表） | 双方 | ⭐⭐⭐ |
| **第 13 天** | 报告润色、引用检查、提交物整理（代码仓库、README、三联图） | 双方 | ⭐⭐⭐ |
| **第 14 天（截止日）** | 最终检查，发送邮件提交 | 双方 | ⭐⭐⭐ |

> **时间紧张时的压缩策略**：若进度滞后，优先保证：U-Net基线、Pix2Pix完整实现、CycleGAN实现（可减少训练轮数）、至少两组损失函数对比（L1 vs L1+GAN）、评估指标实现、三联图生成、报告初稿。

---

## 4. 项目目录与版本管理建议

```
Image-to-Image-Translation-Experiment/
├── data/
│   ├── raw/                   # 原始拼接图像（保持只读）
│   ├── processed/             # 分割后的label和photo图像对
│   └── splits/                # 训练/验证划分索引（无需测试集）
├── src/
│   ├── data/
│   │   ├── dataset.py         # 数据集类定义
│   │   ├── transforms.py      # 数据增强
│   │   └── split_data.py      # 数据划分脚本（可选，数据集已预划分）
│   ├── models/
│   │   ├── generator.py       # U-Net生成器（Pix2Pix）
│   │   ├── discriminator.py   # PatchGAN判别器（Pix2Pix）
│   │   ├── unet_baseline.py   # U-Net基线（仅L1）
│   │   ├── cyclegan_generator.py  # CycleGAN生成器（ResNet-based）
│   │   └── cyclegan_discriminator.py  # CycleGAN判别器
│   ├── losses/
│   │   ├── adversarial_loss.py
│   │   ├── perceptual_loss.py
│   │   └── feature_matching.py
│   ├── training/
│   │   ├── train_pix2pix.py   # Pix2Pix训练脚本
│   │   ├── train_unet.py      # U-Net基线训练
│   │   ├── train_cyclegan.py  # CycleGAN训练脚本
│   │   └── trainer.py         # 训练器基类
│   └── eval/
│       ├── metrics.py         # PSNR、SSIM、MAE、FID
│       └── visualize.py      # 三联图生成、结果可视化
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_unet_baseline.ipynb
│   ├── 02_pix2pix_training.ipynb
│   └── 03_cyclegan_training.ipynb
├── outputs/
│   ├── logs/                  # 训练日志
│   ├── checkpoints/           # 模型权重
│   ├── images/                # 每个epoch的三联图
│   └── figures/               # 训练曲线、对比图
├── report/
│   ├── draft.md (或 LaTeX)
│   └── figures/               # 报告用图表
├── requirements.txt
├── README.md
├── Guide.md (本文件)
└── 作业要求.md
```

- 使用 Git 版本控制，关键阶段打 tag（如 `v0.1-baseline`, `v0.2-pix2pix`, `v1.0-final`）。  
- 每次实验保留配置（YAML/JSON）、日志与指标（CSV/JSON），便于追踪与复现。

---

## 5. 环境与工具准备

### 5.1 共同准备
- **Python 版本**：建议 3.8+（作业要求 ≥ 3.8）。使用 `virtualenv` 或 Conda。  
- **核心库**：`numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`, `scikit-image`。  
- **深度学习栈**：`torch` (≥ 1.12), `torchvision`, `torchmetrics`（用于SSIM等指标）。  
- **评估指标库**：`pytorch-fid` 或 `pytorch-fid`（FID计算），`lpips`（可选，用于Perceptual Loss）。  
- **可选工具**：`wandb`/`tensorboard`（实验跟踪）、`hydra`（配置管理）、`rich`（日志输出）。

### 5.2 数据集下载与处理
1. **下载数据集**：  
   - 数据集地址：https://gitcode.com/open-source-toolkit/04615  
   - 下载后校验：确认图像格式正确，左右拼接结构一致。
2. **图像分割脚本**（成员 A 负责实现）：  
   - 读取拼接图像（左侧photo，右侧label）。  
   - 分割为独立的 `photo` 和 `label` 图像对。  
   - 保存到 `data/processed/`，建议命名：`{image_id}_photo.jpg` 和 `{image_id}_label.png`。  
   - 注意：label图像通常为单通道或RGB彩色标签图，需确认格式。
3. **数据划分**：  
   - **推荐方案**：直接使用现有的train/val划分（训练集2975张，验证集500张），无需划分测试集。  
   - **原因**：图像生成任务主要关注生成质量，验证集已足够用于模型选择、超参数调优和最终评估。  
   - **实现方式**：运行 `src/data/split_data.py` 脚本，生成划分索引文件 `data/splits/cityscapes_split_seed42.json`，包含train和val目录中的所有图像文件名列表，以保证实验一致。  
   - **索引文件格式**：`{"train": ["1.jpg", "2.jpg", ...], "val": ["1.jpg", "2.jpg", ...]}`（无需test字段）  
   - 详细说明请参考 `data/splits/DATA_SPLIT_INFO.md`。

### 5.3 计算资源
- 若本地 GPU 不足，优先使用 Google Colab Pro / Kaggle Notebook / 阿里云PAI。  
- 建议在 Colab 中维护训练脚本，定期下载checkpoints到本地。  
- 图像尺寸建议从 256×256 开始，若显存充足可尝试 512×512。  
- Batch size：根据显存调整，Pix2Pix通常 batch size=1 或 4。

---

## 6. 数据探索与预处理流程

### 6.1 数据探索（EDA，共享 notebook `00_data_exploration.ipynb`）
1. **数据集统计**：  
   - 总图像数量、图像尺寸分布。  
   - 语义标签类别数量与分布（Cityscapes通常有19类或34类）。  
   - 可视化随机样本：展示原始拼接图、分割后的label和photo。
2. **数据质量检查**：  
   - 检查是否有损坏图像。  
   - 观察label图像的颜色映射（是否为RGB彩色标签或单通道索引图）。  
   - 记录数据特点：光照变化、场景多样性、标签精度。

### 6.2 图像预处理
1. **输入归一化**：  
   - Label图像：若为RGB彩色标签，需转换为类别索引或保持RGB；若为单通道，需确认类别映射。  
   - Photo图像：归一化到 `[-1, 1]`（使用Tanh输出时）或 `[0, 1]`。  
   - 统一尺寸：256×256（论文标准）或根据显存调整。
2. **数据增强策略**（成员 B 负责实现与消融）：  
   - **基础增强**：随机水平翻转（label和photo同步翻转）、随机裁剪。  
   - **进阶增强**：随机亮度/对比度调整（仅对photo，label保持不变）、随机缩放。  
   - **注意**：label和photo必须同步应用空间变换，但颜色变换只应用于photo。

---

## 7. Pix2Pix模型实现路线（成员 A）

### 7.1 模型架构（参考论文附录6.1）

#### 7.1.1 U-Net生成器
- **编码器**：C64-C128-C256-C512-C512-C512-C512-C512  
  - Ck表示：Convolution(4×4, stride=2) + BatchNorm + LeakyReLU(0.2)  
  - 第一层C64不使用BatchNorm  
- **解码器**：CD512-CD512-CD512-C512-C256-C128-C64  
  - CDk表示：Convolution + BatchNorm + Dropout(0.5) + ReLU  
  - 最后层：Convolution映射到3通道 + Tanh  
- **Skip Connections**：编码器第i层连接到解码器第n-i层（U-Net结构）

#### 7.1.2 PatchGAN判别器
- **70×70 PatchGAN**（论文推荐）：C64-C128-C256-C512  
  - 所有层：Convolution(4×4, stride=2) + BatchNorm + LeakyReLU(0.2)  
  - 第一层C64不使用BatchNorm  
  - 最后层：Convolution映射到1维输出 + Sigmoid  
  - 输入：concatenate(label, image)，输出：70×70的patch判别结果

### 7.2 损失函数实现

#### 7.2.1 基础损失
1. **L1损失**（像素级重建损失）：  
   ```python
   L_L1 = ||G(x) - y||_1
   ```
2. **对抗损失**（GAN损失）：  
   ```python
   L_GAN = log(D(x, G(x)))  # 生成器
   L_D = -[log(D(x, y)) + log(1 - D(x, G(x)))]  # 判别器
   ```
3. **组合损失**：  
   ```python
   L_total = L_GAN + λ * L_L1  # λ通常为100
   ```

#### 7.2.2 扩展损失（消融实验）
4. **Feature Matching Loss**：  
   - 使用判别器中间层特征，计算真实图像和生成图像的特征距离。
5. **Perceptual Loss**：  
   - 使用预训练VGG网络提取特征，计算特征空间距离（需安装`lpips`或使用`torchvision.models.vgg`）。

### 7.3 训练策略
- **优化器**：Adam，生成器lr=2e-4，判别器lr=2e-4（论文设置）。  
- **学习率调度**：线性衰减（从epoch 100开始衰减到0）。  
- **训练轮数**：200 epochs（论文设置），可根据收敛情况调整。  
- **Batch size**：1（论文设置，避免BatchNorm在batch=1时的问题，或移除bottleneck层的BatchNorm）。  
- **输入尺寸**：256×256。  
- **数据增强**：随机jitter（resize到286×286再随机裁剪回256×256）+ 随机水平翻转。

### 7.4 实现检查清单
- [ ] U-Net生成器前向传播测试（输入输出尺寸正确）  
- [ ] PatchGAN判别器前向传播测试  
- [ ] 损失函数计算正确（L1、GAN）  
- [ ] 训练循环：生成器和判别器交替训练  
- [ ] 每个epoch保存验证集三联图（Label / Generated / Ground Truth）  
- [ ] 记录训练损失曲线

---

## 8. U-Net基线模型路线（成员 B）

### 8.1 基线模型
- **架构**：与Pix2Pix生成器相同的U-Net结构。  
- **损失函数**：仅L1损失（无GAN）。  
- **训练设置**：与Pix2Pix相同的优化器和学习率。  
- **目的**：作为对比基线，展示GAN损失对生成质量的影响。

### 8.2 数据增强消融实验
对比以下策略对生成质量的影响：
1. **无增强**：仅resize到256×256。  
2. **基础增强**：随机水平翻转 + 随机jitter。  
3. **强增强**：基础增强 + 颜色抖动（仅photo）。

### 8.3 可视化任务
1. **训练过程可视化**：  
   - 每个epoch生成验证集三联图（Label / Generated / Ground Truth）。  
   - 绘制训练损失曲线（生成器损失、判别器损失、L1损失）。  
   - 记录评估指标曲线（PSNR、SSIM、MAE、FID）。
2. **结果对比可视化**：  
   - 不同损失函数组合的生成结果对比图。  
   - 不同epoch的生成质量演变。  
   - 失败案例分析（生成模糊、颜色失真、结构错误等）。

---

## 9. CycleGAN模型实现路线（成员 B）

### 9.1 CycleGAN简介
- **核心思想**：无需配对数据，通过循环一致性损失实现无监督的图像到图像转换。  
- **与Pix2Pix的区别**：Pix2Pix需要配对的(label, photo)数据，CycleGAN只需要两个域的图像集合。  
- **在本任务中的应用**：虽然我们有配对数据，但CycleGAN可以作为对比模型，展示不同训练范式（配对 vs 非配对）的效果。

### 9.2 模型架构

#### 9.2.1 CycleGAN生成器
- **架构**：ResNet-based生成器（而非U-Net）  
  - 编码器：3个下采样卷积层  
  - 残差块：6-9个ResNet块（论文推荐9个）  
  - 解码器：3个上采样转置卷积层  
- **激活函数**：使用Instance Normalization替代Batch Normalization（更适合风格转换任务）  
- **输出**：Tanh激活，范围[-1, 1]

#### 9.2.2 CycleGAN判别器
- **架构**：70×70 PatchGAN（与Pix2Pix相同）  
- **输入**：仅图像（不需要concatenate label），因为CycleGAN是无监督的  
- **输出**：70×70的patch判别结果

### 9.3 损失函数

#### 9.3.1 对抗损失（GAN Loss）
- **生成器G（Label→Photo）**：  
  ```python
  L_GAN_G = log(D_photo(G(label)))
  ```
- **生成器F（Photo→Label）**：  
  ```python
  L_GAN_F = log(D_label(F(photo)))
  ```
- **判别器D_photo**：区分真实photo和生成的photo  
- **判别器D_label**：区分真实label和生成的label

#### 9.3.2 循环一致性损失（Cycle Consistency Loss）
- **前向循环**：label → G(label) → F(G(label)) ≈ label  
  ```python
  L_cycle_forward = ||F(G(label)) - label||_1
  ```
- **反向循环**：photo → F(photo) → G(F(photo)) ≈ photo  
  ```python
  L_cycle_backward = ||G(F(photo)) - photo||_1
  ```
- **总循环损失**：  
  ```python
  L_cycle = L_cycle_forward + L_cycle_backward
  ```

#### 9.3.3 身份损失（Identity Loss，可选）
- 帮助生成器保持颜色一致性：  
  ```python
  L_identity = ||G(photo) - photo||_1 + ||F(label) - label||_1
  ```

#### 9.3.4 总损失
```python
L_total_G = L_GAN_G + λ_cycle * L_cycle + λ_identity * L_identity
L_total_F = L_GAN_F + λ_cycle * L_cycle + λ_identity * L_identity
```
- 通常：λ_cycle = 10, λ_identity = 0.5（可选）

### 9.4 训练策略
- **优化器**：Adam，lr=2e-4（前100 epochs），然后线性衰减到0（后100 epochs）  
- **训练轮数**：200 epochs（与Pix2Pix相同）  
- **Batch size**：1（论文设置）  
- **输入尺寸**：256×256  
- **数据增强**：随机jitter + 随机水平翻转（对两个域分别应用）

### 9.5 实现检查清单
- [ ] ResNet-based生成器前向传播测试（G和F两个生成器）  
- [ ] 两个判别器前向传播测试（D_photo和D_label）  
- [ ] 循环一致性损失计算正确  
- [ ] 训练循环：四个网络（G, F, D_photo, D_label）交替训练  
- [ ] 每个epoch保存验证集结果（Label→Photo和Photo→Label）  
- [ ] 记录训练损失曲线（GAN损失、循环损失）

### 9.6 与Pix2Pix的对比分析
- **优势**：不需要严格的配对数据，更灵活  
- **劣势**：在配对数据场景下，可能不如Pix2Pix精确  
- **实验对比重点**：在报告中对Pix2Pix和CycleGAN进行定量（指标）和定性（视觉质量）对比

---

## 10. 评估指标实现（双方协作）

### 9.1 指标定义与实现

| 指标 | 公式/说明 | 实现库 | 负责人 |
| --- | --- | --- | --- |
| **PSNR** | `PSNR = 10 * log10(MAX^2 / MSE)`，MAX通常为1.0或255 | `skimage.metrics.peak_signal_noise_ratio` 或自实现 | 成员 A |
| **SSIM** | 结构相似性指数，考虑亮度、对比度、结构 | `torchmetrics.functional.ssim` 或 `skimage.metrics.structural_similarity` | 成员 A |
| **MAE** | `MAE = mean(|pred - gt|)` | `torch.nn.L1Loss` 或 `numpy.mean(np.abs(pred - gt))` | 成员 A |
| **FID** | Fréchet Inception Distance，使用Inception网络特征 | `pytorch-fid` 库或自实现 | 成员 B |

### 10.2 评估流程
1. **每个epoch评估**：在验证集上计算所有指标，记录到CSV。  
2. **最终评估**：在验证集上计算所有指标，用于报告（图像生成任务无需单独的测试集）。  
3. **批量生成三联图**：保存每个epoch的验证集样例（至少10-20张），用于报告展示。

---

## 11. 消融实验与结果整合

### 11.1 建议至少完成以下实验组合

| 实验编号 | 描述 | 负责人 | 指标重点 | 优先级 |
| --- | --- | --- | --- | --- |
| **E1** | U-Net基线（仅L1损失） | 成员 B | PSNR、SSIM、MAE、主观质量 | ⭐⭐⭐ |
| **E2** | Pix2Pix（L1 + GAN） | 成员 A | 所有指标，与E1对比 | ⭐⭐⭐ |
| **E3** | CycleGAN（循环一致性损失） | 成员 B | 所有指标，与E2对比 | ⭐⭐⭐ |
| **E4** | Pix2Pix vs CycleGAN对比 | 双方 | 配对数据场景下的性能差异 | ⭐⭐⭐ |
| **E5** | L1 vs L1+GAN损失对比 | 成员 A | 生成质量、指标差异 | ⭐⭐⭐ |
| **E6** | L1+GAN vs L1+GAN+Feature Matching | 成员 A | 细节保真度提升 | ⭐⭐ |
| **E7** | Perceptual Loss尝试 | 成员 A | 感知质量提升 | ⭐⭐ |
| **E8** | 数据增强消融（无增强 vs 基础 vs 强增强） | 成员 B | 泛化能力、过拟合情况 | ⭐⭐ |
| **E9** | 超参数敏感性（学习率、λ权重） | 成员 A | 训练稳定性 | ⭐ |

### 11.2 结果汇总建议
- **制作统一表格**（放入报告）：  
  - 行：各实验编号  
  - 列：PSNR、SSIM、MAE、FID、训练时长、备注  
- **绘制图表**：  
  - 训练损失曲线对比（不同损失函数）  
  - 评估指标曲线（PSNR、SSIM随epoch变化）  
  - 不同实验的指标柱状图对比  
  - 生成样例三联图对比（至少8组：不同模型、不同损失、不同epoch）  
  - Pix2Pix vs CycleGAN对比图（相同输入下的生成结果）  
- **失败案例分析**：至少3-5组失败样例（Label / Generated / Ground Truth + 分析文字），包含Pix2Pix和CycleGAN的失败案例对比。

---

## 12. 实验报告撰写框架（建议8-12页）

1. **摘要**：简述任务、方法（Pix2Pix）、主要结果（最佳指标值）、创新点。  
2. **引言**：图像到图像转换背景、Cityscapes数据集意义、Pix2Pix论文贡献（引用原文）。  
3. **相关工作**：概述条件GAN、图像生成相关研究（可列2-3篇参考文献，如CycleGAN、StyleGAN等）。重点对比Pix2Pix（配对数据）和CycleGAN（非配对数据）的区别。  
4. **数据集与预处理**：  
   - 数据源、Cityscapes格式说明  
   - 数据划分策略（使用现有的train/val划分，无需测试集）  
   - 图像预处理步骤（归一化、尺寸调整）  
   - 数据增强方法  
5. **方法**：  
   - **Pix2Pix模型架构**：U-Net生成器、PatchGAN判别器（可附架构图）  
   - **CycleGAN模型架构**：ResNet-based生成器、PatchGAN判别器、循环一致性机制（可附架构图）  
   - **损失函数**：L1损失、对抗损失、循环一致性损失、Feature Matching、Perceptual Loss（公式与说明）  
   - **训练策略**：优化器、学习率、训练轮数、数据增强  
   - **模型对比**：说明Pix2Pix和CycleGAN在配对数据场景下的适用性  
6. **实验设置**：  
   - 硬件环境（GPU型号、显存）  
   - 软件版本（Python、PyTorch等）  
   - 超参数设置（学习率、batch size、λ权重等）  
   - 评价指标定义  
7. **实验结果与分析**：  
   - **主结果表格**：所有实验的PSNR、SSIM、MAE、FID对比  
   - **消融实验分析**：  
     - 模型对比（U-Net基线 vs Pix2Pix vs CycleGAN）  
     - 损失函数消融（L1 vs L1+GAN vs L1+GAN+Feature Matching）  
     - 数据增强影响  
     - 超参数敏感性  
     - Pix2Pix与CycleGAN在配对数据场景下的性能差异分析  
   - **可视化结果**：  
     - 训练过程曲线（损失、指标）  
     - 生成样例三联图（不同实验、不同epoch）  
     - 失败案例分析  
8. **讨论**：  
   - 优势：Pix2Pix相比纯L1损失的优势；CycleGAN在无配对数据场景下的优势  
   - 局限：生成模糊、颜色失真等问题分析；Pix2Pix与CycleGAN各自的适用场景  
   - 模型对比总结：在配对数据场景下，Pix2Pix vs CycleGAN的性能差异及原因  
   - 未来改进方向：更高分辨率、更复杂损失、扩散模型尝试等  
9. **结论**：总结主要发现、最佳配置、贡献点。  
10. **参考文献**：遵循课程要求格式，务必包含Pix2Pix原始论文。

> **提示**：报告中所有图表须有编号与标题；正文中引用图表（如"见图2"），并说明观察到的现象与原因。三联图需清晰展示Label、Generated、Ground Truth的对比。

---

## 13. 提交物清单

- [ ] **实验报告**（PDF格式，8-12页）  
- [ ] **训练代码**（完整可运行，含README说明）  
- [ ] **验证集生成结果样例**（三联图，至少10-20张，展示不同epoch和不同实验）  
- [ ] **评估指标结果**（CSV/JSON，包含所有实验的PSNR、SSIM、MAE、FID）  
- [ ] **最佳模型权重**（checkpoint文件，可选）  
- [ ] **源代码仓库链接**（GitHub/Gitee，含README、运行说明、依赖列表）  
- [ ] **邮件发送**至 `guangyu.ryan@yahoo.com`，主题格式："图像到图像生成实验提交 - 第X小组 - 成员姓名"  
- [ ] **邮件正文**列出附件/链接、运行环境说明、最佳模型指标摘要

---

## 14. 风险与备选方案

- **算力不足**：  
  - 先在较小输入尺寸（128×128）上验证流程；  
  - 使用较小的batch size（1或2）；  
  - 减少训练轮数（先训练50-100 epochs验证流程，再完整训练200 epochs）；  
  - 在Colab分阶段训练并保存中间模型到Google Drive。  
- **训练不稳定**：  
  - GAN训练可能不稳定，尝试调整学习率（降低到1e-4）；  
  - 使用梯度裁剪（`torch.nn.utils.clip_grad_norm_`）；  
  - 调整λ权重（L1损失的权重，尝试50、100、200）。  
- **生成质量不佳**：  
  - 检查数据预处理（归一化是否正确）；  
  - 增加训练轮数；  
  - 尝试不同的数据增强策略；  
  - 检查模型架构实现是否正确（参考官方代码）。  
- **时间不够**：  
  - 确保至少完成：U-Net基线、Pix2Pix完整实现、CycleGAN实现（可减少训练轮数到100 epochs）、L1 vs L1+GAN对比、评估指标实现、三联图生成、报告初稿；  
  - 若无法完成所有消融实验，在报告中说明已完成部分及原因，展示过程思考。  
- **CycleGAN训练困难**：  
  - CycleGAN需要训练四个网络，训练时间较长；  
  - 若时间紧张，可先完成Pix2Pix，再实现CycleGAN的基础版本（减少残差块数量或训练轮数）；  
  - 循环一致性损失权重λ_cycle可能需要调整（尝试5、10、20）。  
- **FID计算困难**：  
  - 使用`pytorch-fid`库简化实现；  
  - 若仍困难，可先完成PSNR、SSIM、MAE，在报告中说明FID计算尝试与局限。

---

## 15. 参考资源

- **Pix2Pix原始论文**：Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.  
- **CycleGAN原始论文**：Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV 2017.  
- **官方代码仓库**：  
  - Pix2Pix：https://github.com/phillipi/pix2pix  
  - CycleGAN：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix（包含Pix2Pix和CycleGAN的PyTorch实现）  
- **PyTorch实现参考**：  
  - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix（官方PyTorch版本，推荐）  
  - https://github.com/mrzhu-cool/pix2pix-pytorch（简化Pix2Pix实现）  
  - https://github.com/eriklindernoren/PyTorch-GAN（包含多种GAN实现，含CycleGAN）  
- **评估指标实现**：  
  - FID：https://github.com/mseitzer/pytorch-fid  
  - SSIM：`torchmetrics` 或 `scikit-image`  
- **Cityscapes数据集**：https://www.cityscapes-dataset.com/  
- **U-Net原始论文**：Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

---

## 16. 执行建议速查表

- **每日保持10-15分钟站会式同步**，更新实验记录和进度。  
- **任何实验需先写下**："目的 → 配置 → 预期 → 实际结果 → 结论"。  
- **结果不理想时，优先检查**：数据加载（图像是否正确分割、归一化是否正确）、学习率（是否过大导致训练不稳定）、损失函数权重（λ是否合适）、模型架构（参考官方代码对比）。  
- **及时保存最优模型与指标**；每个epoch保存checkpoint，不要等到最后一天再整理。  
- **三联图生成脚本提前准备好**，确保每个epoch自动生成并保存。  
- **报告草稿至少提前3天完成**，预留时间校对与完善图表。  
- **提交前进行Checklist对照**，确保所有附件齐全（代码、报告、三联图、指标结果）。

---

## 17. 关键代码实现提示

### 16.1 数据加载器（成员 A）
```python
# 伪代码示例
class CityscapesDataset(Dataset):
    def __init__(self, label_dir, photo_dir, transform=None):
        self.label_dir = label_dir
        self.photo_dir = photo_dir
        self.transform = transform
        # 加载图像对列表
    
    def __getitem__(self, idx):
        label = Image.open(label_path).convert('RGB')  # 或根据实际格式调整
        photo = Image.open(photo_path).convert('RGB')
        if self.transform:
            label, photo = self.transform(label, photo)  # 同步变换
        return label, photo
```

### 16.2 U-Net生成器（成员 A）
- 参考论文附录6.1.1的架构描述
- 注意skip connections的实现（concatenate而非相加）
- 最后一层使用Tanh激活，输出范围[-1, 1]

### 16.3 PatchGAN判别器（成员 A）
- 输入：concatenate(label, image)，通道数为6（3+3）
- 输出：70×70的patch判别结果（不是单个标量）
- 使用BCEWithLogitsLoss或BCELoss

### 16.4 训练循环关键点（成员 A）
```python
# 伪代码示例
for epoch in range(num_epochs):
    for batch in dataloader:
        label, photo = batch
        
        # 训练判别器
        fake_photo = generator(label)
        d_real = discriminator(label, photo)
        d_fake = discriminator(label, fake_photo.detach())
        d_loss = adversarial_loss(d_real, d_fake)
        d_loss.backward()
        optimizer_d.step()
        
        # 训练生成器
        fake_photo = generator(label)
        d_fake = discriminator(label, fake_photo)
        g_gan_loss = adversarial_loss_g(d_fake)
        g_l1_loss = l1_loss(fake_photo, photo)
        g_loss = g_gan_loss + lambda_l1 * g_l1_loss
        g_loss.backward()
        optimizer_g.step()
```

### 17.5 CycleGAN训练循环关键点（成员 B）
```python
# 伪代码示例
for epoch in range(num_epochs):
    for batch in dataloader:
        label, photo = batch
        
        # 训练判别器D_photo
        fake_photo = generator_G(label)
        d_photo_real = discriminator_photo(photo)
        d_photo_fake = discriminator_photo(fake_photo.detach())
        d_photo_loss = adversarial_loss(d_photo_real, d_photo_fake)
        d_photo_loss.backward()
        optimizer_d_photo.step()
        
        # 训练判别器D_label
        fake_label = generator_F(photo)
        d_label_real = discriminator_label(label)
        d_label_fake = discriminator_label(fake_label.detach())
        d_label_loss = adversarial_loss(d_label_real, d_label_fake)
        d_label_loss.backward()
        optimizer_d_label.step()
        
        # 训练生成器G和F
        fake_photo = generator_G(label)
        fake_label = generator_F(photo)
        
        # 循环一致性
        recovered_label = generator_F(fake_photo)
        recovered_photo = generator_G(fake_label)
        cycle_loss = l1_loss(recovered_label, label) + l1_loss(recovered_photo, photo)
        
        # 对抗损失
        g_gan_loss = adversarial_loss_g(discriminator_photo(fake_photo))
        f_gan_loss = adversarial_loss_g(discriminator_label(fake_label))
        
        # 总损失
        g_loss = g_gan_loss + lambda_cycle * cycle_loss
        f_loss = f_gan_loss + lambda_cycle * cycle_loss
        
        g_loss.backward()
        f_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
```

### 17.6 三联图生成（成员 B）
```python
# 伪代码示例
def save_triplet(label, generated, ground_truth, save_path):
    # 将三张图像水平拼接
    triplet = np.hstack([label, generated, ground_truth])
    Image.fromarray(triplet).save(save_path)
```

---

祝顺利完成作业！若遇到阻塞问题，优先记录在共享文档，并在每日同步时讨论解决方案。保持良好协作与实验记录，将大幅提升最终报告质量与得分。记住：**实验完整性、结果对比分析、报告质量是获得高分的关键**。

