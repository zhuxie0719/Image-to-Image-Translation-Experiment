# 数据增强消融实验完整指南

## 📋 实验目标

对比三种数据增强配置对U-Net基线模型性能的影响：
1. **none**: 无增强（仅resize到256x256）
2. **basic**: 基础增强（随机jitter + 水平翻转）
3. **strong**: 强增强（jitter + 翻转 + 颜色抖动 + 随机缩放）

## 🚀 实验步骤

### 第一步：完成无增强训练

1. 打开 `01_unet_baseline.ipynb`
2. 找到 **Cell 13**（训练配置部分）
3. 确保 `aug_mode = "none"`
4. 运行 Cell 13，完成200个epoch的训练
5. 训练完成后会保存：
   - `model_none_final.pth` - 最终模型
   - `history_none.json` - 训练历史（损失、PSNR、SSIM、MAE）
   - `checkpoint_none_epoch_*.pth` - 每10个epoch的checkpoint
   - `sample_*_none.png` - 验证集样例三联图

### 第二步：完成基础增强训练

1. 在 **Cell 13** 中修改：`aug_mode = "basic"`
2. **重新运行 Cell 13**（这会创建新的模型和数据集）
3. 等待训练完成（200 epochs）
4. 训练完成后会保存：
   - `model_basic_final.pth`
   - `history_basic.json`
   - `sample_*_basic.png`

### 第三步：完成强增强训练

1. 在 **Cell 13** 中修改：`aug_mode = "strong"`
2. **重新运行 Cell 13**
3. 等待训练完成（200 epochs）
4. 训练完成后会保存：
   - `model_strong_final.pth`
   - `history_strong.json`
   - `sample_*_strong.png`

### 第四步：对比分析

完成所有三种配置的训练后，运行以下Cells进行对比分析：

1. **Cell 20**: 加载所有训练历史
2. **Cell 22**: 绘制损失曲线和指标曲线对比图
3. **Cell 24**: 生成最终指标对比表和柱状图
4. **Cell 26**: 可视化生成结果对比
5. **Cell 28**: 生成消融实验分析报告

## 📊 输出文件说明

训练完成后，`outputs/unet_baseline/` 目录下会包含：

### 模型文件
- `model_{aug_mode}_final.pth` - 最终模型权重
- `checkpoint_{aug_mode}_epoch_{epoch}.pth` - 中间checkpoint

### 训练历史
- `history_{aug_mode}.json` - 包含每个epoch的：
  - `train_losses`: 训练损失
  - `val_losses`: 验证损失
  - `val_psnrs`: PSNR指标
  - `val_ssims`: SSIM指标
  - `val_maes`: MAE指标

### 可视化文件
- `training_curve_{aug_mode}.png` - 单个配置的损失曲线
- `samples_{aug_mode}.png` - 单个配置的生成样例
- `ablation_comparison_curves.png` - 三种配置的损失和指标曲线对比
- `ablation_metrics_comparison.png` - 三种配置的指标柱状图对比
- `ablation_samples_comparison.png` - 三种配置的生成结果对比

### 分析报告
- `ablation_results.csv` - 最终指标对比表（CSV格式）
- `ablation_report.txt` - 消融实验分析报告（文本格式）

## 📈 评估指标

实验会计算以下指标：

- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，单位dB，**越高越好**
- **SSIM (Structural Similarity Index)**: 结构相似性，范围0-1，**越高越好**
- **MAE (Mean Absolute Error)**: 平均绝对误差，**越低越好**
- **L1 Loss**: L1损失，**越低越好**

## ⚠️ 注意事项

1. **训练时间**: 每种配置需要训练200个epoch，在GPU上大约需要2-3小时
2. **存储空间**: 确保有足够的存储空间保存模型和训练历史
3. **顺序执行**: 必须按顺序完成三种配置的训练，才能进行对比分析
4. **检查点**: 每10个epoch会保存checkpoint，可以用于恢复训练
5. **样例数量**: 默认保存10张验证集样例，可在训练代码中修改 `num_samples` 参数

## 🔍 结果分析建议

完成消融实验后，可以从以下角度分析：

1. **指标对比**: 哪种增强配置的PSNR/SSIM最高？MAE最低？
2. **训练稳定性**: 哪种配置的训练损失更稳定？
3. **过拟合分析**: 训练损失和验证损失的差距如何？
4. **视觉质量**: 主观评价生成结果的质量
5. **结论**: 数据增强是否提升了模型性能？哪种增强策略最有效？

## 📝 报告撰写

使用生成的 `ablation_report.txt` 和可视化图表，撰写实验报告时应包括：

1. 实验设置说明
2. 三种配置的详细对比
3. 指标分析和可视化
4. 生成结果的主观评价
5. 结论和建议


