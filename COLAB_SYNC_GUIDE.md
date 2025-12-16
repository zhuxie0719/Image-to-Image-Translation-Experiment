# Colab 同步设置指南

本指南介绍如何实现本地文件和Google Drive/Colab的自动同步，避免每次手动上传文件。

## 🎯 三种同步方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **方案一：Google Drive for Desktop** | ✅ 最简单<br>✅ 自动同步<br>✅ 无需配置 | ⚠️ 需要安装客户端<br>⚠️ 占用本地空间 | **推荐：适合大多数用户** |
| **方案二：Git + GitHub** | ✅ 版本控制<br>✅ 代码自动同步<br>✅ 无需压缩 | ⚠️ 需要Git知识<br>⚠️ 数据文件需单独处理 | 适合有Git经验的用户 |
| **方案三：手动上传** | ✅ 无需安装<br>✅ 一次性设置 | ❌ 每次需重新上传<br>❌ 浪费时间 | 仅作为备选 |

---

## 📦 方案一：Google Drive for Desktop（推荐）

### 步骤1：安装Google Drive for Desktop

1. 访问：https://www.google.com/drive/download/
2. 下载并安装 **Google Drive for Desktop**
3. 登录你的Google账号

### 步骤2：设置同步文件夹

1. **将项目文件夹添加到Google Drive**：
   - 打开Google Drive网页版或桌面客户端
   - 将整个项目文件夹 `Image-to-Image-Translation-Experiment` 拖拽到Google Drive
   - 等待同步完成（首次同步可能需要一些时间）

2. **或者设置本地同步**：
   - 在Google Drive for Desktop中，选择"同步我的Drive"
   - 将项目文件夹放在 `Google Drive/MyDrive/` 目录下
   - 本地修改会自动同步到云端

### 步骤3：在Colab中使用

1. 打开 `01_unet_baseline.ipynb`
2. 在第二个代码cell中，设置：
   ```python
   SYNC_METHOD = 'drive_sync'
   ```
3. 运行代码，会自动检测并使用同步的文件

### 优点
- ✅ 本地修改后自动同步，无需手动操作
- ✅ 在Colab中直接使用最新文件
- ✅ 支持大文件（数据文件使用符号链接，不占用Colab空间）

---

## 🔄 方案二：Git + GitHub

### 步骤1：初始化Git仓库

```bash
cd Image-to-Image-Translation-Experiment
git init
git add src/ notebooks/ data/splits/ *.md
git commit -m "Initial commit"
```

### 步骤2：创建GitHub仓库并推送

1. 在GitHub上创建新仓库（例如：`Image-to-Image-Translation-Experiment`）
2. 推送代码：
   ```bash
   git remote add origin https://github.com/your-username/Image-to-Image-Translation-Experiment.git
   git branch -M main
   git push -u origin main
   ```

### 步骤3：处理大数据文件

**选项A：使用Git LFS（推荐）**
```bash
git lfs install
git lfs track "data/processed/**"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

**选项B：数据文件单独上传到Google Drive**
- 将 `data/processed/` 上传到Google Drive
- 代码会自动从Drive链接数据文件

### 步骤4：在Colab中使用

1. 在notebook中设置：
   ```python
   SYNC_METHOD = 'github'
   GITHUB_REPO = "your-username/Image-to-Image-Translation-Experiment"
   GITHUB_BRANCH = "main"
   ```
2. 每次更新代码后，在本地执行：
   ```bash
   git add .
   git commit -m "Update code"
   git push
   ```
3. 在Colab中运行代码，会自动拉取最新版本

### 优点
- ✅ 完整的版本控制历史
- ✅ 代码自动同步
- ✅ 可以回退到任意版本

---

## 📤 方案三：手动上传（备选）

如果前两种方案都不适合，可以继续使用手动上传：

1. 压缩项目文件：
   - `src/` → `src.zip`
   - `data/splits/` → `data_splits.zip`
   - `data/processed/` → `data_processed.zip`（如果太大，直接上传文件夹）

2. 上传到Google Drive

3. 在notebook中设置：
   ```python
   SYNC_METHOD = 'manual'
   ```

---

## 🚀 快速开始

### 推荐流程（方案一）

1. **安装Google Drive for Desktop**
2. **将项目文件夹拖到Google Drive**
3. **在notebook中设置 `SYNC_METHOD = 'drive_sync'`**
4. **开始使用！**

以后每次本地修改代码后：
- 保存文件
- Google Drive自动同步（通常几秒内完成）
- 在Colab中直接运行，使用最新文件

### 工作流程示例

```
本地修改代码
    ↓
保存文件（Ctrl+S）
    ↓
Google Drive自动同步（几秒钟）
    ↓
在Colab中运行notebook
    ↓
自动使用最新代码 ✅
```

---

## ⚙️ 配置说明

在notebook的第二个代码cell中，你可以修改以下配置：

```python
# 选择同步方案
SYNC_METHOD = 'drive_sync'  # 'github', 'drive_sync', 或 'manual'

# GitHub配置（如果使用方案二）
GITHUB_REPO = "your-username/Image-to-Image-Translation-Experiment"
GITHUB_BRANCH = "main"

# Google Drive路径（如果使用方案一或三）
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/Image-to-Image-Translation-Experiment")
```

---

## ❓ 常见问题

### Q1: Google Drive同步很慢怎么办？
- 首次同步大文件需要时间，这是正常的
- 后续增量同步会很快
- 可以考虑只同步源代码，数据文件单独处理

### Q2: 数据文件太大，GitHub无法上传？
- 使用Git LFS（见方案二）
- 或者数据文件只上传到Google Drive，代码用GitHub

### Q3: 如何确认文件已同步？
- Google Drive for Desktop会在系统托盘显示同步状态
- 在Google Drive网页版查看文件修改时间
- 在Colab中运行notebook，代码会自动验证文件是否存在

### Q4: 可以同时使用多种方案吗？
- 可以！例如：代码用GitHub，数据用Google Drive
- 代码会自动处理混合情况

---

## 📝 总结

**最推荐的方案**：Google Drive for Desktop
- 设置简单，一次配置，长期使用
- 自动同步，无需手动操作
- 适合大多数用户

**进阶方案**：Git + GitHub
- 适合需要版本控制的用户
- 代码同步更专业
- 数据文件可配合Google Drive使用

现在选择适合你的方案，开始高效开发吧！🚀

