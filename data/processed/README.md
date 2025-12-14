# data/processed/

## 目录说明

此目录用于存放**分割后的label和photo图像对**。

## 文件命名规范

建议命名格式：
- `{image_id}_photo.jpg` - 真实街景照片
- `{image_id}_label.png` - 语义标签图

## 处理流程

1. 从 `data/raw/` 读取拼接图像
2. 分割为独立的 label 和 photo 图像对
3. 保存到此目录

## 注意事项

- label图像通常为单通道或RGB彩色标签图，需确认格式
- 确保label和photo图像对一一对应

