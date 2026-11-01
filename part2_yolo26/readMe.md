# YOLO26 轻量型模型使用说明（readMe）

本目录是"实验4 内容2"的 YOLO26 部分，包含**任务1：跑通推理**与
**任务2：扩大数据集并在个人数据集上训练**两个子任务的完整代码与步骤。

## 1. 环境准备

```bash
# 建议使用 Python 3.10+
pip install ultralytics      # 一行命令安装，自动带上 pytorch 等依赖
```

## 2. 模型选择

YOLO26 提供轻量型模型，官方性能如下（640 输入）：

| 模型 | mAP50-95 | CPU ONNX 速度(ms) | 参数量(M) | FLOPs(B) |
|------|----------|------------------|-----------|----------|
| YOLO26n | 40.9 | 38.9 | 2.4 | 5.5 |
| YOLO26s | 48.6 | 87.2 | 9.5 | 20.9 |

CPU 电脑建议使用 **YOLO26n**（本实验采用）。

## 3. 任务1：跑通推理

预训练权重 `yolo26n.pt` 已放在本目录（若没有，运行脚本时会自动下载，
也可以手动从 https://github.com/ultralytics/assets/releases 下载）。

```bash
python task1_inference.py                 # 对自带测试图推理，结果保存到 runs/
python task1_inference.py 任意图片.jpg    # 对指定图片推理
python task1_inference.py --camera        # 调用摄像头实时推理，按 q 退出
```

也可以用 ultralytics 自带命令行：

```bash
yolo predict model=yolo26n.pt source=models/test_bus.jpg
```

## 4. 任务2：扩大数据集 + 个人数据集训练

个人数据集选择 **person（行人）与 bus（公交车）** 两个类别。
步骤分两步：

### 第一步：扩大数据集

```bash
python task2_expand_dataset.py
```

脚本流程（全自动）：
1. 以 `dataset/raw/` 下少量种子图片为初始数据；
2. 用预训练模型 yolo26n 对种子图片**伪标注**，生成 YOLO 格式
   `.txt` 标签（`类别id x_center y_center width height`，均为归一化坐标）；
3. 对每张标注图做数据增强（水平翻转、亮度/对比度扰动、高斯噪声、
   高斯模糊、随机缩放裁剪等），**同步变换标签框坐标**，
   将数据集从几张图扩充到约 60 张；
4. 按 8:2 划分 train/val，并生成 `data.yaml` 数据集配置文件。

### 第二步：训练

```bash
python task2_train.py              # 默认 20 个 epoch
python task2_train.py --epochs 50  # 指定训练轮数
```

- 基于预训练权重 `yolo26n.pt` 做**迁移学习（fine-tune）**，
  CPU 也能在几分钟内完成；
- 训练结果保存在 `runs/train/`（权重 best.pt、损失曲线 results.png 等）；
- 训练完成后自动在验证集上评估 mAP，并用 `runs/train/weights/best.pt`
  对测试图推理，结果保存在 `runs/train/predict/`。

## 5. 目录结构

```
part2_yolo26/
├── readMe.md                  # 本说明文件
├── task1_inference.py         # 任务1：推理脚本
├── task2_expand_dataset.py    # 任务2第一步：伪标注 + 扩大数据集
├── task2_train.py             # 任务2第二步：训练与评估
├── yolo26n.pt                 # COCO 预训练权重
├── dataset/                   # 数据集（expand 脚本自动生成）
│   ├── raw/                   # 种子图片
│   ├── images/train|val/      # 扩充后的训练/验证图片
│   ├── labels/train|val/      # 对应 YOLO 格式标签
│   └── data.yaml              # 数据集配置
└── runs/                      # 推理与训练输出
```
