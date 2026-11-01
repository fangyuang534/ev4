# -*- coding: utf-8 -*-
"""
实验4 内容2 - 任务2第一步：扩大数据集
---------------------------------------
题目要求"根据提供的少量标注数据，完成扩大数据集"。
本脚本实现完整的扩大数据集流水线：

  1. 种子图片（少量数据）来自 dataset/raw/；
  2. 用预训练 yolo26n 对种子图片进行**伪标注**，生成 YOLO 格式标签
     （个人数据集类别：person=0, bus=1）；
  3. 对每张已标注图片做多种数据增强，并**同步变换标签框**，
     将数据集扩充约 10 倍；
  4. 按 8:2 划分 train/val，生成 data.yaml。

运行：python task2_expand_dataset.py
"""

import os
import random
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS = os.path.join(BASE_DIR, "yolo26n.pt")

# 个人数据集的类别定义（与 COCO 类别 id 的映射关系见 coco_to_ours）
CLASS_NAMES = ["person", "bus"]
COCO_TO_OURS = {0: 0,   # person -> person
                5: 1}   # bus    -> bus

TARGET_PER_IMAGE = 10   # 每张种子图扩充出的图片总数
TRAIN_RATIO = 0.8
random.seed(42)


# ----------------------- 标签文件读写 -----------------------
def save_labels(path, boxes):
    """boxes: [(cls, xc, yc, w, h)] 归一化坐标，保存为 YOLO txt"""
    with open(path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def load_labels(path):
    with open(path, encoding="utf-8") as f:
        boxes = []
        for line in f:
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), *map(float, p[1:])))
    return boxes


# ----------------------- 数据增强 + 标签同步变换 -----------------------
def augment_hflip(img, boxes):
    """水平翻转：x 中心取补"""
    return cv2.flip(img, 1), [(c, 1 - x, y, w, h) for c, x, y, w, h in boxes]


def adjust_brightness_contrast(img, boxes):
    """亮度/对比度扰动：不改变几何，标签原样复制"""
    alpha = random.uniform(0.6, 1.4)   # 对比度
    beta = random.uniform(-40, 40)     # 亮度
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), boxes


def add_gaussian_noise(img, boxes):
    noise = np.random.normal(0, random.uniform(8, 25), img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, boxes


def gaussian_blur(img, boxes):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0), boxes


def jpeg_compress(img, boxes):
    q = random.randint(30, 70)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), boxes


def zoom_crop(img, boxes):
    """随机放大后裁剪回原尺寸：框坐标需从裁剪坐标系重新归一化"""
    h, w = img.shape[:2]
    scale = random.uniform(1.15, 1.4)
    nh, nw = int(h * scale), int(w * scale)
    big = cv2.resize(img, (nw, nh))
    # 随机选择裁剪窗口，保证包含大部分目标
    max_dx, max_dy = nw - w, nh - h
    dx = random.randint(0, max_dx) if max_dx > 0 else 0
    dy = random.randint(0, max_dy) if max_dy > 0 else 0
    crop = big[dy:dy + h, dx:dx + w]
    new_boxes = []
    for c, xc, yc, bw, bh in boxes:
        # 归一化 -> 大图像素坐标 -> 减去裁剪偏移 -> 再归一化
        px, py = xc * nw - dx, yc * nh - dy
        pw, ph = bw * nw, bh * nh
        # 框中心还在画面内才保留，并裁剪到边界
        if -0.5 < px / w < 1.5 and -0.5 < py / h < 1.5:
            xc2 = min(max(px / w, 0), 1)
            yc2 = min(max(py / h, 0), 1)
            new_boxes.append((c, xc2, yc2, min(pw / w, 1), min(ph / h, 1)))
    return crop, new_boxes


AUGMENTATIONS = [augment_hflip, adjust_brightness_contrast,
                 add_gaussian_noise, gaussian_blur, jpeg_compress, zoom_crop]


# ----------------------- 主流程 -----------------------
def pseudo_label(model, img_path):
    """用预训练模型对图片伪标注，返回 (img, boxes)"""
    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(img, conf=0.45, verbose=False)
    boxes = []
    for b in results[0].boxes:
        coco_id = int(b.cls)
        if coco_id in COCO_TO_OURS:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            h, w = img.shape[:2]
            xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            boxes.append((COCO_TO_OURS[coco_id], xc, yc, bw, bh))
    return img, boxes


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p, img):
    ok, buf = cv2.imencode(os.path.splitext(p)[1], img)
    buf.tofile(p)


def main():
    # 1. 准备目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)

    seeds = [f for f in os.listdir(RAW_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not seeds:
        raise FileNotFoundError(f"请先把少量种子图片放入 {RAW_DIR}")

    # 2. 伪标注：用预训练模型给种子图片打标签
    print("[INFO] 加载预训练模型进行伪标注...")
    model = YOLO(WEIGHTS)
    labeled = []  # (name, img, boxes)
    for i, f in enumerate(seeds):
        img, boxes = pseudo_label(model, os.path.join(RAW_DIR, f))
        if not boxes:
            print(f"    [WARN] {f} 未检测到目标类别，跳过")
            continue
        print(f"    {f}: 标注 {len(boxes)} 个框 "
              f"({', '.join(CLASS_NAMES[b[0]] for b in boxes)})")
        labeled.append((os.path.splitext(f)[0], img, boxes))

    # 3. 数据增强扩充：每张种子图生成 TARGET_PER_IMAGE 张
    print("[INFO] 数据增强扩充数据集...")
    all_items = []
    for name, img, boxes in labeled:
        all_items.append((f"{name}_orig", img, boxes))
        for k in range(TARGET_PER_IMAGE - 1):
            aug_img, aug_boxes = img, boxes
            # 随机叠加 2~3 种增强
            for fn in random.sample(AUGMENTATIONS, random.randint(2, 3)):
                aug_img, aug_boxes = fn(aug_img, aug_boxes)
            all_items.append((f"{name}_aug{k}", aug_img, aug_boxes))

    # 4. 打乱后按 8:2 划分 train/val
    random.shuffle(all_items)
    n_train = int(len(all_items) * TRAIN_RATIO)
    for split, items in [("train", all_items[:n_train]),
                         ("val", all_items[n_train:])]:
        for name, img, boxes in items:
            imwrite_unicode(os.path.join(DATASET_DIR, "images", split,
                                         f"{name}.jpg"), img)
            save_labels(os.path.join(DATASET_DIR, "labels", split,
                                     f"{name}.txt"), boxes)

    # 5. 写 data.yaml
    # 注意：path 需写绝对路径（ultralytics 按当前工作目录解析相对路径，易出错），
    # 因此该文件含本机路径，不随仓库提交；换机器后由 task2_train.py 自动重建。
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    n_train_actual = len(os.listdir(os.path.join(DATASET_DIR, "images/train")))
    n_val_actual = len(os.listdir(os.path.join(DATASET_DIR, "images/val")))
    print(f"[DONE] 数据集扩充完成：{len(seeds)} 张种子图 -> "
          f"train {n_train_actual} 张 + val {n_val_actual} 张")
    print(f"[DONE] 数据集配置: {yaml_path}")


if __name__ == "__main__":
    main()
