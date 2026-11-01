# -*- coding: utf-8 -*-
"""
实验4 内容2 - 任务2第二步：在个人数据集上训练 YOLO26n
------------------------------------------------------
基于 COCO 预训练权重 yolo26n.pt 进行迁移学习（fine-tune），
在 task2_expand_dataset.py 生成的 person/bus 个人数据集上训练。

训练完成后自动：
  1. 在验证集上评估 mAP；
  2. 用最好的权重 best.pt 对种子图片推理，保存结果图。

运行：
  python task2_train.py              # 默认 20 个 epoch
  python task2_train.py --epochs 50
"""

import argparse
import ast
import glob
import os

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")
PRETRAINED = os.path.join(BASE_DIR, "yolo26n.pt")
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# 数据集类别（与 task2_expand_dataset.py 中 CLASS_NAMES 保持一致）
DEFAULT_CLASSES = ["person", "bus"]


def ensure_data_yaml():
    """保证 data.yaml 存在，且其中的 path 指向本机的 dataset 目录。

    说明：data.yaml 的 path 必须是绝对路径（ultralytics 按当前工作目录解析相对路径，
    容易出错），因此该文件不随仓库提交（已写入 .gitignore）。
    换机器后运行本脚本会自动按当前目录重新生成，保证开箱可用。
    """
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    names = DEFAULT_CLASSES
    need_write = True

    if os.path.exists(DATA_YAML):
        # 只取 path / names 两个字段，无需引入 pyyaml 依赖
        cfg = {}
        with open(DATA_YAML, "rt", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    cfg[k.strip()] = v.strip()
        try:
            parsed = ast.literal_eval(cfg.get("names", ""))
            if isinstance(parsed, (list, tuple)) and parsed:
                names = list(parsed)
        except (ValueError, SyntaxError):
            pass
        path = cfg.get("path", "")
        # path 有效（且 train 图像目录存在）时无需重写
        if path and os.path.isdir(os.path.join(path, "images", "train")):
            need_write = False

    if need_write:
        with open(DATA_YAML, "wt", encoding="utf-8") as f:
            f.write(f"path: {dataset_dir.replace(os.sep, '/')}\n")
            f.write("train: images/train\nval: images/val\n")
            f.write(f"nc: {len(names)}\n")
            f.write(f"names: {names}\n")
        print(f"[INFO] 已按当前目录生成 data.yaml -> {DATA_YAML}")

    return DATA_YAML


def main():
    parser = argparse.ArgumentParser(description="YOLO26n 个人数据集训练")
    parser.add_argument("--mode", default="train", choices=["train", "predict"],
                        help="train=训练并评估（默认）；predict=用已有权重推理")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=320, help="训练输入尺寸")
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--weights", default=None,
                        help="predict 模式使用的权重，默认 runs/train/yolo26n_person_bus/weights/best.pt")
    parser.add_argument("--source", default=None,
                        help="predict 模式的输入图片，默认 dataset/raw 下第一张图")
    args = parser.parse_args()

    best_pt = (args.weights if args.weights
               else os.path.join(BASE_DIR, "runs", "train", "yolo26n_person_bus",
                                 "weights", "best.pt"))

    # ---- predict 模式：仅用训练好的权重推理，不训练 ----
    if args.mode == "predict":
        if not os.path.exists(best_pt):
            print(f"[ERROR] 权重不存在: {best_pt}\n请先运行 python task2_train.py 完成训练")
            return
        model = YOLO(best_pt)
        source = args.source or (sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg"))) or [None])[0]
        if source is None:
            print(f"[ERROR] 未找到输入图片，请用 --source 指定")
            return
        print(f"[INFO] 使用权重 {best_pt} 对 {source} 推理")
        model.predict(source, conf=0.3, save=True,
                      project=os.path.join(BASE_DIR, "runs", "train",
                                           "yolo26n_person_bus"),
                      name="predict", exist_ok=True)
        print("[RESULT] 推理结果已保存到 runs/train/yolo26n_person_bus/predict/")
        return

    # 0. 确保数据集配置文件可用（换机器后自动重建）
    data_yaml = ensure_data_yaml()

    # 1. 加载 COCO 预训练权重做迁移学习
    model = YOLO(PRETRAINED)
    print(f"[INFO] 开始训练：epochs={args.epochs}, imgsz={args.imgsz}, "
          f"batch={args.batch}（CPU 训练约需几分钟）")

    # 2. 训练。device="cpu" 指定 CPU 训练；结果保存在 runs/train/
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,           # Windows 下避免多进程问题
        project=os.path.join(BASE_DIR, "runs", "train"),
        name="yolo26n_person_bus",
        exist_ok=True,
        patience=50,         # 关闭早停（数据集很小）
        plots=True,          # 生成损失曲线、混淆矩阵等图
    )

    # 3. 用训练得到的最佳权重在验证集上评估
    best_model = YOLO(best_pt)
    metrics = best_model.val(data=data_yaml, device="cpu", verbose=False)
    print("=" * 60)
    print("[RESULT] 验证集评估结果:")
    print(f"    mAP50-95: {metrics.box.map:.4f}")
    print(f"    mAP50   : {metrics.box.map50:.4f}")
    print(f"    最佳权重: {best_pt}")

    # 4. 用最佳权重对种子图片推理，保存可视化结果
    seed_imgs = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))[:1]
    if seed_imgs:
        best_model.predict(seed_imgs[0], conf=0.3, save=True,
                           project=os.path.join(BASE_DIR, "runs", "train",
                                                "yolo26n_person_bus"),
                           name="predict", exist_ok=True)
        print(f"[RESULT] 训练后模型推理结果已保存到 runs/train/yolo26n_person_bus/predict/")


if __name__ == "__main__":
    main()
