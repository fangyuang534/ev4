# -*- coding: utf-8 -*-
"""
实验4 内容1：基于 OpenCV DNN 模块的 SSD(MobileNet-SSD) 目标检测
---------------------------------------------------------------
功能：
  1. 对单张图片执行目标检测，绘制检测框并保存结果；
  2. 调用本机摄像头执行实时目标检测（按 q 键退出）。

原理说明：
  SSD（Single Shot MultiBox Detector）属于单阶段检测器，
  在多个不同尺度的特征图上直接回归目标框与类别，速度较快。
  本例使用在 VOC0712 数据集上预训练的 MobileNet-SSD 模型
  （Caffe 格式），通过 cv2.dnn.readNetFromCaffe 加载推理。
  MobileNet-SSD 支持 20 个 PASCAL VOC 类别。

运行方式：
  python ssd_detection.py            # 默认检测 test_bus.jpg
  python ssd_detection.py 图片路径    # 检测指定图片
  python ssd_detection.py --camera   # 摄像头实时检测
"""

import argparse
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

# ----------------------- 基本配置 -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")


def ascii_safe(path):
    """
    OpenCV 在 Windows 下用 C 标准库 fopen 读文件，无法打开含中文的路径。
    若路径包含非 ASCII 字符，则把文件复制到纯英文的临时目录再返回新路径。
    """
    try:
        path.encode("ascii")
        return path  # 纯英文路径，直接使用
    except UnicodeEncodeError:
        tmp_dir = os.path.join(tempfile.gettempdir(), "cv_dnn_models")
        os.makedirs(tmp_dir, exist_ok=True)
        new_path = os.path.join(tmp_dir, os.path.basename(path))
        if (not os.path.exists(new_path)
                or os.path.getsize(new_path) != os.path.getsize(path)):
            shutil.copy(path, new_path)
        return new_path


def imread_unicode(path):
    """支持中文路径的图片读取（cv2.imread 在 Windows 下不支持中文路径）"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """支持中文路径的图片保存"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok

PROTOTXT_FILE = os.path.join(MODEL_DIR, "deploy.prototxt")              # 网络结构
CAFFEMODEL_FILE = os.path.join(MODEL_DIR, "mobilenet_iter_73000.caffemodel")  # 权重
NAMES_FILE = os.path.join(MODEL_DIR, "voc.names")                       # 20 类类别名

CONF_THRESHOLD = 0.5  # 置信度阈值
INPUT_SIZE = (300, 300)  # SSD 网络输入尺寸

# VOC 20 类固定颜色
rng = np.random.default_rng(7)
COLORS = rng.uniform(60, 255, size=(21, 3)).astype(np.uint8).tolist()


def load_model():
    """加载 Caffe 格式的 MobileNet-SSD 模型"""
    print("[INFO] 正在加载 MobileNet-SSD 模型...")
    net = cv2.dnn.readNetFromCaffe(ascii_safe(PROTOTXT_FILE),
                                   ascii_safe(CAFFEMODEL_FILE))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] 模型加载完成")
    return net


def load_class_names():
    """
    读取 VOC 20 类类别名称。
    注意：SSD 的输出类别 id 从 0 开始，0 是 background（背景类），
    而 voc.names 文件只有 20 个实际类别，因此要在前面补上 background。
    """
    with open(NAMES_FILE, "rt", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return ["background"] + names


def detect_objects(net, frame, class_names):
    """对一帧图像执行 SSD 检测，返回标注后的图像与检测信息"""
    h, w = frame.shape[:2]

    # SSD 的输入预处理：缩放到 300x300，均值 127.5，尺度 1/127.5
    # 注意：Caffe 模型使用 BGR 通道顺序，因此 swapRB=False
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 127.5, INPUT_SIZE, (127.5, 127.5, 127.5),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    # 前向推理：输出形状 (1, 1, N, 7)，每行为
    # [_, class_id, confidence, x1, y1, x2, y2]，坐标已归一化
    detections = net.forward()

    result = frame.copy()
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESHOLD:
            continue
        class_id = int(detections[0, 0, i, 1])
        # 归一化坐标 -> 像素坐标
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        # 防止坐标越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        label = f"{class_names[class_id]}: {confidence:.2f}"
        results.append((class_names[class_id], confidence, (x1, y1, x2, y2)))
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 6 if y1 - 15 > 0 else y1 + 15
        cv2.putText(result, label, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result, results


def detect_image(net, names, image_path):
    """对单张图片进行检测并保存结果"""
    frame = imread_unicode(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return
    print(f"[INFO] 正在检测图片: {image_path}")
    t0 = time.time()
    result, results = detect_objects(net, frame, names)
    print(f"[INFO] 推理耗时 {time.time() - t0:.2f}s，检测到 {len(results)} 个目标:")
    for name, conf, (x1, y1, x2, y2) in results:
        print(f"    - {name:<12s} 置信度={conf:.2f}  位置=({x1},{y1})-({x2},{y2})")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"ssd_{os.path.basename(image_path)}")
    imwrite_unicode(out_path, result)
    print(f"[INFO] 结果已保存: {out_path}")
    if os.environ.get("EXP4_NO_SHOW") != "1":
        cv2.imshow("MobileNet-SSD Detection (press any key to exit)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_camera(net, names, camera_id=0):
    """调用摄像头进行实时检测，按 q 键退出"""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}")
        return
    print("[INFO] 摄像头已打开，开始实时检测，按 q 键退出...")

    fps, prev = 0.0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 读取摄像头画面失败")
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now

        result, _ = detect_objects(net, frame, names)
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("MobileNet-SSD Real-time Detection (press q to quit)", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 摄像头检测结束")


def main():
    parser = argparse.ArgumentParser(description="OpenCV DNN MobileNet-SSD 目标检测")
    parser.add_argument("image", nargs="?", default=None, help="待检测图片路径")
    parser.add_argument("--camera", action="store_true", help="使用摄像头实时检测")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头编号，默认 0")
    args = parser.parse_args()

    net = load_model()
    names = load_class_names()

    if args.camera:
        detect_camera(net, names, args.camera_id)
    else:
        image = args.image or os.path.join(MODEL_DIR, "test_bus.jpg")
        detect_image(net, names, image)


if __name__ == "__main__":
    main()
