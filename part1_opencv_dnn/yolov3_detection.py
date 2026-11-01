# -*- coding: utf-8 -*-
"""
实验4 内容1：基于 OpenCV DNN 模块的 YOLOv3 目标检测
-----------------------------------------------------
功能：
  1. 对单张图片执行目标检测，绘制检测框并保存结果；
  2. 调用本机摄像头执行实时目标检测（按 q 键退出）。

原理说明：
  cv2.dnn 是 OpenCV 中的通用深度学习推理引擎，支持加载
  Caffe / TensorFlow / Darknet / ONNX 等框架训练好的模型。
  本例将 Darknet 框架的 YOLOv3 网络结构与权重文件加载进
  cv2.dnn.readNetFromDarknet，执行前向推理得到候选检测框，
  再经过置信度过滤 + NMS（非极大值抑制）得到最终结果。

运行方式：
  python yolov3_detection.py            # 默认检测 test_bus.jpg
  python yolov3_detection.py 图片路径    # 检测指定图片
  python yolov3_detection.py --camera   # 摄像头实时检测
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

CFG_FILE = os.path.join(MODEL_DIR, "yolov3.cfg")          # 网络结构文件
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov3.weights")  # 预训练权重
NAMES_FILE = os.path.join(MODEL_DIR, "coco.names")        # 80 类类别名

CONF_THRESHOLD = 0.5   # 置信度阈值：低于该值的检测框被丢弃
NMS_THRESHOLD = 0.4    # 非极大值抑制阈值：IoU 大于该值的重叠框被抑制
INPUT_SIZE = (416, 416)  # YOLOv3 网络输入尺寸（也可用 320/608）

# 固定一组 BGR 颜色，为每个类别分配固定颜色，便于观察
rng = np.random.default_rng(3)
COLORS = rng.uniform(60, 255, size=(80, 3)).astype(np.uint8).tolist()


def load_model():
    """加载 YOLOv3 网络结构与权重，返回 dnn 网络"""
    print("[INFO] 正在加载 YOLOv3 模型（首次加载需要几秒）...")
    net = cv2.dnn.readNetFromDarknet(ascii_safe(CFG_FILE),
                                     ascii_safe(WEIGHTS_FILE))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 无 GPU 时使用 CPU 推理
    print("[INFO] 模型加载完成")
    return net


def load_class_names():
    """读取 COCO 数据集 80 个类别名称"""
    with open(NAMES_FILE, "rt", encoding="utf-8") as f:
        names = f.read().rstrip("\n").split("\n")
    return names


def get_output_layers(net):
    """获取 YOLOv3 的三个检测输出层（三个不同尺度的 yolo 层）"""
    layer_names = net.getLayerNames()
    # getUnconnectedOutLayers() 返回需要输出的层索引
    out_layer_idxs = net.getUnconnectedOutLayers().flatten()
    return [layer_names[i - 1] for i in out_layer_idxs]


def detect_objects(net, frame, class_names):
    """
    对一帧图像执行检测。
    返回绘制好检测框的图像以及检测信息列表。
    """
    h, w = frame.shape[:2]

    # 1. 构造输入 blob：缩放到 416x416、减去均值、交换 R/B 通道
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False
    )
    net.setInput(blob)

    # 2. 前向推理，得到三个尺度输出层的检测结果
    outputs = net.forward(get_output_layers(net))

    boxes, confidences, class_ids = [], [], []
    # 3. 遍历每个输出层：形状为 (候选框数量, 5 + 类别数)
    for output in outputs:
        for detection in output:
            scores = detection[5:]                # 各类别得分
            class_id = int(np.argmax(scores))     # 得分最高的类别
            confidence = float(scores[class_id])  # 最高类别得分
            if confidence > CONF_THRESHOLD:
                # YOLO 输出的中心点坐标是相对 416x416 的归一化值
                box = detection[:4] * np.array([w, h, w, h])
                (cx, cy, bw, bh) = box
                # 转换为左上角坐标 (x, y) + 宽高
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

    # 4. 非极大值抑制（NMS），去掉大量重叠的冗余框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    result = frame.copy()
    detections = []
    if len(idxs) > 0:
        for i in np.array(idxs).flatten():
            x, y, bw, bh = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            detections.append((class_names[class_ids[i]], confidences[i],
                               (x, y, bw, bh)))
            color = COLORS[class_ids[i]]
            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
            # 标签文字：检测框上方，放不下则画在框内
            ty = y - 6 if y - 15 > 0 else y + 15
            cv2.putText(result, label, (x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result, detections


def detect_image(net, names, image_path):
    """对单张图片进行检测并保存结果"""
    frame = imread_unicode(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return
    print(f"[INFO] 正在检测图片: {image_path}")
    t0 = time.time()
    result, detections = detect_objects(net, frame, names)
    print(f"[INFO] 推理耗时 {time.time() - t0:.2f}s，检测到 {len(detections)} 个目标:")
    for name, conf, (x, y, bw, bh) in detections:
        print(f"    - {name:<12s} 置信度={conf:.2f}  位置=({x},{y},{bw},{bh})")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"yolov3_{os.path.basename(image_path)}")
    imwrite_unicode(out_path, result)
    print(f"[INFO] 结果已保存: {out_path}")
    if os.environ.get("EXP4_NO_SHOW") != "1":
        cv2.imshow("YOLOv3 Detection (press any key to exit)", result)
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
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)  # 平滑后的实时帧率
        prev = now

        result, _ = detect_objects(net, frame, names)
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLOv3 Real-time Detection (press q to quit)", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 摄像头检测结束")


def main():
    parser = argparse.ArgumentParser(description="OpenCV DNN YOLOv3 目标检测")
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
