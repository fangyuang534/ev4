# -*- coding: utf-8 -*-
"""
实验4 内容2 - 任务1：使用 YOLO26n 源码跑通推理
------------------------------------------------
ultralytics 框架（YOLO 官方源码库）内部实现了 YOLO26 的
网络结构、前向推理与后处理（YOLO26 为端到端无 NMS 设计）。

功能：
  1. 对图片执行推理，结果保存到 runs/predict/；
  2. --camera 调用摄像头实时推理（按 q 退出）。

运行：
  python task1_inference.py
  python task1_inference.py 图片路径
  python task1_inference.py --camera
"""

import argparse
import os

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "yolo26n.pt")
DEFAULT_IMG = os.path.join(BASE_DIR, "models", "test_bus.jpg")


def infer_image(model, img_path):
    """对单张图片推理并保存可视化结果"""
    # conf：置信度阈值。YOLO26 端到端输出，无需额外 NMS 参数
    results = model.predict(img_path, conf=0.25, save=True,
                            project=os.path.join(BASE_DIR, "runs"),
                            name="predict", exist_ok=True)
    r = results[0]
    print(f"[INFO] 图片: {img_path}")
    print(f"[INFO] 检测到 {len(r.boxes)} 个目标:")
    for box in r.boxes:
        cls_name = r.names[int(box.cls)]
        print(f"    - {cls_name:<10s} 置信度={float(box.conf):.2f}  "
              f"框={box.xyxy[0].tolist()}")
    print(f"[INFO] 结果图已保存到: {r.save_dir}")


def infer_camera(model, camera_id=0):
    """摄像头实时推理，按 q 退出"""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 {camera_id}")
        return
    print("[INFO] 摄像头已打开，开始实时推理，按 q 键退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # stream 模式复用内部缓冲，适合视频流场景
        for r in model.predict(frame, conf=0.3, verbose=False):
            annotated = r.plot()  # 画出检测框
            fps_txt = "YOLO26n real-time (press q to quit)"
            cv2.putText(annotated, fps_txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("YOLO26 Camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLO26n 推理")
    parser.add_argument("image", nargs="?", default=None)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    model = YOLO(WEIGHTS)  # 加载预训练权重（文件不存在时会自动下载）
    if args.camera:
        infer_camera(model, args.camera_id)
    else:
        infer_image(model, args.image or DEFAULT_IMG)


if __name__ == "__main__":
    main()
