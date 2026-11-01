# -*- coding: utf-8 -*-
"""
实验4 内容1：一键下载 YOLOv3 与 MobileNet-SSD 所需模型文件。

原因：yolov3.weights 约 236MB，超过 GitHub 单文件 100MB 上限，无法随仓库提交，
因此仓库只保留配置文件（.cfg / .prototxt / .names），权重由本脚本下载。

用法：
    python download_models.py
    python download_models.py --mirror ghproxy   # 国内镜像（默认）
"""

import argparse
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MIRRORS = {
    "ghproxy": "https://ghproxy.net/",
    "direct": "",
}

# (保存文件名, 原始 URL, 最小合法字节数)
FILES = [
    ("yolov3.weights",
     "https://pjreddie.com/media/files/yolov3.weights",
     200_000_000),
    ("yolov3.cfg",
     "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
     5_000),
    ("coco.names",
     "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
     300),
    ("deploy.prototxt",
     "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
     10_000),
    ("mobilenet_iter_73000.caffemodel",
     "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
     10_000_000),
    ("voc.names",
     "https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names",
     100),
    ("test_bus.jpg",
     "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
     20_000),
]


def download(name, url, min_size, mirror, retries=3):
    """下载单个文件，带重试与体积校验（避免下载到 HTML 错误页）。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, name)

    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"[SKIP] {name} 已存在（{os.path.getsize(dest) / 1e6:.1f} MB）")
        return True

    full_url = MIRRORS[mirror] + url
    for attempt in range(1, retries + 1):
        try:
            print(f"[GET ] {name}  <- {full_url[:70]}...  (第 {attempt} 次)")
            req = urllib.request.Request(
                full_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)

            size = os.path.getsize(dest)
            if size >= min_size:
                print(f"[OK  ] {name}  {size / 1e6:.1f} MB")
                return True
            print(f"[WARN] {name} 体积异常（{size} B），重试中")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {name} 下载失败：{e}")

    print(f"[FAIL] {name} 下载失败，请手动下载到 {MODEL_DIR}")
    return False


def main():
    parser = argparse.ArgumentParser(description="下载实验4 内容1 所需模型文件")
    parser.add_argument("--mirror", default="ghproxy",
                        choices=list(MIRRORS.keys()),
                        help="ghproxy=国内镜像（默认）；direct=直连原始地址")
    args = parser.parse_args()

    print("=" * 60)
    print("实验4 内容1 模型下载器  |  镜像:", args.mirror)
    print("保存目录:", MODEL_DIR)
    print("=" * 60)

    ok = True
    for name, url, min_size in FILES:
        ok &= download(name, url, min_size, args.mirror)

    print("=" * 60)
    if ok:
        print("全部就绪，可运行：python yolov3_detection.py / python ssd_detection.py")
    else:
        print("部分文件缺失，请检查网络或手动下载。")
        sys.exit(1)


if __name__ == "__main__":
    main()
