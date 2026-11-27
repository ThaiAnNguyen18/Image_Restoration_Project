"""Sinh dữ liệu cố định cho thực nghiệm: 2 ảnh PNG 256x256 và 3 mask TN1 (.npy).

Chạy từ thư mục gốc dự án:
    .venv/Scripts/python.exe scripts/generate_data.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_processing import save_image
from src.mask import create_scratch_mask, create_square_mask

SIZE = 256
DATA = Path(__file__).resolve().parents[1] / "data"


def make_smooth_image() -> np.ndarray:
    """Ảnh 1 - nền mượt: hàm sin*cos liên tục, biên độ đạo hàm ~5 giá trị/pixel."""
    X, Y = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    return 127.5 * (1.0 + np.sin(2 * np.pi * X / 160) * np.cos(2 * np.pi * Y / 160))


def make_edges_image() -> np.ndarray:
    """Ảnh 2 - nhiều cạnh: bàn cờ 16px tương phản cao + hình học rỗng + đường chéo."""
    X, Y = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
    checker = ((X // 16 + Y // 16) % 2) * 255
    img = Image.fromarray(checker.astype(np.uint8), mode="L")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 120, 100], outline=255, width=3)
    d.ellipse([140, 130, 230, 220], outline=0, width=3)
    d.line([10, 240, 246, 10], fill=0, width=2)
    return np.asarray(img, dtype=np.float64)


def main() -> None:
    img1 = make_smooth_image()
    save_image(img1, str(DATA / "original" / "image1_smooth.png"))
    img2 = make_edges_image()
    save_image(img2, str(DATA / "original" / "image2_edges.png"))
    print(f"image1_smooth.png: shape={img1.shape}, dtype={img1.dtype}")
    print(f"image2_edges.png: shape={img2.shape}, dtype={img2.dtype}")

    masks = [
        ("mask_scratch_2x50.npy", "scratch 2x50 @ (row=127, col=103)",
         create_scratch_mask((SIZE, SIZE), 127, 103, 2, 50)),
        ("mask_sq_10x10.npy", "square 10x10 @ (row=123, col=123)",
         create_square_mask((SIZE, SIZE), 123, 123, 10)),
        ("mask_sq_30x30.npy", "square 30x30 @ (row=113, col=113)",
         create_square_mask((SIZE, SIZE), 113, 113, 30)),
    ]
    for name, desc, mask in masks:
        np.save(DATA / "masks" / name, mask)
        print(f"{name}: {desc}, sum={int(mask.sum())}")


if __name__ == "__main__":
    main()
