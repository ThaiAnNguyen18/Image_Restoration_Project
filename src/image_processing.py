import numpy as np
from PIL import Image


def load_grayscale(path: str) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, chuyển về grayscale (L) và trả về mảng 2D float64 miền [0.0, 255.0]."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Ảnh phải là mảng 2D grayscale, nhận được ndim={arr.ndim}.")
    return arr


def restore_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    unknown_pixels: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Tạo bản sao (copy) của ảnh gốc và điền vector nghiệm x vào đúng danh sách tọa độ unknown_pixels."""
    if image.shape != mask.shape:
        raise ValueError(
            f"image.shape {image.shape} phải khớp mask.shape {mask.shape}."
        )
    if len(unknown_pixels) != len(x):
        raise ValueError(
            f"len(unknown_pixels)={len(unknown_pixels)} phải khớp len(x)={len(x)}."
        )
    out = image.astype(np.float64, copy=True)
    out[unknown_pixels[:, 0], unknown_pixels[:, 1]] = x
    return out


def save_image(image: np.ndarray, path: str) -> None:
    """Giới hạn giá trị về [0, 255], làm tròn, ép kiểu uint8 và lưu ra file PNG."""
    clipped = np.clip(image, 0.0, 255.0)
    rounded = np.round(clipped).astype(np.uint8)
    Image.fromarray(rounded, mode="L").save(path, format="PNG")
