import numpy as np
from PIL import Image


def load_grayscale(path: str) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, chuyển về grayscale (L) và trả về mảng 2D float64 miền [0.0, 255.0]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def restore_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    unknown_pixels: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Tạo bản sao (copy) của ảnh gốc và điền vector nghiệm x vào đúng danh sách tọa độ unknown_pixels."""
    out = image.copy()
    out[unknown_pixels[:, 0], unknown_pixels[:, 1]] = x
    return out


def save_image(image: np.ndarray, path: str) -> None:
    """Giới hạn giá trị về [0, 255], làm tròn, ép kiểu uint8 và lưu ra file PNG."""
    clipped = np.clip(image, 0.0, 255.0)
    rounded = np.round(clipped).astype(np.uint8)
    Image.fromarray(rounded).save(path, format="PNG")
