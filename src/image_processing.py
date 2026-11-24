import numpy as np
from PIL import Image


def load_grayscale(path: str) -> np.ndarray:
    """Đọc ảnh từ đường dẫn, chuyển về grayscale (L) và trả về mảng 2D float64 miền [0.0, 255.0]."""
    raise NotImplementedError("Sơn triển khai ở Tuần 2.")


def restore_pixels(
    image: np.ndarray,
    mask: np.ndarray,
    unknown_pixels: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Tạo bản sao (copy) của ảnh gốc và điền vector nghiệm x vào đúng danh sách tọa độ unknown_pixels."""
    raise NotImplementedError("Sơn triển khai ở Tuần 2.")


def save_image(image: np.ndarray, path: str) -> None:
    """Giới hạn giá trị về [0, 255], làm tròn, ép kiểu uint8 và lưu ra file PNG."""
    raise NotImplementedError("Sơn triển khai ở Tuần 2.")
