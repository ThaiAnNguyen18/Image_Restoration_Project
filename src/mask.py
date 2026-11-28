import numpy as np


def create_square_mask(shape: tuple, row: int, col: int, size: int) -> np.ndarray:
    """Tạo mask vuông size x size (0 = đã biết, 1 = bị khuyết), cách mép ảnh ít nhất 1 pixel."""
    return create_scratch_mask(shape, row, col, size, size)


def create_scratch_mask(
    shape: tuple, row: int, col: int, height: int, width: int
) -> np.ndarray:
    """Tạo mask chữ nhật height x width nằm hoàn toàn bên trong ảnh (cách mép ít nhất 1 pixel)."""
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Kích thước vùng khuyết phải dương, nhận được {height}x{width}."
        )
    h, w = shape
    if row < 1 or col < 1 or row + height > h - 1 or col + width > w - 1:
        raise ValueError(
            f"Vùng khuyết (row={row}, col={col}, {height}x{width}) "
            f"phải cách mép ảnh {shape} ít nhất 1 pixel."
        )
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row:row + height, col:col + width] = 1
    return mask
