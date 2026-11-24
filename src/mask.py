import numpy as np


def create_square_mask(shape: tuple, row: int, col: int, size: int) -> np.ndarray:
    """Tạo mask vuông size x size (0 = đã biết, 1 = bị khuyết), cách mép ảnh ít nhất 1 pixel."""
    raise NotImplementedError("Sơn triển khai ở Tuần 2.")


def create_scratch_mask(
    shape: tuple, row: int, col: int, height: int, width: int
) -> np.ndarray:
    """Tạo mask chữ nhật height x width nằm hoàn toàn bên trong ảnh (cách mép ít nhất 1 pixel)."""
    raise NotImplementedError("Sơn triển khai ở Tuần 2.")
