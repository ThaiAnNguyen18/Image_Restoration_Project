import numpy as np


def compute_relative_residual(
    A, b: np.ndarray, x: np.ndarray, eps: float = 1e-12
) -> float:
    """Tính phần dư tương đối theo chuẩn vô cùng: ||b - Ax||_inf / max(||b||_inf, eps)."""
    raise NotImplementedError("Hiếu triển khai ở Tuần 2.")


def compute_solution_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """Tính sai lệch cực đại so với nghiệm tham chiếu spsolve: ||x - x_ref||_inf."""
    raise NotImplementedError("Hiếu triển khai ở Tuần 2.")


def compute_mse_psnr(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> dict:
    """Tính MSE và PSNR (dB) CHỈ trên tập hợp các pixel bị khuyết (mask == 1)."""
    raise NotImplementedError("Hiếu triển khai ở Tuần 2.")
