import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def build_system(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Xây dựng hệ phương trình thưa Ax = b dạng CSR từ image (float64) và mask (0/1).

    Trả về: (A_csr, b, unknown_pixels)
    """
    raise NotImplementedError("An triển khai ở Tuần 2.")


def make_initial_guess(
    image: np.ndarray,
    mask: np.ndarray,
    unknown_pixels: np.ndarray,
    mode: str = "zero",
    seed: int = 42,
) -> np.ndarray:
    """Khởi tạo vector x0 kích thước n theo 3 chế độ: 'zero', 'random', 'boundary_mean'."""
    raise NotImplementedError("An triển khai ở Tuần 2.")


def solve_reference(A_csr: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Giải hệ Ax = b bằng spsolve làm nghiệm tham chiếu kiểm chứng."""
    raise NotImplementedError("An triển khai ở Tuần 2.")
