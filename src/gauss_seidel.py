import numpy as np
from scipy.sparse import isspmatrix_csr


def gauss_seidel(
    A,
    b: np.ndarray,
    x0: np.ndarray = None,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> tuple:
    """Giải hệ Ax = b bằng phương pháp lặp Gauss-Seidel trực tiếp trên cấu trúc CSR.

    Trả về: (x, iterations, residual_history, converged)
    """
    raise NotImplementedError("Giáp triển khai ở Tuần 2.")
