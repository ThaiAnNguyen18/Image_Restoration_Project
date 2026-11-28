import numpy as np
from scipy.sparse import isspmatrix_csr


def gauss_seidel(
    A,
    b: np.ndarray,
    x0: np.ndarray = None,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> tuple:
    """
    Giải hệ phương trình Ax = b bằng phương pháp lặp Gauss-Seidel trực tiếp trên cấu trúc CSR.

    Tham số:
        A: Ma trận hệ số (csr_matrix hoặc có thể chuyển sang CSR)
        b: Vector vế phải (n,)
        x0: Vector khởi tạo ban đầu (n,). Nếu None thì mặc định là vector 0.
        tol: Ngưỡng dừng sai số phần dư tương đối
        max_iter: Số vòng lặp tối đa để tránh lặp vô hạn

    Trả về:
        (x, iterations, residual_history, converged)
    """
    if not isspmatrix_csr(A):
        A = A.tocsr()

    n = A.shape[0]
    if A.shape[1] != n or len(b) != n:
        raise ValueError("Kích thước ma trận A và vector b không tương thích.")
    if tol <= 0 or max_iter <= 0:
        raise ValueError("Ngưỡng dừng tol và số vòng lặp max_iter phải dương.")

    diag = A.diagonal()
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("Ma trận A có phần tử trên đường chéo chính bằng 0.")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        if len(x0) != n:
            raise ValueError(f"Kích thước x0 ({len(x0)}) không khớp với số ẩn n ({n}).")
        x = np.array(x0, dtype=np.float64, copy=True)

    indptr = A.indptr
    indices = A.indices
    data = A.data
    denom = max(float(np.linalg.norm(b, ord=np.inf)), 1e-12)

    residual_history = []
    converged = False

    for it in range(1, max_iter + 1):
        # Cập nhật tuần tự từng ẩn x[i] sử dụng ngay các giá trị x[j] mới vừa tính
        for i in range(n):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            sigma = 0.0
            aii = 1.0
            for idx in range(row_start, row_end):
                j = indices[idx]
                val = data[idx]
                if j == i:
                    aii = val
                else:
                    sigma += val * x[j]
            x[i] = (b[i] - sigma) / aii

        # Tính phần dư tương đối ở cuối vòng lặp thứ it
        residual = b - A.dot(x)
        r_inf = float(np.linalg.norm(residual, ord=np.inf))
        r_rel = r_inf / denom
        residual_history.append(r_rel)

        if r_rel <= tol:
            converged = True
            return x, it, residual_history, converged

    return x, max_iter, residual_history, converged
