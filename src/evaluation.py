import time
from typing import Any, Callable, Dict

import numpy as np


def compute_relative_residual(
    A, b: np.ndarray, x: np.ndarray, eps: float = 1e-12
) -> float:
    """Tính phần dư tương đối theo chuẩn vô cùng: ||b - Ax||_inf / max(||b||_inf, eps).

    Tham số:
        A: Ma trận hệ số vuông kích thước (n, n) (CSR sparse hoặc ndarray).
        b: Vector vế phải độ dài n.
        x: Vector nghiệm cần đánh giá độ dài n.
        eps: Hằng số dương tránh chia cho 0 khi b = 0 (mặc định 1e-12).

    Trả về:
        float: Giá trị phần dư tương đối r_rel.
    """
    b_arr = np.asarray(b, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    if b_arr.ndim != 1 or x_arr.ndim != 1:
        raise ValueError("Vector b và x phải là mảng 1 chiều.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Ma trận A phải là ma trận vuông 2 chiều.")
    if len(b_arr) != A.shape[0] or len(x_arr) != A.shape[0]:
        raise ValueError(
            f"Kích thước không tương thích: A={A.shape}, len(b)={len(b_arr)}, len(x)={len(x_arr)}."
        )
    if eps <= 0:
        raise ValueError("Tham số eps phải là số dương.")

    r_inf = float(np.linalg.norm(b_arr - A.dot(x_arr), ord=np.inf))
    denom = max(float(np.linalg.norm(b_arr, ord=np.inf)), float(eps))
    return r_inf / denom


def compute_solution_error(x: np.ndarray, x_ref: np.ndarray) -> float:
    """Tính sai lệch cực đại so với nghiệm tham chiếu spsolve: ||x - x_ref||_inf.

    Tham số:
        x: Vector nghiệm thu được từ phương pháp lặp Gauss-Seidel.
        x_ref: Vector nghiệm tham chiếu trực tiếp từ spsolve.

    Trả về:
        float: Sai lệch chuẩn vô cùng e_ref.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_ref_arr = np.asarray(x_ref, dtype=np.float64)

    if x_arr.ndim != 1 or x_ref_arr.ndim != 1:
        raise ValueError("Vector nghiệm x và x_ref phải là mảng 1 chiều.")
    if len(x_arr) != len(x_ref_arr):
        raise ValueError("Vector nghiệm x và x_ref không cùng kích thước.")

    return float(np.linalg.norm(x_arr - x_ref_arr, ord=np.inf))


def compute_mse_psnr(
    original: np.ndarray, restored: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """Tính MSE và PSNR (dB) CHỈ trên tập hợp các pixel bị khuyết (mask == 1).

    Lưu ý: So sánh trực tiếp trên mảng số thực float64 trước khi ép kiểu uint8.
    Nếu MSE_mask == 0.0 thì PSNR trả về float('inf').

    Tham số:
        original: Ảnh gốc 2D (float64, miền [0.0, 255.0]).
        restored: Ảnh đã khôi phục 2D (float64).
        mask: Ma trận mặt nạ 2D (0 = đã biết, 1 = bị khuyết).

    Trả về:
        dict: {'mse': float, 'psnr': float}.
    """
    orig_arr = np.asarray(original, dtype=np.float64)
    rest_arr = np.asarray(restored, dtype=np.float64)
    mask_arr = np.asarray(mask)

    if orig_arr.shape != rest_arr.shape or orig_arr.shape != mask_arr.shape:
        raise ValueError("original, restored và mask phải có cùng kích thước.")
    if not np.isin(mask_arr, [0, 1]).all():
        raise ValueError("Mask chỉ được chứa giá trị 0 (đã biết) và 1 (bị khuyết).")

    omega = mask_arr == 1
    n_missing = int(np.sum(omega))
    if n_missing == 0:
        raise ValueError("Mask không chứa pixel khuyết nào để tính MSE/PSNR.")

    diff = orig_arr[omega] - rest_arr[omega]
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mse": mse, "psnr": psnr}


def benchmark_solver(
    solver_fn: Callable[..., Any],
    *args,
    repeats: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Đo thời gian thực thi của riêng hàm giải lặp bằng time.perf_counter().

    Lặp `repeats` lần (mặc định 5 lần) trên cùng dữ liệu đầu vào và trả về
    thời gian trung vị (median) cùng kết quả của bộ giải.

    Tham số:
        solver_fn: Hàm bộ giải (ví dụ: gauss_seidel).
        *args, **kwargs: Tham số truyền trực tiếp vào solver_fn.
        repeats: Số lần lặp đo thời gian (mặc định 5, phải >= 1).
         Trả về:
        dict: {
            'result': kết quả trả về từ solver_fn,
            'median_time': float (giây),
            'times': list[float] (danh sách thời gian từng lần chạy)
        }
    """
    if repeats < 1:
        raise ValueError("Số lần lặp đo thời gian (repeats) phải >= 1.")

    times = []
    result = None
    for _ in range(repeats):
        t_start = time.perf_counter()
        result = solver_fn(*args, **kwargs)
        t_end = time.perf_counter()
        times.append(t_end - t_start)

    return {
        "result": result,
        "median_time": float(np.median(times)),
        "times": times,
    }
