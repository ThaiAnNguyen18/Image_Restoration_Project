import numpy as np
from scipy.sparse import csr_matrix
from src.gauss_seidel import gauss_seidel


def test_gauss_seidel_2x2():
    # Ma trận A dạng CSR và vector b từ bài test mẫu 2x2 (4 ẩn)
    A_dense = np.array(
        [
            [4.0, -1.0, -1.0, 0.0],
            [-1.0, 4.0, 0.0, -1.0],
            [-1.0, 0.0, 4.0, -1.0],
            [0.0, -1.0, -1.0, 4.0],
        ],
        dtype=np.float64,
    )
    A_csr = csr_matrix(A_dense)
    b = np.array([200.0, 320.0, 110.0, 270.0], dtype=np.float64)

    # Nghiệm giải tích chính xác
    x_expected = np.array(
        [1265.0 / 12.0, 1645.0 / 12.0, 1015.0 / 12.0, 1475.0 / 12.0],
        dtype=np.float64,
    )

    # 1. Chạy với x0 = None (mặc định vector 0)
    x_sol, iters, hist, conv = gauss_seidel(A_csr, b, x0=None, tol=1e-10, max_iter=1000)
    assert conv, "Thuật toán không báo hội tụ!"
    assert np.allclose(x_sol, x_expected, atol=1e-7), f"Nghiệm lệch chuẩn: {x_sol} vs {x_expected}"
    print(f"[TEST 1 PASS] x0=None: Hội tụ sau {iters} vòng lặp.")

    # 2. Chạy với x0 khởi tạo tùy ý
    x0 = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    x_sol2, iters2, hist2, conv2 = gauss_seidel(A_csr, b, x0=x0, tol=1e-10, max_iter=1000)
    assert conv2, "Thuật toán không báo hội tụ với x0 tùy ý!"
    assert np.allclose(x_sol2, x_expected, atol=1e-7), "Nghiệm lệch chuẩn với x0 tùy ý!"
    print(f"[TEST 2 PASS] x0=[100,100,100,100]: Hội tụ sau {iters2} vòng lặp.")

    # 3. Kiểm tra các ngoại lệ (validation)
    try:
        gauss_seidel(A_csr, b, tol=-1.0)
        assert False, "Không bắt lỗi tol âm!"
    except ValueError:
        pass

    try:
        gauss_seidel(A_csr, b[:2])
        assert False, "Không bắt lỗi lệch chiều vector b!"
    except ValueError:
        pass

    print("[TEST 3 PASS] Bắt ngoại lệ đúng chuẩn.")
    print("\n>>> TẤT CẢ CÁC BÀI TEST CỦA GAUSS-SEIDEL ĐÃ ĐẠT CHUẨN 100%! <<<")


if __name__ == "__main__":
    test_gauss_seidel_2x2()
