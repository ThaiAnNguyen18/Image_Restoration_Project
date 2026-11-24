import numpy as np
from src.mask import create_square_mask
from src.laplace_model import build_system, make_initial_guess, solve_reference
from src.gauss_seidel import gauss_seidel
from src.image_processing import restore_pixels
from src.evaluation import (
    compute_relative_residual,
    compute_solution_error,
    compute_mse_psnr,
)


def run_test_2x2():
    I_orig = np.array(
        [
            [100.0, 120.0, 140.0, 160.0],
            [80.0, 110.0, 120.0, 180.0],
            [60.0, 90.0, 100.0, 200.0],
            [40.0, 50.0, 70.0, 90.0],
        ],
        dtype=np.float64,
    )

    mask = create_square_mask(I_orig.shape, row=1, col=1, size=2)
    A_csr, b, unknown_pixels = build_system(I_orig, mask)

    A_expected = np.array(
        [
            [4, -1, -1, 0],
            [-1, 4, 0, -1],
            [-1, 0, 4, -1],
            [0, -1, -1, 4],
        ],
        dtype=np.float64,
    )
    b_expected = np.array([200.0, 320.0, 110.0, 270.0], dtype=np.float64)
    x_expected = np.array(
        [1265 / 12, 1645 / 12, 1015 / 12, 1475 / 12], dtype=np.float64
    )

    assert np.allclose(A_csr.toarray(), A_expected), "Sai ma trận A!"
    assert np.allclose(b, b_expected), "Sai vector vế phải b!"

    x_ref = solve_reference(A_csr, b)
    assert np.allclose(x_ref, x_expected, atol=1e-10), "Sai nghiệm tham chiếu spsolve!"

    x0 = make_initial_guess(I_orig, mask, unknown_pixels, mode="boundary_mean")
    assert np.allclose(x0, np.full(4, 900.0 / 8.0)), "Sai khởi tạo boundary_mean!"

    x_gs, iters, hist, conv = gauss_seidel(A_csr, b, x0=x0, tol=1e-10, max_iter=1000)
    assert conv and np.allclose(x_gs, x_expected, atol=1e-7), "Sai nghiệm bộ giải Gauss-Seidel!"

    I_restored = restore_pixels(I_orig, mask, unknown_pixels, x_gs)
    assert np.allclose(I_restored[mask == 0], I_orig[mask == 0]), "Pixel ngoài mask bị biến đổi!"

    r_rel = compute_relative_residual(A_csr, b, x_gs)
    assert r_rel <= 1e-10, "Phần dư tương đối vượt ngưỡng!"

    metrics = compute_mse_psnr(I_orig, I_restored, mask)
    assert metrics["mse"] > 0 and compute_solution_error(x_gs, x_ref) < 1e-7

    print(
        f"ALL TESTS PASSED! Hội tụ sau {iters} vòng, "
        f"MSE_mask = {metrics['mse']:.4f}, PSNR = {metrics['psnr']:.2f} dB"
    )


if __name__ == "__main__":
    run_test_2x2()
