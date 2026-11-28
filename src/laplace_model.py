import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse.linalg import spsolve


def build_system(image: np.ndarray, mask: np.ndarray) -> tuple:
    """Xây dựng hệ phương trình thưa Ax = b dạng CSR từ image (float64) và mask (0/1).

    Trả về: (A_csr, b, unknown_pixels)
    """
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("image và mask phải là mảng 2 chiều.")
    if image.shape != mask.shape:
        raise ValueError("image và mask phải có cùng kích thước.")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask chỉ được chứa giá trị 0 (đã biết) và 1 (bị khuyết).")
    if (
        np.any(mask[0, :] == 1)
        or np.any(mask[-1, :] == 1)
        or np.any(mask[:, 0] == 1)
        or np.any(mask[:, -1] == 1)
    ):
        raise ValueError("Mask không được có pixel khuyết chạm mép ngoài cùng của ảnh.")

    unknown_pixels = np.argwhere(mask == 1)
    n = len(unknown_pixels)
    if n == 0:
        raise ValueError("Mask không có pixel bị khuyết nào (n = 0).")

    # Bảng tra cứu nhanh từ tọa độ (i, j) sang chỉ số ẩn k (0..n-1)
    idx_map = np.full(mask.shape, -1, dtype=np.int64)
    for k, (i, j) in enumerate(unknown_pixels):
        idx_map[i, j] = k

    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)

    for k, (i, j) in enumerate(unknown_pixels):
        rows.append(k)
        cols.append(k)
        data.append(4.0)

        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if mask[ni, nj] == 1:
                m = int(idx_map[ni, nj])
                rows.append(k)
                cols.append(m)
                data.append(-1.0)
            else:
                b[k] += float(image[ni, nj])

    A_csr = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return A_csr, b, unknown_pixels


def make_initial_guess(
    image: np.ndarray,
    mask: np.ndarray,
    unknown_pixels: np.ndarray,
    mode: str = "zero",
    seed: int = 42,
) -> np.ndarray:
    """Khởi tạo vector x0 kích thước n theo 3 chế độ: 'zero', 'random', 'boundary_mean'."""
    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("image và mask phải là mảng 2 chiều.")
    if image.shape != mask.shape:
        raise ValueError("image và mask phải có cùng kích thước.")
    n = len(unknown_pixels)
    if n == 0:
        raise ValueError("Danh sách unknown_pixels rỗng (n = 0).")

    if mode == "zero":
        return np.zeros(n, dtype=np.float64)

    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 255.0, size=n).astype(np.float64)

    if mode == "boundary_mean":
        h, w = mask.shape
        boundary_coords = set()
        for i, j in unknown_pixels:
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] == 0:
                    boundary_coords.add((int(ni), int(nj)))
        if not boundary_coords:
            raise ValueError("Không tìm thấy pixel biên đã biết nào kề với vùng khuyết.")
        mean_val = float(np.mean([image[ni, nj] for ni, nj in boundary_coords]))
        return np.full(n, mean_val, dtype=np.float64)

    raise ValueError(
        f"Chế độ khởi tạo không hợp lệ: '{mode}'. Chọn 'zero', 'random' hoặc 'boundary_mean'."
    )


def solve_reference(A_csr: csr_matrix, b: np.ndarray) -> np.ndarray:
    """Giải hệ Ax = b bằng spsolve làm nghiệm tham chiếu kiểm chứng."""
    if not isspmatrix_csr(A_csr):
        A_csr = A_csr.tocsr()
    if A_csr.shape[0] != A_csr.shape[1] or A_csr.shape[0] != len(b):
        raise ValueError("Kích thước ma trận A và vector b không tương thích.")
    return np.asarray(spsolve(A_csr, b), dtype=np.float64)
