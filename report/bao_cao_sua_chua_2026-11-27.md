# BÁO CÁO SỬA CHỮA THEO PHẢN HỒI — 27/11/2026

**Người thực hiện:** Sơn · **Nhánh:** `trinhson` · **Nguồn phản hồi:** 3 ảnh chụp trong `report/` (`image.png`, `image (1).png`, `image (2).png`)

Tổng cộng **3 việc**, mỗi việc một commit, kiểm chứng bằng test (TDD: viết test fail trước → triển khai → test pass).

---

## Việc 1 — Đổi tên 5 file dữ liệu theo Mục 3.1 & Notebook của Hiếu

**Commit:** `16d1a45` — `chore: rename 5 data files to match Muc 3.1 and Hieu's notebook`

Đã đối chiếu trực tiếp notebook `experiments/experiments.ipynb` (nhánh `upstream/feat/hieu-experiments-skeleton` của Hiếu) — notebook đang gọi đúng các tên chuẩn mới, nên đổi tên là bắt buộc để Mục 4 (nap dữ liệu) không văng `FileNotFoundError`:

| Tên cũ | Tên chuẩn mới |
| :--- | :--- |
| `data/original/image1_smooth_256.png` | `data/original/image1_smooth.png` |
| `data/original/image2_edges_256.png` | `data/original/image2_edges.png` |
| `data/masks/mask_2x50.npy` | `data/masks/mask_scratch_2x50.npy` |
| `data/masks/mask_10x10.npy` | `data/masks/mask_sq_10x10.npy` |
| `data/masks/mask_30x30.npy` | `data/masks/mask_sq_30x30.npy` |

**Cập nhật tại:** `scripts/generate_data.py`, `data/SOURCES.md`, `tests/test_image_mask.py`.
**Thực hiện:** xóa 5 file cũ (`git rm`) → chạy `scripts/generate_data.py` sinh lại theo tên chuẩn → git nhận diện rename 100% (nội dung 5 file không đổi: mask sum 100/100/900, ảnh 256×256).

## Việc 2 — Xóa thư mục `docs/superpowers/` trước khi merge

**Commit:** `9ff6aa5` — `chore: remove AI-generated plan and spec docs before merge`

Xóa 2 file kế hoạch/spec do công cụ AI agent sinh ra (có dòng chỉ dẫn `For agentic workers...`) để không đẩy vào nhánh `main` nộp giảng viên:
- `docs/superpowers/plans/2026-11-26-son-data-slide.md`
- `docs/superpowers/specs/2026-11-26-son-data-slide-design.md`

Thư mục `docs/` giờ trống; `README.md` không tham chiếu gì đến chúng.

## Việc 3 — Bổ sung chốt chặn `ValueError` + ép kiểu theo Mục 3.2 (trang 8–9)

**Commit:** `314a628` — `feat: add ValueError guards and float64/L type enforcement per Muc 3.2`

### `src/image_processing.py`
- **`restore_pixels`:** thêm `.astype(np.float64, copy=True)` — ảnh truyền vào là `uint8` sẽ không còn cắt phần thập phân của nghiệm `x`; đồng thời 2 chốt chặn:
  - `image.shape != mask.shape` → `ValueError`;
  - `len(unknown_pixels) != len(x)` → `ValueError`.
- **`load_grayscale`:** thêm `if arr.ndim != 2: raise ValueError(...)`.
- **`save_image`:** thêm `mode="L"` vào `Image.fromarray(..., mode="L")`.

### `src/mask.py`
- Thêm kiểm tra `height <= 0 or width <= 0` (bao gồm cả `size <= 0` của mask vuông) → `ValueError`, không còn tạo mask rỗng im lặng khi truyền 0/số âm.
- `create_square_mask` giờ **gọi lại `create_scratch_mask(shape, row, col, size, size)`** (đúng mẫu, bỏ lặp code).

---

## Kiểm chứng

| Lệnh (chạy từ thư mục gốc dự án) | Kết quả |
| :--- | :--- |
| `.venv\Scripts\python.exe -m tests.test_image_mask` | `ALL IMAGE/MASK TESTS PASSED!` — **11 test** (3 test mới cho Việc 3, 2 test đổi tên ở Việc 1) |
| `.venv\Scripts\python.exe scripts/generate_data.py` | Sinh đúng 5 file tên mới, sum mask 100/100/900 |
| `.venv\Scripts\python.exe -m tests.test_2x2` | Vẫn fail tại `build_system` — `NotImplementedError` của An (**đúng dự đoán, ngoài phạm vi Sơn**) |

## Commit trên nhánh `trinhson` (lần sửa này)

```
314a628 feat: add ValueError guards and float64/L type enforcement per Muc 3.2
9ff6aa5 chore: remove AI-generated plan and spec docs before merge
16d1a45 chore: rename 5 data files to match Muc 3.1 and Hieu's notebook
```

## Ghi chú khác

1. **Chốt chặn `ndim != 2` trong `load_grayscale` là phòng thủ** — qua PIL, `convert("L")` luôn cho mảng 2D nên không thể tạo case fail bằng test tự nhiên; dòng này tuân thủ Mục 3.2 nhưng không có test kích hoạt.
2. **Các mục trong phản hồi trùng với "deferred minor" của review 26/11 đã được giải quyết:** mask `size<=0` (mục 3), kiểm tra khớp `mask`/`unknown_pixels`/`x` (mục 3).
3. **Vẫn còn treo (ngoài phạm vi phản hồi, chưa làm):** `save_image` ghi NaN thành vùng đen chỉ kèm warning; test RGB chỉ dùng pixel xám đều; thông báo `ValueError` có dấu tiếng Việt sẽ crash `print(e)` trên console cp1252.
4. 3 ảnh phản hồi `report/image*.png` đang ở trạng thái untracked (chưa commit).
