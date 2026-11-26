# Sơn — Dữ liệu & Slide (Tuần 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện `src/image_processing.py` và `src/mask.py`, sinh cố định 2 ảnh PNG 256×256 + 3 mask TN1 dạng `.npy`, viết `data/SOURCES.md`, và tự kiểm chứng toàn bộ phần của Sơn.

**Architecture:** TDD bằng file test riêng `tests/test_image_mask.py` theo đúng pattern `test_2x2.py` (plain asserts, chạy bằng `python -m`, không pytest). Script `scripts/generate_data.py` tái lập được sinh cả ảnh lẫn mask; dữ liệu là file tĩnh được commit vào repo. Script tái sử dụng các hàm của `src/` thay vì tự cài lại.

**Tech Stack:** Python 3.14 (venv theo README), NumPy, Pillow, scipy (cài chung trong `requirements.txt`).

**Spec:** `docs/superpowers/specs/2026-11-26-son-data-slide-design.md`

## Global Constraints
- Mask: `0` = pixel đã biết, `1` = pixel khuyết; dtype `uint8`.
- Ảnh tính toán `float64` miền `[0.0, 255.0]`; chỉ ép `uint8` khi lưu file.
- Thứ tự ẩn số = `np.argwhere(mask == 1)`; `restore_pixels` phải điền `x` đúng theo thứ tự này.
- `restore_pixels` bắt buộc `.copy()` — không được ghi đè ảnh gốc (quy ước README).
- Mask phải cách mép ảnh ≥ 1 pixel; vi phạm → ném `ValueError` (không cắt ngầm).
- TN1: `2x50` (100 pixel), `10x10` (100 pixel), `30x30` (900 pixel), căn giữa ảnh 256×256, lưu `.npy` cố định trước khi chạy.
- Test chạy bằng `.venv\Scripts\python.exe -m tests.<module>` (project không dùng pytest).
- Nhánh làm việc: `trinhson`; mỗi task một commit.

## Review Focus
1. `load_grayscale` nhận PNG **RGB** (không phải ảnh xám) → phải convert `"L"` đúng và trả 2D float64 — ghim bằng `test_load_grayscale_rgb` (Task 2).
2. `save_image` với giá trị ngoài `[0,255]` và số thực lẻ → phải clip + làm tròn đúng, ra `uint8` — ghim bằng `test_save_image_clips_and_rounds` (Task 2).
3. `restore_pixels` với `x` theo thứ tự `np.argwhere(mask==1)` → điền đúng tọa độ, giữ nguyên vùng ngoài mask, **không đổi ảnh gốc** — ghim bằng `test_restore_pixels` (Task 2).
4. Mask chạm/vượt mép ảnh → phải `ValueError` thay vì bị cắt ngầm (mask lệch vị trí làm sai hoàn toàn TN1) — ghim bằng `test_edge_guard` (Task 1).
5. Ảnh 1 phải thực sự mượt, ảnh 2 phải thực sự nhiều cạnh (nếu không, kịch bản "mượt vs nhiều cạnh" của TN1/validation là giả) — ghim bằng `test_images_smooth_vs_edges` (Task 3).
6. Mask TN1 sai tọa độ hoặc sai số pixel `1` → TN1 sai ngay từ đầu — ghim bằng `test_tn1_masks_fixed` (Task 3).

---

### Task 1: Môi trường venv + `src/mask.py` (TDD)

**Files:**
- Create: `tests/test_image_mask.py`
- Modify: `src/mask.py` (2 hàm đang `NotImplementedError`)

**Interfaces:**
- Produces (Task 3 dùng lại): `create_square_mask(shape: tuple, row: int, col: int, size: int) -> np.ndarray` và `create_scratch_mask(shape: tuple, row: int, col: int, height: int, width: int) -> np.ndarray` — trả `uint8` 0/1 đúng `shape`, vùng 1 nằm tại `[row:row+h, col:col+w]`, ném `ValueError` nếu `row < 1 or col < 1 or row+h > H-1 or col+w > W-1`.

- [x] **Step 1: Tạo venv và cài dependencies theo README**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import numpy, scipy, PIL; print('env ok')"
```
Expected: in `env ok` (không lỗi import).

- [x] **Step 2: Viết test thất bại cho mask**

Thêm vào `tests/test_image_mask.py` (pattern y hệt `test_2x2.py`: hàm `run_tests()` + in kết quả, `if __name__ == "__main__"`):

```python
import numpy as np
from src.mask import create_square_mask, create_scratch_mask

def test_square_mask():
    m = create_square_mask((10, 12), row=2, col=3, size=4)
    assert m.dtype == np.uint8 and m.shape == (10, 12)
    assert set(np.unique(m).tolist()) <= {0, 1}
    assert m[2:6, 3:7].sum() == 16 and m.sum() == 16
    m2 = create_square_mask((4, 4), row=1, col=1, size=2)   # đúng ca test_2x2
    assert m2.sum() == 4 and m2[1:3, 1:3].sum() == 4

def test_scratch_mask():
    m = create_scratch_mask((10, 12), row=4, col=1, height=2, width=6)
    assert m.dtype == np.uint8 and m.shape == (10, 12)
    assert m.sum() == 12 and m[4:6, 1:7].sum() == 12

def test_edge_guard():
    bad = [
        (create_square_mask, ((10, 12), 0, 3, 4)),    # row = 0
        (create_square_mask, ((10, 12), 7, 3, 4)),    # 7+4 = 11 > 9
        (create_scratch_mask, ((10, 12), 4, 0, 2, 6)),   # col = 0
        (create_scratch_mask, ((10, 12), 4, 7, 2, 6)),   # 7+6 = 13 > 11
    ]
    for fn, args in bad:
        try:
            fn(*args)
            assert False, f"phải raise ValueError: {fn.__name__}{args}"
        except ValueError:
            pass

def run_tests():
    test_square_mask(); test_scratch_mask(); test_edge_guard()
    print("ALL MASK TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
```

- [x] **Step 3: Chạy test để xác nhận thất bại**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: FAIL với `NotImplementedError` ("Sơn triển khai ở Tuần 2.").

- [x] **Step 4: Triển khai 2 hàm trong `src/mask.py`**

Giữ nguyên docstring có sẵn. Body: cấp phát `np.zeros(shape, dtype=np.uint8)`, kiểm tra 4 điều kiện khoảng cách mép (ném `ValueError` kèm thông tin vi phạm), ghi `1` vào lát cắt tương ứng. Không sửa signature.

- [x] **Step 5: Chạy test để xác nhận đạt**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: `ALL MASK TESTS PASSED!`

- [x] **Step 6: Commit**

```powershell
git add src/mask.py tests/test_image_mask.py
git commit -m "feat: implement mask creation with edge guard and tests"
```

---

### Task 2: `src/image_processing.py` (TDD)

**Files:**
- Modify: `src/image_processing.py` (3 hàm đang `NotImplementedError`)
- Test: append vào `tests/test_image_mask.py`

**Interfaces:**
- Consumes: `create_square_mask` (Task 1).
- Produces: `load_grayscale(path: str) -> np.ndarray` (2D float64 [0,255]), `restore_pixels(image: np.ndarray, mask: np.ndarray, unknown_pixels: np.ndarray, x: np.ndarray) -> np.ndarray` (bản sao đã điền x), `save_image(image: np.ndarray, path: str) -> None`. Task 3 sẽ dùng `load_grayscale` và `save_image`.

- [x] **Step 1: Viết test thất bại (append vào file test)**

```python
import os, tempfile
from PIL import Image
from src.image_processing import load_grayscale, save_image, restore_pixels

def test_load_grayscale_rgb():
    p = os.path.join(tempfile.mkdtemp(), "rgb.png")
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(p)
    a = load_grayscale(p)
    assert a.dtype == np.float64 and a.shape == (8, 8)
    assert 0.0 <= a.min() and a.max() <= 255.0
    assert np.allclose(a, 200.0)   # RGB(200,200,200) -> L = 200

def test_save_image_clips_and_rounds():
    p = os.path.join(tempfile.mkdtemp(), "out.png")
    save_image(np.array([[-10.0, 127.6], [300.0, 255.4]]), p)
    b = np.asarray(Image.open(p))
    assert b.dtype == np.uint8
    assert b.flatten().tolist() == [0, 128, 255, 255]

def test_restore_pixels():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    img_orig = img.copy()
    mask = create_square_mask((4, 4), 1, 1, 2)
    unknown = np.argwhere(mask == 1)
    x = np.array([99.0, 98.0, 97.0, 96.0])
    out = restore_pixels(img, mask, unknown, x)
    assert np.allclose(out[mask == 1], x)                 # đúng thứ tự argwhere
    assert np.allclose(out[mask == 0], img[mask == 0])    # giữ vùng ngoài
    assert np.allclose(img, img_orig)                     # không đổi ảnh gốc
```

Thêm `test_load_grayscale_rgb(); test_save_image_clips_and_rounds(); test_restore_pixels()` vào `run_tests()`.

- [x] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: FAIL với `NotImplementedError`.

- [x] **Step 3: Triển khai 3 hàm trong `src/image_processing.py`**

- `load_grayscale`: `Image.open(path).convert("L")` → `np.asarray(..., dtype=np.float64)` (mảng 2D).
- `restore_pixels`: `out = image.copy()`; điền `x` vào tọa độ `unknown_pixels` (dạng `(n, 2)`), ví dụ `out[unknown_pixels[:, 0], unknown_pixels[:, 1]] = x`; trả `out`.
- `save_image`: `np.clip` về `[0, 255]` → `np.round` → `astype(np.uint8)` → `Image.fromarray(...).save(path, format="PNG")`.

- [x] **Step 4: Chạy test để xác nhận đạt**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: PASS toàn bộ (mask + image), in `ALL MASK TESTS PASSED!` (đổi chuỗi in thành `ALL IMAGE/MASK TESTS PASSED!` cho đúng phạm vi).

- [x] **Step 5: Commit**

```powershell
git add src/image_processing.py tests/test_image_mask.py
git commit -m "feat: implement grayscale load, restore and PNG save"
```

---

### Task 3: Sinh dữ liệu cố định + `SOURCES.md`

**Files:**
- Create: `scripts/generate_data.py`
- Create: `data/original/image1_smooth_256.png`, `data/original/image2_edges_256.png`
- Create: `data/masks/mask_2x50.npy`, `data/masks/mask_10x10.npy`, `data/masks/mask_30x30.npy`
- Modify: `data/SOURCES.md` (đang rỗng)
- Test: append vào `tests/test_image_mask.py`

**Interfaces:**
- Consumes: `create_square_mask`, `create_scratch_mask` (Task 1), `load_grayscale`, `save_image` (Task 2).
- Produces: 5 file dữ liệu tĩnh đúng tên trên — các task sau (notebook TN1) chỉ đọc chúng.

- [x] **Step 1: Viết test thất bại cho dữ liệu (append vào file test)**

```python
def test_images_smooth_vs_edges():
    i1 = load_grayscale("data/original/image1_smooth_256.png")
    i2 = load_grayscale("data/original/image2_edges_256.png")
    for a in (i1, i2):
        assert a.dtype == np.float64 and a.shape == (256, 256)
    d1 = np.abs(np.diff(i1, axis=0)).max(), np.abs(np.diff(i1, axis=1)).max()
    assert max(d1) < 10                          # ảnh 1 gần như phẳng từng pixel
    e2 = (np.abs(np.diff(i2, axis=0)) > 50).sum() + (np.abs(np.diff(i2, axis=1)) > 50).sum()
    assert e2 >= 500                             # ảnh 2 nhiều rìa sắc nét

def test_tn1_masks_fixed():
    specs = [("mask_2x50.npy", 100, (slice(127, 129), slice(103, 153))),
             ("mask_10x10.npy", 100, (slice(123, 133), slice(123, 133))),
             ("mask_30x30.npy", 900, (slice(113, 143), slice(113, 143)))]
    for name, n, region in specs:
        m = np.load(f"data/masks/{name}")
        assert m.dtype == np.uint8 and m.shape == (256, 256)
        assert set(np.unique(m).tolist()) <= {0, 1} and m.sum() == n
        assert m[region].sum() == n               # đúng vị trí căn giữa, lệch 1 pixel cũng fail
```

Thêm 2 hàm trên vào `run_tests()`.

- [x] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: FAIL với `FileNotFoundError` (file chưa tồn tại).

- [x] **Step 3: Viết `scripts/generate_data.py`**

Nội dung bắt buộc:
- Dòng đầu: `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` (để `import src...` chạy được khi gọi `python scripts/generate_data.py` từ thư mục gốc) + `from src.mask import ...`, `from src.image_processing import save_image`.
- `X, Y = np.meshgrid(np.arange(256), np.arange(256))`.
- **Ảnh 1 (mượt):** `img1 = 127.5 * (1.0 + np.sin(2*np.pi*X/160) * np.cos(2*np.pi*Y/160))` — biên độ đạo hàm mỗi pixel ≈ 5, bảo đảm test `max diff < 10`.
- **Ảnh 2 (nhiều cạnh):** bàn cờ `((X//16 + Y//16) % 2) * 255` vẽ bằng PIL `ImageDraw` (chế độ `"L"`), cộng thêm hình chữ nhật rỗng, vòng tròn rỗng và 1 đường chéo độ dày 2 px.
- Lưu ảnh qua `save_image(img1, "data/original/image1_smooth_256.png")` và `save_image(img2_array, "data/original/image2_edges_256.png")` (mảng `float64` từ `np.asarray(pil_image, dtype=np.float64)`).
- 3 mask: `create_scratch_mask((256,256), 127, 103, 2, 50)`, `create_square_mask((256,256), 123, 123, 10)`, `create_square_mask((256,256), 113, 113, 30)` → `np.save` đúng 3 tên file ở trên.
- In ra tọa độ, kích thước và `int(m.sum())` của từng mask để đối chiếu.

- [x] **Step 4: Chạy script rồi chạy test**

Run: `.venv\Scripts\python.exe scripts/generate_data.py` rồi `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: script in 3 dòng summary mask; test in `ALL IMAGE/MASK TESTS PASSED!`.

- [x] **Step 5: Viết `data/SOURCES.md`**

Nội dung bắt buộc (tiếng Việt, markdown):
- Bảng nguồn 2 ảnh: **tự sinh** bởi `scripts/generate_data.py`, ngày `2026-11-26`, thông số 256×256, grayscale, PNG 8-bit; khẳng định không dùng tài nguyên bên thứ ba → không có giới hạn bản quyền.
- Bảng 3 mask TN1: tên file, hình dạng, tọa độ gốc, số pixel khuyết, dạng `.npy` (`uint8` 0/1), sinh cùng script.
- Ghi chú: nhánh B (ảnh thật từ nguồn công khai, có license) sẽ được bổ sung vào file này khi thực hiện.

- [x] **Step 6: Commit**

```powershell
git add scripts/generate_data.py data/original data/masks data/SOURCES.md tests/test_image_mask.py
git commit -m "chore: generate fixed 256x256 images and TN1 masks with SOURCES.md"
```

---

### Task 4: Kiểm chứng tổng hợp (gate hoàn thành)

**Files:** không sửa file nào.

**Interfaces:**
- Consumes: toàn bộ kết quả Task 1–3.

- [x] **Step 1: Chạy bộ test của Sơn**

Run: `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: `ALL IMAGE/MASK TESTS PASSED!`

- [x] **Step 2: Chạy test 2x2 của nhóm (kết quả dự kiến)**

Run: `.venv\Scripts\python.exe -m tests.test_2x2`
Expected: FAIL với `NotImplementedError: An triển khai ở Tuần 2.` tại `build_system` — **đúng dự đoán**, không phải gate hôm nay (đủ 4 người sẽ pass). Ghi nhận kết quả này vào báo cáo cuối ngày.

- [x] **Step 3: Chạy lại script sinh dữ liệu (tái lập)**

Run: `.venv\Scripts\python.exe scripts/generate_data.py` rồi `.venv\Scripts\python.exe -m tests.test_image_mask`
Expected: vẫn `ALL IMAGE/MASK TESTS PASSED!` (dữ liệu sinh lại vẫn đúng chuẩn).

- [x] **Step 4: Kiểm tra git sạch**

Run: `git status --short` và `git log --oneline -5`
Expected: không có thay đổi chưa commit; 4 commit mới trên nhánh `trinhson`.
