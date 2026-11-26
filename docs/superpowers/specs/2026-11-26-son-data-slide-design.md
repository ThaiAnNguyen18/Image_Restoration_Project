# Thiết kế: Nhiệm vụ Sơn — Dữ liệu & Slide (Tuần 2)

**Ngày:** 2026-11-26 · **Nhánh:** `trinhson` · **Trạng thái:** chờ duyệt

## 1. Mục tiêu (intent & why)
Hoàn thành 2 nhiệm vụ được giao trong hôm nay:
1. Chuẩn bị 2 ảnh PNG 256×256 (Ảnh 1 mượt, Ảnh 2 nhiều cạnh), viết `data/SOURCES.md`, kiểm chứng đọc ảnh sang `float64`.
2. Hoàn thiện `src/image_processing.py` (3 hàm) và `src/mask.py` (2 hàm); tạo & lưu cố định 3 file mask cho TN1.

## 2. Mục tiêu / Không phải mục tiêu
**Làm:**
- 5 hàm trong 2 module của Sơn.
- Sinh 2 ảnh PNG tổng hợp + 3 mask `.npy` bằng 1 script tái lập được.
- Viết `data/SOURCES.md`.
- Tự kiểm chứng dữ liệu & phần code của Sơn.

**Không làm (lần này):**
- Notebook `experiments/`, module của An/Giáp/Hiếu, slide — việc khác.
- **Nhánh B** (tải ảnh thật từ nguồn công khai) — để phát triển sau, `SOURCES.md` sẽ được bổ sung khi làm.

## 3. Ràng buộc
- Quy ước README: mask `0`/`1`; ảnh tính toán `float64` miền `[0, 255]`; thứ tự ẩn số = `np.argwhere(mask == 1)`; không ghi đè ảnh gốc (dùng `.copy()`).
- Bảng thiết kế TN1 (nhánh `zap`): mask `2x50`, `10x10`, `30x30`; lưu dạng `.npy` **cố định trước khi chạy**; TN1 dùng Ảnh 1 (mượt), validation dùng Ảnh 2 (nhiều cạnh).
- Test `tests/test_2x2.py` gọi `create_square_mask(4x4, row=1, col=1, size=2)` và `restore_pixels` — implementation phải thỏa test này.
- Môi trường: tạo `.venv` theo README, cài `requirements.txt` (hiện thiếu `scipy`).

## 4. Các hướng đã cân nhắc (Approaches)
| Hạng mục | Đã chọn | Bỏ / để sau |
| :--- | :--- | :--- |
| Nguồn 2 ảnh | **A: tự sinh bằng script** — tái lập được, không bản quyền, offline | B: tải ảnh thật (Wikimedia/skimage) — phát triển sau |
| Định dạng mask | **`.npy`** (đã chốt trong bảng thiết kế zap) | PNG mask — chỉ để minh họa, làm sau nếu cần cho slide |
| Vị trí 3 mask TN1 | **Căn giữa ảnh 256×256** — trực quan trên slide, đổi vị trí chỉ cần sửa tham số | Đặt lệch/đối xứng trục khác — không cần thiết |
| Script dữ liệu | **`scripts/generate_data.py`** chạy 1 lần sinh cả ảnh lẫn mask | Sinh inline trong notebook — khó tái lập, khó trích dẫn |

## 5. Thiết kế chi tiết

### 5.1 `src/image_processing.py`
- `load_grayscale(path)`: `Image.open(path).convert("L")` → `np.asarray(..., dtype=np.float64)` → trả mảng 2D, miền `[0, 255]`.
- `restore_pixels(image, mask, unknown_pixels, x)`: `out = image.copy()`; điền `x` vào đúng tọa độ `unknown_pixels` (thứ tự `np.argwhere(mask == 1)`), giữ nguyên mọi pixel ngoài mask.
- `save_image(image, path)`: `np.clip` về `[0, 255]` → làm tròn → `astype(np.uint8)` → `Image.fromarray(...).save(path)` PNG.

### 5.2 `src/mask.py`
- `create_square_mask(shape, row, col, size)`: mask `uint8` 0/1, vuông `size×size` bắt đầu tại `(row, col)`, bắt buộc cách mép ảnh ≥ 1 pixel, ném `ValueError` nếu vi phạm.
- `create_scratch_mask(shape, row, col, height, width)`: tương tự với hình chữ nhật `height×width`.

### 5.3 `scripts/generate_data.py` (tái lập được, chạy 1 lần)
- **Ảnh 1 — mượt** `data/original/image1_smooth_256.png`: hàm liên tục mượt (gradient sin/kosông có biên độ lớn), không có bất kỳ rìa sắc nét nào.
- **Ảnh 2 — nhiều cạnh** `data/original/image2_edges_256.png`: bàn cờ tương phản cao + vài hình hình học (chữ nhật, vòng tròn, đường chéo) để tăng mật độ rìa.
- **3 mask TN1** → `data/masks/`:
  - `mask_2x50.npy` — vết xước `2×50` căn giữa: `(row=127, col=103)` → 100 pixel khuyết.
  - `mask_10x10.npy` — vuông căn giữa: `(row=123, col=123)` → 100 pixel khuyết.
  - `mask_30x30.npy` — vuông căn giữa: `(row=113, col=113)` → 900 pixel khuyết.
- Script in ra tọa độ, kích thước, số pixel `1` của từng mask để dễ đối chiếu.

### 5.4 `data/SOURCES.md`
- Nguồn 2 ảnh: **tự sinh** bởi `scripts/generate_data.py` (nhánh A), ngày tạo, thông số kỹ thuật (256×256, grayscale, PNG 8-bit) — không dùng tài nguyên bên thứ ba nên không có giới hạn bản quyền.
- Nguồn mask: sinh cố định từ cùng script, dạng `.npy`, vị trí/kích thước như §5.3.
- Ghi chú nhánh B (ảnh thật công khai) sẽ được bổ sung khi thực hiện.

### 5.5 Kiểm chứng (mục tiêu hoàn thành hôm nay)
1. `.venv` + `pip install -r requirements.txt` theo README.
2. Chạy `scripts/generate_data.py` → sinh dữ liệu.
3. Self-check: 2 ảnh load về đúng `float64`, shape `(256, 256)`, miền `[0, 255]`; mask `uint8` 0/1 đúng 100/100/900 pixel; `restore_pixels` giữ nguyên vùng ngoài mask.
4. Chạy `python -m tests.test_2x2` — **dự kiến vẫn lỗi `NotImplementedError` ở module An/Giáp/Hiếu** (đúng phạm vi); phần của Sơn được kiểm bằng self-check ở bước 3. Test đủ `ALL TESTS PASSED!` khi đủ 4 người.
5. Commit vào nhánh `trinhson`.

## 6. Còn lại cho người thực hiện
- Chi tiết thẩm mỹ khi sinh ảnh (bước sóng gradient, kiểu bàn cờ, hình trang trí) — quyết định tại chỗ khi code, không ảnh hưởng yêu cầu.
