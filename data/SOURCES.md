# NGUỒN DỮ LIỆU (SOURCES)

## 1. Ảnh gốc (`data/original/`)

| File | Nguồn | Ngày tạo | Thông số |
| :--- | :--- | :--- | :--- |
| `image1_smooth_256.png` | **Tự sinh** bởi `scripts/generate_data.py` (hàm `make_smooth_image`: hàm `127.5 * (1 + sin(2πX/160) * cos(2πY/160))`) | 2026-11-26 | 256×256, grayscale, PNG 8-bit |
| `image2_edges_256.png` | **Tự sinh** bởi `scripts/generate_data.py` (hàm `make_edges_image`: bàn cờ 16 px + hình chữ nhật/vòng tròn rỗng + đường chéo) | 2026-11-26 | 256×256, grayscale, PNG 8-bit |

- Cả 2 ảnh **không lấy từ tài nguyên bên thứ ba** → không có giới hạn bản quyền.
- Ảnh 1 dùng cho TN1/TN2/TN3 (nền mượt); ảnh 2 dùng cho phần validation (nhiều cạnh sắc nét).
- Chạy lại script là tái tạo đúng 2 file trên: `.venv/Scripts/python.exe scripts/generate_data.py`.

## 2. Mask cố định (`data/masks/`)

| File | Hình dạng | Tọa độ gốc (row, col) | Số pixel khuyết | Dạng |
| :--- | :--- | :--- | :--- | :--- |
| `mask_2x50.npy` | Vết xước 2×50 | (127, 103) | 100 | `.npy`, `uint8` 0/1, shape (256, 256) |
| `mask_10x10.npy` | Vuông 10×10 | (123, 123) | 100 | `.npy`, `uint8` 0/1, shape (256, 256) |
| `mask_30x30.npy` | Vuông 30×30 | (113, 113) | 900 | `.npy`, `uint8` 0/1, shape (256, 256) |

- Quy ước: `0` = pixel đã biết, `1` = pixel bị khuyết; cả 3 mask căn giữa ảnh 256×256, cách mép ≥ 1 pixel.
- Sinh cố định bởi `scripts/generate_data.py` (dùng `create_scratch_mask` / `create_square_mask` từ `src/mask.py`), **lưu trước khi chạy thí nghiệm** (bảng thiết kế TN1, quy ước "cố định tái lập").
- 3 mask này phục vụ **TN1** (ảnh hưởng hình dạng & kích thước vùng khuyết).

## 3. Ghi chú phát triển sau (Nhánh B)

- Hiện tại (nhánh A) dùng ảnh **tự sinh** để đảm bảo tái lập được và không ràng buộc bản quyền.
- **Nhánh B** (ảnh thật từ nguồn công khai như Wikimedia Commons / mẫu scikit-image) sẽ được bổ sung vào file này: URL nguồn, license, ngày tải, và cách cắt/xuống đúng 256×256.
