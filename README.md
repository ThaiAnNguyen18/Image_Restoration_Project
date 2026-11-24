# Khôi Phục Vùng Ảnh Grayscale Bị Khuyết (Phương Pháp Số)

Dự án học phần **Phương pháp tính** (Hướng B) — Nhóm 4 sinh viên: **An (Nhóm trưởng), Giáp, Sơn, Hiếu**.

## 1. Giới thiệu
Mô hình hóa bài toán khôi phục vùng ảnh xám $256 \times 256$ bị khuyết bằng quy tắc trung bình 4 pixel lân cận (phương trình Laplace rời rạc) thành hệ phương trình tuyến tính thưa $Ax = b$ ở định dạng CSR, tự cài đặt bộ giải lặp **Gauss-Seidel** và khảo sát thực nghiệm theo 3 kịch bản:
- **TN1:** Ảnh hưởng của hình dạng & kích thước vùng khuyết (`2x50`, `10x10`, `30x30`).
- **TN2:** Ảnh hưởng của vector khởi tạo $x^{(0)}$ (`zero`, `random` với `seed=42`, `boundary_mean`).
- **TN3:** Ảnh hưởng của ngưỡng dừng `TOL` ($\{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$).

## 2. Quy ước dùng chung (Bắt buộc)
- **Mask:** `0` = pixel đã biết (nguyên vẹn), `1` = pixel bị khuyết (ẩn số).
- **Ảnh tính toán:** `float64` miền `[0.0, 255.0]`; chỉ ép kiểu `uint8` khi lưu/hiển thị.
- **Thứ tự ẩn số:** Dùng duy nhất `unknown_pixels = np.argwhere(mask == 1)` từ `build_system`.
- **Chỉ số chất lượng:** `MSE` và `PSNR` chỉ tính trên tập pixel bị khuyết (`mask == 1`).

## 3. Cài đặt môi trường

    python -m venv .venv
    # Windows (Git Bash):
    source .venv/Scripts/activate
    # Windows (PowerShell):
    .\.venv\Scripts\Activate.ps1

    pip install -r requirements.txt

## 4. Chạy kiểm thử tự động (Bộ test 2x2)
Trước khi tạo Pull Request vào `main`, chạy lệnh sau từ thư mục gốc dự án:

    python -m tests.test_2x2

Kết quả đạt khi màn hình in ra `ALL TESTS PASSED!`.

## 5. Chạy lại toàn bộ thực nghiệm
Mở Jupyter Notebook và chọn **Kernel -> Restart & Run All**:

    jupyter notebook experiments/experiments.ipynb
