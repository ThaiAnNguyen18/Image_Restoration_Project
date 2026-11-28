image_processing.py: Đọc ảnh grayscale float64, ghép nghiệm an toàn trên bản sao copy(), và lưu ảnh PNG uint8.
mask.py: Sinh mặt nạ vuông và vết xước chữ nhật (0 = đã biết, 1 = bị khuyết) có kiểm tra biên cách mép >= 1 pixel.
laplace_model.py: Trích xuất unknown_pixels, lắp ráp ma trận thưa A (CSR) và vector b, khởi tạo x0 và giải nghiệm tham chiếu spsolve.
gauss_seidel.py: Cài đặt thuật toán lặp Gauss-Seidel trực tiếp trên mảng CSR (indptr, indices, data).
evaluation.py: Tính phần dư tương đối, sai số so với spsolve, MSE_mask/PSNR_mask trên vùng khuyết và benchmark thời gian trung vị.
