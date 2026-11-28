# BẢNG THIẾT KẾ PHƯƠNG ÁN THỰC NGHIỆM (MỐC BẮT BUỘC TUẦN 2)
**Học phần:** Phương pháp tính (Hướng B)  
**Nhóm sinh viên:** An (Nhóm trưởng), Giáp, Sơn, Hiếu  
**Phụ trách thực nghiệm:** Giáp (TN1, TN3) | Hiếu (TN2, điều phối Notebook) | Sơn (Dữ liệu, Ảnh 2) | An (Kiểm soát mô hình & rà soát chéo)

---

## 1. Câu hỏi nghiên cứu và Phương pháp đối chứng
- **Câu hỏi nghiên cứu:** Hình dạng/kích thước vùng khuyết, phương thức khởi tạo $x^{(0)}$ và ngưỡng dừng $TOL$ ảnh hưởng thế nào đến tốc độ hội tụ, chi phí tính toán và chất lượng ảnh khôi phục của phương pháp Gauss–Seidel trên ma trận thưa CSR?
- **Bộ giải nghiên cứu chính:** Phương pháp lặp Gauss–Seidel tự cài đặt trực tiếp trên định dạng CSR (`src/gauss_seidel.py`).
- **Nghiệm tham chiếu kiểm chứng:** `scipy.sparse.linalg.spsolve` (chỉ dùng để tính sai số đại số $e_{ref} = \|x^{(k)} - x_{ref}\|_\infty$, không phải thuật toán nghiên cứu thứ hai).

---

## 2. Thiết kế chi tiết 3 Kịch bản Thực nghiệm

| Thí nghiệm | Biến khảo sát & Tham số cố định | Chỉ số thu thập (xuất CSV) | Hệ quả lý thuyết (Chắc chắn) | Giả thuyết thực nghiệm (Chờ số liệu kiểm chứng) |
| :--- | :--- | :--- | :--- | :--- |
| **TN1: Hình dạng & Kích thước vùng khuyết** *(Giáp chủ trì)* | **Biến đổi:** 3 mask cố định `.npy`: vết xước $2 \times 50$ ($n=100$), vuông $10 \times 10$ ($n=100$), vuông $30 \times 30$ ($n=900$).<br>**Cố định:** Ảnh 1 (`image1_smooth.png`), `boundary_mean`, $TOL = 10^{-4}$, `max_iter = 10000`. | Số vòng lặp, thời gian chạy (trung vị 5 lần), $r_{rel}$, $e_{ref}$, $MSE_{mask}$, $PSNR_{mask}$. | Ở mask $2 \times 50$, 100% pixel khuyết chạm biên đã biết. Với lưới vuông $m \times m$, bán kính phổ $\rho(T_{GS}) = \cos^2\left(\frac{\pi}{m+1}\right)$ tiến sát 1 khi $m$ tăng từ 10 lên 30. | **Dự kiến** mask mảnh $2 \times 50$ hội tụ nhanh hơn đáng kể so với mask vuông $10 \times 10$ cùng diện tích, và vùng vuông $30 \times 30$ chậm hơn rõ rệt so với $10 \times 10$; mức chênh lệch cụ thể do số liệu thực tế quyết định. |
| **TN2: Giá trị khởi tạo $x^{(0)}$** *(Hiếu chủ trì)* | **Biến đổi:** `zero`, `random` (`seed=42`, miền $[0, 255]$), `boundary_mean`.<br>**Cố định:** Ảnh 1, mask vuông $10 \times 10$, $TOL = 10^{-4}$, `max_iter = 10000`. | Số vòng lặp, thời gian chạy (trung vị 5 lần), $r_{rel}$, $e_{ref}$, $MSE_{mask}$, $PSNR_{mask}$, lịch sử phần dư theo vòng lặp. | Vì $A$ đối xứng xác định dương nên cả 3 phương thức khởi tạo chắc chắn hội tụ về cùng một nghiệm duy nhất $x^*$. | **Dự kiến** `boundary_mean` cần ít vòng lặp hơn nhờ khoảng cách ban đầu $\|x^{(0)} - x^*\|$ nhỏ hơn; mức độ rút ngắn cụ thể được kiểm chứng bằng số liệu chạy thực tế. |
| **TN3: Ngưỡng dừng $TOL$** *(Giáp chủ trì)* | **Biến đổi:** $TOL \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$ (truyền tường minh `tol=current_tol`).<br>**Cố định:** Ảnh 1, mask vuông $10 \times 10$, `boundary_mean`, `max_iter = 10000`. | Số vòng lặp, thời gian chạy (trung vị 5 lần), $r_{rel}$, $e_{ref}$, $MSE_{mask}$, $PSNR_{mask}$. *(Kiểm tra thêm mask $30 \times 30$ ở $TOL = 10^{-5}$).* | Khi giảm $TOL$, số vòng lặp tăng tuyến tính theo $-\log_{10}(TOL)$ và sai số đại số $e_{ref}$ giảm đơn điệu. | **Dự kiến** $MSE_{mask}$ và $PSNR_{mask}$ sẽ bão hòa từ một ngưỡng $TOL$ nhất định vì chạm giới hạn sai số của mô hình nội suy 4 lân cận; mốc bão hòa cụ thể để số liệu thực tế trả lời. |

---

## 3. Kiểm chứng bổ sung trên Ảnh 2 (nhiều cạnh sắc nét)
- **Cấu hình chuẩn:** Mask vuông $10 \times 10$ (`mask_sq_10x10.npy`), `x0 = boundary_mean`, $TOL = 10^{-4}$, `max_iter = 10000` trên `image2_edges.png`, đối chiếu trực tiếp với Ảnh 1.
- **Hệ quả lý thuyết:** Ma trận $A$ chỉ phụ thuộc hình học của mask nên tốc độ hội tụ đại số không phụ thuộc nội dung ảnh.
- **Giả thuyết thực nghiệm:** **Dự kiến** số vòng lặp trên Ảnh 2 tương đương Ảnh 1, nhưng $PSNR_{mask}$ trên vùng có cạnh sắc nét sẽ thấp hơn vùng nền mượt do tính chất làm trơn của mô hình Laplace.

---

## 4. Quy ước kỹ thuật và Kiểm soát sai số
1. **Đo thời gian:** Dùng `time.perf_counter()` chỉ bọc riêng hàm `gauss_seidel`, chạy lặp 5 lần trên cùng dữ liệu/cùng $x^{(0)}$ và lấy **trung vị (median)**.
2. **Không ghi đè ảnh gốc:** Hàm `restore_pixels` bắt buộc tạo bản sao `float64` trước khi điền nghiệm.
3. **Phạm vi đánh giá ảnh:** Chỉ tính $MSE_{mask}$ và $PSNR_{mask}$ trên vùng pixel bị khuyết (`mask == 1`) trực tiếp trên mảng `float64` trước khi ép `uint8`.
4. **Cố định tái lập:** Cố định `seed = 42` và lưu cố định các ma trận mask `.npy` trong `data/masks/` trước khi chạy thực nghiệm.

---

## 5. Thông số môi trường thực nghiệm (Ghi nhận chuẩn hóa)
*(Cập nhật lại nếu ở Tuần 4 nhóm chốt chạy số liệu chính thức trên máy khác)*

| Hạng mục | Thông số ghi nhận |
| :--- | :--- |
| Hệ điều hành | Windows 11 (AMD64) |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel (16 luồng) |
| RAM | 15.69 GB |
| Python | 3.12.10 |
| NumPy / SciPy | 2.5.5 / 1.18.1 |
| Pillow / Matplotlib | 13.0.0 / 3.11.2 |
