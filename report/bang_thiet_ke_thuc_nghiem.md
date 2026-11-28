# BẢNG THIẾT KẾ PHƯƠNG ÁN THỰC NGHIỆM (MỐC BẮT BUỘC TUẦN 2)
**Học phần: Phương pháp tính (Hướng B)**  
**Nhóm sinh viên: An (Nhóm trưởng), Giáp, Sơn, Hiếu**  
**Người phụ trách chính TN1 & TN3: Giáp | Phụ trách TN2: Hiếu**

---

## 1. Mục tiêu và Câu hỏi nghiên cứu thực nghiệm
- **Câu hỏi nghiên cứu:** Hình dạng/kích thước vùng khuyết, phương thức khởi tạo $x^{(0)}$ và ngưỡng dừng $TOL$ ảnh hưởng thế nào đến tốc độ hội tụ, chi phí tính toán và chất lượng ảnh khôi phục của phương pháp Gauss–Seidel trên ma trận thưa CSR?
- **Phương pháp số cốt lõi:** Phương pháp lặp Gauss–Seidel tự cài đặt trên định dạng CSR.
- **Nghiệm đối chứng:** Nghiệm trực tiếp từ `scipy.sparse.linalg.spsolve`.

---

## 2. Thiết kế chi tiết 3 Kịch bản Thực nghiệm

| Thí nghiệm | Mục đích khảo sát | Biến độc lập (Khảo sát) | Tham số cố định (Kiểm soát) | Chỉ số đo lường (Đầu ra) | Giả thuyết đối chiếu lý thuyết |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TN1: Hình dạng & Kích thước vùng khuyết** *(Giáp chủ trì)* | Đánh giá ảnh hưởng của hình học miền khuyết đến tốc độ hội tụ | 3 loại mask:<br>1. Vết xước mảnh $2 \times 50$ ($n=100$)<br>2. Hình vuông $10 \times 10$ ($n=100$)<br>3. Hình vuông $30 \times 30$ ($n=900$) | - Ảnh: Ảnh 1 (nền mượt)<br>- Khởi tạo: $x^{(0)} = \text{boundary\_mean}$<br>- $TOL = 10^{-4}$<br>- $max\_iter = 10000$ | - Số vòng lặp hội tụ<br>- Thời gian (trung vị 5 lần đo)<br>- Phần dư tương đối $r_{rel}$<br>- $MSE_{mask}$, $PSNR_{mask}$ | - Mask $2 \times 50$ hội tụ nhanh hơn $10 \times 10$ vì 100% pixel chạm biên đã biết.<br>- Mask $30 \times 30$ hội tụ chậm hơn rõ rệt vì bán kính phổ $\rho(T_{GS}) = \cos^2\left(\frac{\pi}{m+1}\right)$ tiến gần 1 khi $m$ tăng. |
| **TN2: Phương thức khởi tạo vector $x^{(0)}$** *(Hiếu chủ trì)* | Khảo sát vai trò của điểm xuất phát đến số vòng lặp | 3 chế độ khởi tạo:<br>1. `zero` (vector 0)<br>2. `random` (seed=42, miền $[0, 255]$)<br>3. `boundary_mean` (trung bình pixel biên đã biết) | - Ảnh: Ảnh 1<br>- Mask: Vuông $10 \times 10$<br>- $TOL = 10^{-4}$<br>- $max\_iter = 10000$ | - Số vòng lặp hội tụ<br>- Lịch sử hội tụ phần dư qua từng vòng lặp<br>- Sai số với nghiệm chuẩn: $\|x - x_{ref}\|_{\infty}$ | - Cả 3 cách chắc chắn hội tụ về cùng 1 nghiệm duy nhất do ma trận $A$ đối xứng xác định dương.<br>- `boundary_mean` hội tụ nhanh nhất do khoảng cách ban đầu $\|x^{(0)} - x^*\|$ nhỏ nhất. |
| **TN3: Ngưỡng dung sai dừng $TOL$** *(Giáp chủ trì)* | Khảo sát sự đánh đổi giữa chi phí tính toán và chất lượng ảnh | 4 mức dung sai:<br>$TOL \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$ | - Ảnh: Ảnh 1<br>- Mask: Vuông $10 \times 10$<br>- Khởi tạo: $x^{(0)} = \text{boundary\_mean}$<br>- $max\_iter = 10000$ | - Số vòng lặp hội tụ<br>- Sai số giải hệ $e_{ref} = \|x - x_{ref}\|_{\infty}$<br>- $MSE_{mask}$, $PSNR_{mask}$ | - Số vòng lặp tăng tuyến tính theo $-\log_{10}(TOL)$.<br>- $MSE_{mask}$ và $PSNR_{mask}$ sẽ bão hòa ở một ngưỡng nhất định vì chạm giới hạn sai số bản chất của mô hình Laplace 4 lân cận. |

---

## 3. Thí nghiệm kiểm chứng bổ sung (Validation on Image 2)
- **Cấu hình chuẩn:** Chạy lại cấu hình chuẩn (Mask $10 \times 10$, $x^{(0)} = \text{boundary\_mean}$, $TOL = 10^{-4}$) trên **Ảnh 2 (ảnh có nhiều cạnh sắc nét)**.
- **Giả thuyết kiểm chứng:** Tốc độ hội tụ đại số (số vòng lặp) trên Ảnh 2 sẽ tương đương Ảnh 1 vì ma trận $A$ chỉ phụ thuộc hình học mask, nhưng $PSNR_{mask}$ trên vùng cạnh sắc nét sẽ thấp hơn vùng mượt do tính chất làm phẳng tự nhiên của phương trình Laplace.

---

## 4. Quy ước kỹ thuật và Kiểm soát sai số
1. **Đo thời gian:** Chỉ dùng `time.perf_counter()` bọc quanh hàm `gauss_seidel`, thực hiện 5 lần liên tiếp trên cùng dữ liệu và lấy giá trị **trung vị (median)**.
2. **Không ghi đè ảnh gốc:** Bắt buộc sử dụng `.copy()` trước khi điền nghiệm tái tạo.
3. **Phạm vi đánh giá:** Chỉ tính $MSE$ và $PSNR$ trên tập hợp các pixel bị khuyết ($mask == 1$).
4. **Cố định tái lập:** Cố định `seed=42` và lưu file mask chuẩn `.npy` trước khi chạy.
