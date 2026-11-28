# BẢN NHÁP MỤC 1: ĐẶT VẤN ĐỀ, MỤC TIÊU & PHẠM VI MÔ HÌNH
**Người thực hiện: An (Nhóm trưởng — Mô hình & Tích hợp)**

---

## 1.1. Bối cảnh bài toán thực tế và Giới hạn tự nhiên của mô hình
Trong xử lý ảnh số, một bức ảnh grayscale kích thước $256 \times 256$ (mức xám thuộc đoạn $[0.0, 255.0]$) có thể bị trầy xước hoặc mất dữ liệu tại một vùng cục bộ $\Omega$. Để khôi phục giá trị độ sáng của các điểm ảnh (pixel) bị khuyết mà chỉ sử dụng thông tin từ các pixel nguyên vẹn bao quanh, nhóm áp dụng quy tắc nội suy địa phương: độ sáng tại mỗi pixel khuyết $(i, j) \in \Omega$ bằng trung bình cộng độ sáng của 4 pixel lân cận trực tiếp (trên, dưới, trái, phải):

$$u(i,j) = \frac{u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1)}{4}$$

**Giới hạn tự nhiên của mô hình toán học:** Quy tắc trung bình 4 lân cận tương ứng với phương trình Laplace rời rạc trên lưới hai chiều với điều kiện biên Dirichlet là các pixel nguyên vẹn bao quanh. Theo nguyên lý cực đại – cực tiểu rời rạc, nghiệm nội suy luôn đạt giá trị lớn nhất và nhỏ nhất trên biên vùng khuyết và không thể tự sinh cực trị độ sáng mới bên trong vùng khuyết. Hệ quả là mô hình tạo ra vùng chuyển tiếp mức xám mượt mà nhưng sẽ làm mờ các cạnh sắc nét, chữ viết hoặc chi tiết phức tạp đã mất. Đây là giới hạn bản chất của mô hình toán học, không phải lỗi của thuật toán giải hệ Gauss–Seidel.

---

## 1.2. Câu hỏi nghiên cứu và Ba mục tiêu cụ thể
**Câu hỏi nghiên cứu chính:** Khi mô hình hóa bài toán khôi phục vùng ảnh grayscale bị khuyết thành hệ phương trình tuyến tính thưa $Ax = b$ dựa trên quy tắc trung bình cộng 4 pixel lân cận, việc thay đổi hình dạng/kích thước vùng khuyết, giá trị khởi tạo $x^{(0)}$ và ngưỡng dừng $TOL$ ảnh hưởng như thế nào đến tốc độ hội tụ, chi phí tính toán của phương pháp Gauss–Seidel và chất lượng ảnh khôi phục?

**Ba mục tiêu cụ thể:**
1. Mô hình hóa bài toán nội suy vùng ảnh khuyết thành hệ phương trình tuyến tính thưa $Ax = b$ lưu trữ theo cấu trúc ma trận thưa CSR (`Compressed Sparse Row`).
2. Tự cài đặt bộ giải lặp Gauss–Seidel trực tiếp trên cấu trúc CSR, xây dựng các chế độ khởi tạo $x^{(0)}$ và kiểm chứng độ chính xác thông qua bộ test giải tay $2 \times 2$ cùng nghiệm tham chiếu `spsolve`.
3. Triển khai 3 kịch bản thực nghiệm có chủ đích (**TN1, TN2, TN3**) cùng kiểm chứng bổ sung trên Ảnh 2 (nhiều cạnh sắc nét) để đối chiếu thực nghiệm với lý thuyết hội tụ.

---

## 1.3. Phân định phần tự cài đặt và phần sử dụng thư viện có sẵn
Nhằm đảm bảo tính minh bạch về thuật toán số trong học phần Phương pháp tính, nhóm phân tách rõ hai nhóm thành phần sau:

| Thành phần trong dự án | Cách thực hiện | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| Lắp ráp ma trận thưa $A$, vector vế phải $b$, danh sách tọa độ `unknown_pixels` | **Tự cài đặt hoàn toàn** | `src/laplace_model.py` (`build_system`) |
| Khởi tạo vector ban đầu $x^{(0)}$ (`zero`, `random`, `boundary_mean`) | **Tự cài đặt hoàn toàn** | `src/laplace_model.py` (`make_initial_guess`, `seed=42`) |
| Thuật toán lặp Gauss–Seidel trực tiếp trên mảng CSR (`indptr`, `indices`, `data`) | **Tự cài đặt hoàn toàn** | `src/gauss_seidel.py` (`gauss_seidel`) |
| Các chỉ số đánh giá ($r_{rel}$, $e_{ref}$, $MSE_{mask}$, $PSNR_{mask}$) và đo thời gian | **Tự cài đặt hoàn toàn** | `src/evaluation.py` (chỉ tính MSE/PSNR trên `mask == 1`) |
| Đọc/lưu ảnh PNG, chuyển grayscale và thao tác mảng số thực | **Thư viện có sẵn** | `Pillow`, `NumPy` (`src/image_processing.py`, `src/mask.py`) |
| Cấu trúc lưu trữ CSR và nghiệm tham chiếu đối chứng | **Thư viện có sẵn** | `SciPy` (`csr_matrix`, `spsolve` — chỉ dùng để kiểm chứng) |
| Điều phối thực nghiệm và trực quan hóa đồ thị | **Thư viện có sẵn** | `Jupyter Notebook`, `Matplotlib` |
