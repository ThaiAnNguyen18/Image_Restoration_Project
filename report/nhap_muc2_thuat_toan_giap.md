# BẢN NHÁP MỤC 2.2: THUẬT TOÁN GAUSS-SEIDEL TRÊN MA TRẬN THƯA CSR
**Người thực hiện: Giáp (Phụ trách Thuật toán số, TN1 & TN3)**

---

## 2.2. Phương pháp giải hệ phương trình: Lặp Gauss–Seidel trên CSR

### 1. Nguyên lý thuật toán Gauss–Seidel
Sau khi rời rạc hóa bài toán khôi phục ảnh bằng phương trình Laplace 5 điểm, ta thu được hệ phương trình đại số tuyến tính có dạng:
$$A x = b$$
trong đó $n$ là tổng số pixel bị khuyết ($n = |\Omega|$), $A \in \mathbb{R}^{n \times n}$ là ma trận hệ số và $b \in \mathbb{R}^n$ là vector vế phải chứa tổng độ sáng các pixel lân cận đã biết.

Khác với phương pháp Jacobi (chỉ sử dụng nghiệm ở vòng lặp cũ $x^{(k)}$ để tính toàn bộ vector mới), phương pháp Gauss–Seidel cập nhật tuần tự từng ẩn $x_i$ và **sử dụng ngay lập tức các giá trị $x_j^{(k+1)}$ $(j < i)$ vừa tính được trong cùng vòng lặp**. Công thức truy hồi tại vòng lặp thứ $k+1$ được biểu diễn như sau:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right), \quad \forall i = 0, 1, \dots, n-1$$

Trong bài toán Laplace của chúng ta, các phần tử đường chéo chính luôn cố định: $a_{ii} = 4$.

---

### 2. Cơ sở toán học đảm bảo hội tụ duy nhất
Để đảm bảo thuật toán không bị phân kỳ, ma trận $A$ cần thỏa mãn các điều kiện lý thuyết của giải tích số:
1. **Tính đối xứng:** Do mối quan hệ tương hỗ giữa hai pixel lân cận bất kỳ $i$ và $j$ đều nhận trọng số liên kết là $-1$, ma trận $A$ thỏa mãn $A = A^T$.
2. **Tính xác định dương:** Vì mọi thành phần liên thông của vùng khuyết đều tiếp xúc ít nhất với một pixel nguyên vẹn ở biên (đóng vai trò điều kiện biên Dirichlet), toán tử vi phân Laplace rời rạc không có trị riêng bằng $0$. Mọi trị riêng của $A$ đều thực và dương ($\lambda_i > 0$).
3. **Tính chéo trội:**
   - Tại các pixel khuyết nằm sâu bên trong (có đủ 4 lân cận khuyết): $|a_{ii}| = 4 = \sum_{j \ne i} |a_{ij}| = 4$ (chéo trội yếu).
   - Tại các pixel khuyết tiếp giáp biên đã biết (có ít nhất 1 lân cận nguyên vẹn): $|a_{ii}| = 4 > \sum_{j \ne i} |a_{ij}| \le 3$ (chéo trội ngặt).
   - Do đồ thị liên thông giữa các pixel khuyết luôn dẫn ra biên, ma trận $A$ là ma trận chéo trội không khả quy (irreducibly diagonally dominant).

**Định lý hội tụ:** Khi ma trận $A$ đối xứng và xác định dương (Symmetric Positive Definite - SPD), phương pháp lặp Gauss–Seidel **chắc chắn hội tụ về nghiệm giải tích duy nhất** $x^* = A^{-1}b$ với mọi vector khởi tạo ban đầu $x^{(0)} \in \mathbb{R}^n$.

---

### 3. Tối ưu hóa bộ nhớ và thời gian bằng định dạng CSR (Compressed Sparse Row)
Trong bài toán thực tế:
- Với vùng khuyết kích thước $30 \times 30$, số lượng ẩn số là $n = 900$. Ma trận đầy đủ $A$ có kích thước $900 \times 900 = 810.000$ phần tử.
- Tuy nhiên, mỗi pixel chỉ có tối đa 4 lân cận trực tiếp, nghĩa là mỗi hàng của $A$ chỉ có tối đa 5 phần tử khác không ($1$ số $4$ và tối đa bốn số $-1$). Tổng số phần tử khác không tối đa chỉ là $900 \times 5 = 4.500$ (chiếm chưa đầy $0.6\%$ ma trận, hơn $99.4\%$ còn lại là số $0$).

Thay vì lưu trữ toàn bộ $n^2$ phần tử, nhóm sử dụng định dạng **Compressed Sparse Row (CSR)** của thư viện `scipy.sparse.csr_matrix` bao gồm 3 mảng 1 chiều:
1. `data`: Mảng chứa các giá trị khác không liên tiếp theo từng hàng.
2. `indices`: Mảng chứa chỉ số cột tương ứng của từng phần tử trong `data`.
3. `indptr`: Mảng con trỏ đánh dấu vị trí bắt đầu và kết thúc của mỗi hàng $i$ trong `data`.

**Lợi thế về độ phức tạp tính toán:**
Trong vòng lặp cập nhật, để tính $\sum_{j \ne i} a_{ij} x_j$, thuật toán chỉ cần duyệt các chỉ số `idx` từ `indptr[i]` đến `indptr[i+1]` (tối đa 5 phần tử) thay vì duyệt toàn bộ $n$ cột. Do đó:
- Độ phức tạp duyệt mỗi hàng: $O(1)$.
- Độ phức tạp mỗi vòng lặp toàn hệ: $O(n)$ thay vì $O(n^2)$ như ma trận đặc thông thường.
- Tiết kiệm bộ nhớ từ $O(n^2)$ xuống $O(n)$.

---

### 4. Tiêu chuẩn dừng thuật toán (Stopping Criteria)
Thuật toán kết thúc vòng lặp khi thỏa mãn một trong hai điều kiện:
1. **Hội tụ:** Phần dư tương đối theo chuẩn vô cùng đạt ngưỡng dung sai:
   $$r_{rel}^{(k)} = \frac{\|b - A x^{(k)}\|_{\infty}}{\max(\|b\|_{\infty}, 10^{-12})} \le TOL$$
   *(Trong đó $10^{-12}$ là hệ số an toàn chống chia cho 0 khi $b = 0$).*
2. **Chạm giới hạn vòng lặp an toàn:** Đạt $max\_iter = 10000$ vòng lặp (nhằm tránh tình trạng vòng lặp vô hạn khi thử nghiệm với các vùng khuyết kích thước lớn hoặc ngưỡng $TOL$ quá khắt khe).
