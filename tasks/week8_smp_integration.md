# Week 8 Task Assignment — Tích hợp MathIR vào `smp-math.com`

**Branch tích hợp:** `feat/hybrid-integration`  
**Mục tiêu:** Đưa hệ thống Hybrid MathIR 3 tầng vào tìm kiếm thực tế trên kho đề của `smp-math.com`.

---

## Phân công thành viên

### 1. Trần Đức Toàn (`25125091`) — Export & Chuẩn hóa Kho đề `smp-math.com`
- [ ] Export dữ liệu đề thi/bài toán từ database của `smp-math.com` ra file `corpus_formulas.jsonl` theo định dạng `{"doc_id": "...", "latex": "..."}`.
- [ ] Lọc bỏ nhiễu HTML/LaTeX hỏng và chạy thử tạo `corpus_vecs.npy` + FAISS index cho kho đề mới.

### 2. Phạm Trung Duy (`25125010`) — Tối ưu Truy vấn Tiếng Việt & Giao diện Tìm kiếm
- [ ] Bổ sung thêm các từ khóa/thuật ngữ toán học phổ biến trên `smp-math.com` vào `src/preprocessing/vi_normalizer.py`.
- [ ] Phối hợp nối thanh tìm kiếm trên frontend của `smp-math.com` gọi API `POST /search` (`scripts/api_server.py`) và hiển thị danh sách kết quả + tag lý do.

### 3. Nguyễn Nguyễn Gia Bảo (`25125006`) — Kiểm thử Thực tế & Đo Độ trễ API
- [ ] Chạy kiểm thử tải và đo thời gian phản hồi trung bình của `scripts/api_server.py` trên kho đề `smp-math.com`.
- [ ] Ghi nhận các ca LaTeX đặc thù của `smp-math.com` bị SymPy CAS parse lỗi để báo nhóm tinh chỉnh.

### 4. Đỗ Viết Hoàng Long (`25125024` — Leader) — Review, Deploy & Final Report
- [ ] Review PR của 3 thành viên và điều chỉnh tham số $\lambda$, `cas_boost` phù hợp với dữ liệu thực tế của `smp-math.com`.
- [ ] Tổng hợp ảnh chụp demo thực tế trên `smp-math.com` và bảng số liệu benchmark vào `Report/Final_Report/` và `Slide/Final_Defense/`.
