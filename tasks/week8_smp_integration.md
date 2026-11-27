# Week 8 Task Assignment — Tích hợp MathIR vào `smp-math.com`

**Branch tích hợp:** `feat/hybrid-integration`  
**Mục tiêu:** Đưa hệ thống Hybrid MathIR 3 tầng vào tìm kiếm thực tế trên kho đề của `smp-math.com`.

---

## Phân công thành viên

### 1. Trần Đức Toàn (`25125091`) — Sưu tầm & Trích xuất Kho sách/đề PDF
- [ ] Sưu tầm các quyển sách/giáo trình toán PDF và viết script trích xuất nội dung + công thức (dùng `pymupdf` / `pdfplumber`) sang `pdf_formulas.jsonl` theo định dạng `{"doc_id": "file.pdf#page=...", "latex": "..."}`.
- [ ] Chạy tạo `corpus_vecs.npy` + FAISS index và test tìm kiếm trên kho sách PDF đó.

### 2. Phạm Trung Duy (`25125010`) — Tối ưu Truy vấn Tiếng Việt
- [ ] Bổ sung thêm các từ khóa/thuật ngữ toán học phổ biến trên `smp-math.com` vào `src/preprocessing/vi_normalizer.py`.
- [ ] Tạo bộ test 20–30 câu truy vấn tiếng Việt thực tế để kiểm tra chất lượng tìm kiếm.

### 3. Nguyễn Nguyễn Gia Bảo (`25125006`) — Kiểm thử API & Độ trễ
- [ ] Chạy kiểm thử tải và đo thời gian phản hồi trung bình của `scripts/api_server.py` trên kho đề `smp-math.com`.
- [ ] Ghi nhận các ca LaTeX đặc thù của `smp-math.com` bị SymPy CAS parse lỗi để báo nhóm tinh chỉnh.

### 4. Đỗ Viết Hoàng Long (`25125024` — Leader & Admin `smp-math.com`) — Export DB, Deploy API & Nối Web
- [ ] Export dữ liệu đề thi/bài toán từ database của `smp-math.com` ra JSONL (sử dụng quyền admin).
- [ ] Deploy `scripts/api_server.py` và nối trực tiếp thanh tìm kiếm trên `smp-math.com` gọi `POST /search`.
- [ ] Tổng hợp ảnh chụp demo thực tế trên `smp-math.com` và bảng số liệu benchmark vào `Report/Final_Report/` và `Slide/Final_Defense/`.
