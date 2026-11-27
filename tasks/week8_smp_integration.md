# Week 8 Task Assignment — Tích hợp MathIR vào `smp-math.com`

**Branch tích hợp (MÔ HÌNH HOÀN CHỈNH):** `feat/hybrid-integration` *(Cả nhóm lưu ý: `git checkout feat/hybrid-integration` để lấy toàn bộ mô hình 3 tầng hoàn chỉnh + `api_server.py`, **KHÔNG** code hay push trực tiếp lên `main`)*  
**Mục tiêu:** Đưa hệ thống Hybrid MathIR 3 tầng vào tìm kiếm thực tế trên kho đề của `smp-math.com` và kho sách PDF qua Chrome Extension.

---

## Phân công thành viên

### 1. Trần Đức Toàn (`25125091`) — Sưu tầm & Trích xuất Kho sách/đề PDF
- [ ] Sưu tầm các quyển sách/giáo trình toán PDF và viết script trích xuất nội dung + công thức (dùng `pymupdf` / `pdfplumber`) sang `pdf_formulas.jsonl` theo định dạng `{"doc_id": "file.pdf#page=...", "latex": "..."}`.
- [ ] Chạy tạo `corpus_vecs.npy` + FAISS index và test tìm kiếm trên kho sách PDF đó.

### 2. Phạm Trung Duy (`25125010`) — Chrome Extension UI & Truy vấn Tiếng Việt
- [ ] Xây dựng **Chrome Extension** (`manifest.json`, `popup.html`, `popup.js`) cho phép mở trực tiếp khi đang đọc PDF/web trên trình duyệt để nhập hoặc bôi đen câu hỏi/công thức tìm kiếm.
- [ ] Bổ sung thêm các từ khóa/thuật ngữ toán học phổ biến vào `src/preprocessing/vi_normalizer.py` và test 20–30 câu truy vấn tiếng Việt.

### 3. Nguyễn Nguyễn Gia Bảo (`25125006`) — Kết nối Extension–API & Kiểm thử
- [ ] Nối Chrome Extension gọi API `POST /search` của `scripts/api_server.py`, hiển thị Top-$K$ kết quả (số trang PDF / bài toán + điểm số + lý do Stage 2/3).
- [ ] Đo độ trễ phản hồi API dưới tải thực tế và ghi nhận các ca LaTeX đặc thù bị SymPy CAS parse lỗi.

### 4. Đỗ Viết Hoàng Long (`25125024` — Leader & Admin `smp-math.com`) — Export DB, Deploy API & Nối Web
- [ ] Export dữ liệu đề thi/bài toán từ database của `smp-math.com` ra JSONL (sử dụng quyền admin).
- [ ] Deploy `scripts/api_server.py` và nối trực tiếp thanh tìm kiếm trên `smp-math.com` gọi `POST /search`.
- [ ] Tổng hợp ảnh chụp demo thực tế trên `smp-math.com` và bảng số liệu benchmark vào `Report/Final_Report/` và `Slide/Final_Defense/`.
