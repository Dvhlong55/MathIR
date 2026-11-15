# Week 7 Task Assignment — MathIR Neuro-Symbolic Hybrid

**Repo:** `https://github.com/Dvhlong55/MathIR`  
**Mục tiêu tuần:** Khởi động xây dựng hệ thống **Neuro-Symbolic Hybrid 3 tầng** để khắc phục các điểm yếu của MathBERT (đảo dấu, đảo tử/mẫu, nhầm hàm lượng giác, suy giảm tiếng Việt).

---

## Phân công thành viên

### 1. Đỗ Viết Hoàng Long (`25125024` — Leader) | Branch: `feat/hybrid-gate`
- [ ] Ghép nối pipeline 3 tầng hoàn chỉnh trong `src/`.
- [ ] Code **Stage 2: Graded Structural Penalty Gate** (kiểm tra đảo dấu `>`/`<`, đảo tử/mẫu, sai toán tử gốc).
- [ ] Tune hệ số phạt $\lambda$ trên tập validation (tránh làm tụt điểm partial match).
- [ ] Chạy **ablation study** (Stage 1 vs. Stage 1+2 vs. Stage 1+2+3) và kiểm định thống kê ($p < 0.05$).

### 2. Trần Đức Toàn (`25125091`) | Branch: `feat/data-tangent`
- [ ] Tải và trích xuất tập mẫu thực nghiệm từ **ARQMath / NTCIR-12 / MIRB** (50k–100k cặp công thức/bài toán).
- [ ] Chạy **LaTeXML + Tangent-S** (khuyến nghị dùng WSL/Ubuntu hoặc Colab) để trích xuất cây OPT/SLT.
- [ ] Tái lập baseline **Tangent-CFT / Tangent-CFTED** làm đầu vào cho Stage 2.

### 3. Phạm Trung Duy (`25125010`) | Branch: `feat/vi-en-faiss`
- [ ] Code **Stage 1: Bộ chuẩn hóa truy vấn Việt → Anh** (tách giữ nguyên khối `$...$` bằng regex + từ điển thuật ngữ toán / `deep-translator` hoặc `Helsinki-NLP/opus-mt-vi-en`, không dùng LLM nặng).
- [ ] Tạo bộ test truy vấn song ngữ **Việt–Anh** riêng để đánh giá phần tiếng Việt.
- [ ] Đánh chỉ mục **FAISS** cho vector `tbs17/MathBERT` để lấy Top-100 nhanh.

### 4. Nguyễn Nguyễn Gia Bảo (`25125006`) | Branch: `feat/sympy-calibration`
- [ ] Code **Stage 3: SymPy CAS có timeout** cho từng candidate, log **tỉ lệ parse thành công** trên LaTeX nhiễu của ARQMath và fallback an toàn khi gặp dấu `...` ($1+2+\dots+n$).
- [ ] Tune **bounded score boost** cho các cặp $\Delta = 0$ (thay vì ép cứng lên Rank 1).
- [ ] Đo **cosine nền** của MathBERT trên cặp ngẫu nhiên và cặp tương đương thật để làm mốc so sánh cho các ca 93%–99%.

---

## Quy trình Git
```bash
git clone https://github.com/Dvhlong55/MathIR.git
cd MathIR
git checkout -b feat/<ten-branch-cua-ban>
# ... code & commit ...
git push origin feat/<ten-branch-cua-ban>
# Tạo Pull Request trên GitHub để Leader review & merge vào main
```
