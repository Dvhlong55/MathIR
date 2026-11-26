# Tóm Tắt Tiến Độ & Kết Quả Thực Nghiệm MathIR (Group 2)

**Nhánh tích hợp:** `feat/hybrid-integration` (giữ nguyên `main` cho bản cuối cùng)  
**Dữ liệu thực nghiệm:** 6.109 công thức & 115 câu truy vấn chuẩn từ **MIRB (`mseformula` + `wikiformula`)**

---

## 1. Những gì đã hoàn thành
1. **Ghép nối code của cả 4 thành viên** lên nhánh `feat/hybrid-integration`:
   - `src/preprocessing/` (`dataset_loader.py`, `formula_sampler.py`, `vi_normalizer.py`)
   - `src/representation/` (`tree_parser.py`, `tuple_extractor.py`)
   - `src/retrieval/` (`stage1_pipeline.py`, `faiss_index.py`, `tangent_retriever.py`, `structural_gate.py`, `symbolic_verifier.py`, `hybrid_pipeline.py`)
   - `src/evaluation/benchmark.py` và `scripts/run_hybrid_benchmark.py`
2. **Kiểm thử phần mềm:** **27/27 unit tests PASS** (trên Python 3.11.9).
3. **Kéo & chuẩn hóa bộ test:** Đã tải và trích xuất xong `data/processed/corpus_formulas.jsonl` (6.109 công thức), `benchmark_queries.jsonl` (115 câu truy vấn), và `benchmark_qrels.json`.
4. **Chạy benchmark thật:** Đã chạy xong cả bộ đối kháng (`tests/test_bottlenecks.py`) lẫn bộ 115 câu chuẩn (`reports/hybrid_benchmark_results.json`).

---

## 2. Kết quả đối chiếu: Baseline vs. Hệ thống Hybrid của nhóm

### A. Trên bộ test chuẩn (115 câu truy vấn, 6.109 công thức)
| Mô hình | `P@10` | `mAP` | `nDCG@10` |
| :--- | :---: | :---: | :---: |
| **Baseline: MathBERT** (`tbs17/MathBERT`) | **83.48%** | **84.46%** | **87.33%** |
| **Hệ thống mới: Full Hybrid** (3 tầng) | **83.22%** | **84.11%** | **87.27%** |

- **Kiểm định thống kê (Wilcoxon):** $p = 0.6121$ ($> 0.05$).
- **Nhận xét:** Trên tập chuẩn có chấm *partial match*, Full Hybrid đạt kết quả **tương đương** MathBERT (kém nhẹ `0.06%` `nDCG@10` do cổng phạt cấu trúc $\lambda = 0.5$ hơi gắt với các ca khớp một phần).

### B. Trên bộ test đối kháng (`tests/test_bottlenecks.py`)
| Ca kiểm thử | Baseline MathBERT | Hệ thống Hybrid |
| :--- | :---: | :--- |
| Đảo phân số ($\frac{x}{y}$ vs. $\frac{y}{x}$) | **99.0%** (nhận nhầm) | Bị cổng cấu trúc phạt giảm điểm (`× 0.5`) |
| Đảo dấu ($x > 0$ vs. $x < 0$) | **93.0%** (nhận nhầm) | Bị cổng cấu trúc phạt giảm điểm (`× 0.5`) |
| Sai hằng đẳng thức ($(a+b)^2$ vs. $a^2+b^2$) | **93.4%** (nhận nhầm) | SymPy CAS phát hiện $\Delta = 2ab \neq 0$ |
| Truy vấn tiếng Việt vs. tiếng Anh | **60.1%** (tụt điểm) | Stage 1 chuẩn hóa Việt $\to$ Anh giữ nguyên LaTeX |
| Chuỗi viết tắt ($1+2+\dots+n$) | SymPy **Syntax Crash** | Timeout & fallback an toàn, không crash |
