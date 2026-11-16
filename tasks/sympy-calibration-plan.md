# Implementation Plan: `feat/sympy-calibration` (Stage 3 & MathBERT Calibration)

> **Owner:** Nguyễn Nguyễn Gia Bảo (`25125006`)  
> **Git Branch:** `feat/sympy-calibration` (branched from `main` at `8bb60e9`)  
> **Pipeline Scope:** Stage 3 (Timeout-Guarded SymPy CAS Verification + Bounded Score Boost) and MathBERT Cosine Baseline Calibration for the **Three-Stage Neuro-Symbolic Hybrid MathIR Pipeline**.  
> **Project References:** [tasks/week7.md](week7.md), [reports/weekly/w7/Group2_Week7_Report.tex](../reports/weekly/w7/Group2_Week7_Report.tex), [reports/proposal/Group2_MathIR.tex](../reports/proposal/Group2_MathIR.tex), [reports/empirical_benchmark_report.md](../reports/empirical_benchmark_report.md), [server.py](../server.py), [tests/test_bottlenecks.py](../tests/test_bottlenecks.py).

---

## 1. Committed Work on `feat/sympy-calibration`

The Stage 3 core infrastructure, evaluation metrics, experiment scripts, and unit tests have been split into focused Conventional Commits on `feat/sympy-calibration`:

| Commit | Type & Message | Files | Summary of What Was Delivered |
| :--- | :--- | :--- | :--- |
| `b55404f` | `feat(retrieval): add timeout-guarded Stage 3 SymPy verifier and bounded boost` | `src/retrieval/symbolic_verifier.py`, `src/retrieval/__init__.py` | Implemented `TimeoutSymbolicVerifier` (Windows-safe persistent spawned worker process with per-candidate timeout), `VerificationStats` telemetry, LaTeX noise cleaning (`\sin^2(x)` Lark disambiguation, presentation macro stripping), early ellipsis (`...`, `\dots`, `\ldots`) safe fallback, relational canonicalization, and `apply_bounded_boost` (`residual`, `clipped_additive`, `hard_rank1`). |
| `37113b5` | `feat(evaluation): add ARQMath prime metrics and NTCIR-12 bpref` | `src/evaluation/metrics.py` | Added judged-only IR metrics (`precision_at_k_prime`, `average_precision_prime`, `mean_average_precision_prime`, `ndcg_prime`) and Buckley & Voorhees `bpref` alongside the existing binary metrics. |
| `bdf559f` | `feat(scripts): add SymPy parse-rate audit and bounded boost tuning scripts` | `scripts/eval_sympy_parse_rate.py`, `scripts/tune_bounded_boost.py` | Added CLI scripts to audit SymPy parse success rates / error breakdowns / p50-p95-max latency on noisy ARQMath LaTeX, and to grid-search `(mode, beta, min_context_score)` against `stage12_only` and `hard_rank1`. |
| `7080435` | `feat(calibration): add MathBERT cosine baseline calibration script` | `scripts/calibrate_mathbert_cosine.py` | Added CLI script using `tbs17/MathBERT` (`[CLS]` L2-normalized, matching `server.py`) to measure cosine distributions across random unrelated pairs, truly equivalent pairs, and adversarial cases. |
| `572e6de` | `test(stage3): add unit tests for symbolic verifier, metrics, and tuning` | `tests/test_symbolic_verifier.py` | Added 9 unit tests covering algebraic/trig equivalence, polarity/inversion rejection, flipped inequality normalization, ellipsis fallback, broken LaTeX handling, Windows timeout recovery, bounded boost reranking, prime/`bpref` metrics, and tuning/calibration helpers. |
| *(this commit)* | `docs(tasks): move and translate sympy-calibration-plan to English` | `tasks/sympy-calibration-plan.md` *(moved from root)* | Relocated this execution plan into `tasks/`, translated to English, and documented committed deliverables and verified test/diagnostic outputs. |

### Verified Local Checks (Observed Before/After Commits)
- `.\.venv\Scripts\pytest.exe -v tests/test_symbolic_verifier.py`: **9/9 passed** (0 warnings).
- `.\.venv\Scripts\python.exe scripts/eval_sympy_parse_rate.py --demo-noisy`: **50% parse success** on the 10-item built-in diagnostic set (`20%` `ellipsis_sequence`, `30%` `parse_error`, `p50 = 20.588 ms`).
- `.\.venv\Scripts\python.exe scripts/calibrate_mathbert_cosine.py --include-adversarial`: Reproduced the exact `tbs17/MathBERT` adversarial raw cosine scores reported in `Group2_Week7_Report.tex`:
  - `Fraction Inversion (x/y vs y/x)`: `0.99`
  - `Inequality Flip (x > 0 vs x < 0)`: `0.9298`
  - `Algebraic Fallacy ((a+b)^2 vs a^2+b^2)`: `0.9342`
  - `Exponent Sign (e^x vs e^{-x})`: `0.9086`
  - `Trig Orthogonal (sin(x) vs cos(x))`: `0.936`
- `evaluate_pair_core` verified on `TEST_CASES[4]` (`Let $a \ge 2$. Find the minimum value of $P = a + \frac{1}{a}$.` vs. English paraphrase) → `EQUIV_PROVEN` (`a + 1/a`), whereas legacy `server.py` (which lacked `backend="lark"` and `antlr4`) failed with `CAS Parse Error`.

---

## 2. Current Pipeline State & Remaining Gaps

### 2.1. Existing Repository Baseline (Verified from Source)
1. **Prototype pair evaluator (`server.py`):**
   - `get_mathbert_embedding` (`server.py:48-57`) encodes inputs using `tbs17/MathBERT` (`[CLS]` token embedding, L2-normalized).
   - `evaluate_hybrid` (`server.py:178-349`) parses expressions with SymPy and unconditionally overwrites `fused_score = 1.0` whenever `is_equiv == True` (`server.py:291-293`), with no per-candidate timeout.
2. **Preprocessing stub (`src/preprocessing/canonicalizer.py`):**
   - Contains `FormulaCanonicalizer.simplify_expression` using `sympy.sympify` + `sympy.expand` only (no LaTeX support, no timeout).
3. **Diagnostic stress-test (`tests/test_bottlenecks.py`):**
   - Covers 8 adversarial cases (polarity flip, fraction inversion, algebraic misconception, exponent sign, EN–VI gap, ellipsis `1 + 2 + ... + n`, domain restriction `\sqrt{x^2}` vs. `x`).
4. **Local Python environment (verified):**
   - System Python `3.14.5`; project `.venv` (ignored by `.gitignore`) now has `sympy 1.14.0`, `lark 1.3.1`, `pytest 9.2.0`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch 2.15.0`, `transformers 5.23.0`.

### 2.2. What Is Still Needed to Complete Week 7 Goals (Data-Dependent)
Because the **ARQMath / NTCIR-12 / MIRB** datasets and **Stage 1 + Stage 2** outputs from teammates are not yet in the repository, the following empirical runs remain pending:
1. **Audit Stage 3 parse rate on real noisy ARQMath LaTeX** (waiting on formula sample from Toàn `25125091`).
2. **Measure full MathBERT baseline distributions** on random unrelated pairs ($N \ge 1,000$) and truly equivalent pairs (waiting on corpus sample / ground-truth pairs).
3. **Tune bounded score boost** $(\text{mode}, \beta, \tau_{\min})$ on a held-out validation split containing `stage12_score` and human `relevance` judgments (waiting on Stage 1+2 pipeline scores from Duy `25125010` and Long `25125024`).
4. **Integration review** with Long (`25125024`) to wire `TimeoutSymbolicVerifier` into the end-to-end 3-stage pipeline and optionally update `server.py`.

> **Citation Correction Note:**  
> `reports/empirical_benchmark_report.md` (line 32) mistakenly attributed `tbs17/MathBERT` to Peng et al. (2021). Per `reports/weekly/w7/Group2_Week7_Report.tex` (Section 1.1, lines 92 & 103–106) and primary paper verification, **`tbs17/MathBERT` is Shen et al. (2021, arXiv:2106.07340)**, a pure 1D-text MathBERT model, distinct from Peng et al. (2021, arXiv:2105.00377) which injects Operator Trees into attention.

---

## 3. Scientific Literature & Design Principles Extracted

| Paper / Reference | Verified Core Contribution | Direct Design Decision Derived for `feat/sympy-calibration` |
| :--- | :--- | :--- |
| **Shen et al. (2021)**, *MathBERT: A Pre-trained Language Model for General NLP Tasks in Mathematics Education* (arXiv:2106.07340, NeurIPS 2021 MATHAI4ED Workshop) | Pre-trains BERT-base (110M params) on Pre-K through graduate/arXiv math corpora with `mathVocab`. Operates on linear 1D subword tokens without explicit 2D tree attention. | Explains why `tbs17/MathBERT` captures broad math context yet confuses structurally opposite expressions with identical tokens ($\frac{x}{y}$ vs. $\frac{y}{x}$, $x>0$ vs. $x<0$) — motivating symbolic verification in Stage 3. |
| **Ethayarajh (2019)**, *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings* (EMNLP 2019) | Shows Transformer contextual embedding spaces are **anisotropic** (vectors cluster in a narrow cone), inflating raw cosine similarity even between random unrelated inputs; requires **random-baseline correction**. | **Directly motivates Task 3 (Calibration):** Raw scores like `93%–99%` cannot be interpreted in isolation; we must report random unrelated and true-equivalent distributions alongside adversarial pairs. |
| **Zanibbi et al. (2020)**, *Overview of ARQMath 2020: CLEF Lab on Answer Retrieval for Questions with Math* (CLEF 2020, LNCS 12260) | Evaluates math retrieval **in problem-solving context** with graded relevance and incomplete pooling, introducing **prime metrics** ($\text{nDCG}'$, $\text{MAP}'$, $\text{P}'@10$) over judged hits only. | **Directly motivates Task 2 (Bounded Boost):** Algebraic identity ($\Delta = 0$) does not guarantee contextual relevance to a user's question; unconditional Rank-1 promotion (`1.0`) harms context ranking and must be replaced by a validation-tuned bounded boost. |
| **Mansouri et al. (2019, 2020)** & **Zanibbi, Mansouri, & Agarwal (2024)**, *Tangent-CFT* (ICTIR '19), *Tangent-CFTED* (CEUR Vol-2696), *Mathematical Information Retrieval* (arXiv:2408.11646) | Establishes **multi-stage coarse-to-fine retrieval**: fast dense embedding recall (Top-$K$) followed by expensive structural comparison (Tree Edit Distance) only on candidates. | **Directly motivates Stage 3 scope & timeout:** SymPy `simplify()` is too expensive for full-corpus scanning and can stall on pathological expressions; run Stage 3 strictly on Top-$K$ candidates with a hard per-candidate timeout. |
| **Meurer et al. (2017)**, *SymPy: Symbolic Computing in Python* (PeerJ Computer Science 3:e103) | Pure-Python CAS supporting symbolic ASTs, polynomial/trig simplification, and LaTeX parsing (`parse_latex` via `antlr` or `lark` backend). | Grounds Stage 3 implementation while highlighting syntactic fragility on informal notation (`...`) and parser edge cases (including Lark's `Tree('_ambig', ...)` behavior observed locally). |
| **Ju & Dong (2025)**, *MIRB: Mathematical Information Retrieval Benchmark* (arXiv:2505.15585) | Unified benchmark spanning 4 tasks and 12 datasets (Semantic Statement, QA, Premise, Formula Retrieval) evaluated with `nDCG@10`, `mAP`, `MRR`, `Recall`. | Provides multi-task generalization check so bounded boost tuning does not overfit a single dataset. |
| **Zanibbi et al. (2016)**, *NTCIR-12 MathIR Task* | English Wikipedia Math Corpus (~319,689 articles, ~580K–590K formulas) evaluated with **`bpref`** for incomplete relevance pools. | Provides an independent secondary benchmark alongside ARQMath. |

---

## 4. Scope & Team Interfaces

### 4.1. In-Scope (`25125006`)
- **Stage 3 Verifier (`src/retrieval/symbolic_verifier.py`):** Timeout-guarded SymPy verification on Top-$K$ candidates, noise cleaning, ellipsis rejection, structured outcome statuses (`EQUIV_PROVEN`, `NOT_EQUIV`, `UNSUPPORTED_SYNTAX`, `PARSE_ERROR`, `TIMEOUT`, `SIMPLIFY_ERROR`), and parse telemetry.
- **Bounded Score Boost (`apply_bounded_boost` & `scripts/tune_bounded_boost.py`):** Replacing unconditional `1.0` with tunable boost functions and evaluating on validation data.
- **MathBERT Cosine Calibration (`scripts/calibrate_mathbert_cosine.py`):** Measuring random, true-equivalent, and adversarial cosine distributions on `tbs17/MathBERT`.

### 4.2. Out-of-Scope (Owned by Teammates)
- VN→EN query normalization, bilingual test set, and FAISS indexing (`25125010` — Phạm Trung Duy).
- Extracting 50k–100k ARQMath/NTCIR-12/MIRB samples, LaTeXML + Tangent-S, Tangent-CFT/CFTED (`25125091` — Trần Đức Toàn).
- Stage 2 Graded Structural Penalty Gate ($\lambda$) and full 3-stage ablation + significance testing (`25125024` — Đỗ Viết Hoàng Long).

---

## 5. Technical Design (Verified Mechanics vs. Reasoned Proposals)

### 5.1. Windows-Safe Timeout & SymPy Parsing (`src/retrieval/symbolic_verifier.py`)
- **Verified platform constraint:** `signal.SIGALRM` is unavailable on Windows, and Python threads cannot interrupt CPU-bound `sympy.simplify()` calls.
- **Implemented engineering solution:** A **persistent `spawn` worker process** (`_worker_loop`) communicates over unidirectional `multiprocessing.Pipe` connections:
  - Child pre-imports SymPy and warms both `parse_latex` and `parse_expr` before sending `"READY"` so `timeout_seconds` measures only actual candidate verification time (avoiding ~0.5–0.8s Windows process-spawn overhead per candidate).
  - If `resp_recv.poll(timeout_seconds)` expires, the parent terminates/kills the stalled worker, records `VerificationStatus.TIMEOUT`, and lazily respawns a clean worker on the next call.
- **Observed Lark pitfall & mitigation:** Testing on `\sin^2(x) + \cos^2(x)` revealed that SymPy 1.14's `lark` backend returns `lark.tree.Tree('_ambig', ...)` rather than raising an exception on ambiguous parses. `clean_latex_noise` disambiguates `\trig^k(arg)` into `(\trig(arg))^k`, and `_parse_latex_with_fallback` strictly enforces `isinstance(parsed, sp.Basic)` so raw `Tree` objects never reach `simplify()`.

### 5.2. Bounded Score Boost Formulations (Design Proposal — To Be Tuned)
Let $s_{12} \in [0, 1]$ be the normalized Stage 1 + Stage 2 score:
1. **Residual Bounded Boost (Recommended default):**
   $$s_{\text{final}} = s_{12} + \beta \cdot (1 - s_{12}) \cdot \mathbb{I}(\Delta = 0) \cdot \mathbb{I}(s_{12} \ge \tau_{\min})$$
   - Keeps $s_{\text{final}} \in [0, 1]$, preserves relative contextual ranking among $\Delta = 0$ candidates, and prevents boosting off-topic candidates below $\tau_{\min}$.
2. **Clipped Additive Boost (Ablation alternative):**
   $$s_{\text{final}} = \min\big(1.0,\; s_{12} + \beta \cdot \mathbb{I}(\Delta = 0) \cdot \mathbb{I}(s_{12} \ge \tau_{\min})\big)$$
3. **Validation grid:** Search $\beta \in \{0.05, 0.1, 0.15, 0.2, 0.3, 0.5\}$ and $\tau_{\min} \in \{0.0, 0.3, 0.5\}$ against both `stage12_only` ($\beta=0$) and `hard_rank1` (`1.0`). *(Optimal values are not assumed beforehand and must be selected empirically on validation data.)*

### 5.3. MathBERT Anisotropy Calibration Protocol
- Uses the exact `[CLS]` L2-normalized embedding path from `server.py:48-57` to ensure direct comparability with prior `93%–99%` measurements.
- Compares 3 reference groups: (1) $N \ge 1,000$ random unrelated pairs, (2) truly equivalent pairs ($\Delta = 0$ / benchmark duplicates), and (3) adversarial failure cases (`ADVERSARIAL_CASES`), reporting `mean`, `std`, `median`, `p5`, `p95`, z-score vs. random, and anisotropy-rescaled score $(s - \mu_{\text{random}}) / (1 - \mu_{\text{random}})$.

---

## 6. Evaluation Benchmarks & Success Criteria

| Deliverable | Dataset / Benchmark | Evaluation Metrics | Success Criteria |
| :--- | :--- | :--- | :--- |
| **Stage 3 Parser & Timeout** | ARQMath LaTeX sample (50k–100k target from `25125091`) + `tests/test_bottlenecks.py` | Parse success rate (%), failure-reason breakdown, p50/p95/max latency (ms), crash rate | 0 unhandled crashes on noisy LaTeX and `...`; reproducible parse audit JSON; timeout cleanly interrupts slow expressions without breaking subsequent calls. |
| **Bounded Score Boost** | ARQMath (Task 1/2 val & test), NTCIR-12 MathIR, MIRB | $\text{nDCG}'$ (primary), $\text{MAP}'$, $\text{P}'@10$, `bpref`, `nDCG@10`; paired t-test / Wilcoxon ($p < 0.05$) | Validation-tuned bounded boost outperforms or matches both `stage12_only` and unconditional `hard_rank1` (`1.0`). |
| **MathBERT Calibration** | $\ge 1,000$ random pairs + true-equivalent pairs + `ADVERSARIAL_CASES` | Mean $\pm$ Std, Median, P5/P95, z-score vs. random, rescaled similarity | Quantified baseline distributions demonstrating how anisotropic baseline cosine inflates adversarial scores relative to true equivalents. |

---

## 7. Next Execution Commands (Once Team Datasets Are Ready)

```powershell
# 0. Activate local virtual environment and run unit tests
.\.venv\Scripts\pytest.exe -v tests/test_symbolic_verifier.py

# 1. Audit Stage 3 parse rate & latency on ARQMath sample
.\.venv\Scripts\python.exe scripts/eval_sympy_parse_rate.py `
  --input data/arqmath_sample.jsonl --formula-col latex --timeout 0.75 `
  --output reports/stage3_parse_audit.json

# 2. Run MathBERT cosine baseline calibration
.\.venv\Scripts\python.exe scripts/calibrate_mathbert_cosine.py `
  --corpus-formulas data/arqmath_formulas.jsonl --n-random 1000 `
  --equiv-pairs data/equiv_pairs.jsonl --include-adversarial `
  --output reports/mathbert_calibration.json

# 3. Tune bounded score boost on Stage 1+2 validation outputs
.\.venv\Scripts\python.exe scripts/tune_bounded_boost.py `
  --input data/val_stage12.jsonl --timeout 0.75 `
  --output reports/boost_tuning.json
```

---

## 8. Open Dependencies & Risks
- **Dataset availability:** ARQMath, NTCIR-12, and MIRB files are not yet present in `data/` (awaiting `25125091`).
- **Stage 1+2 schema alignment:** Validation tuning expects `.jsonl` rows with `query_id`, `query_formula`, `doc_id`, `cand_formula`, `stage12_score`, `relevance`; coordinate final field names with `25125024` during integration.
- **Scale runtime control:** Running SymPy across 50k–100k pairs can be slow if `timeout` is too large; tune `timeout` (e.g., `0.5s–0.75s`) on a 1,000-formula pilot audit before scaling to the full split, and restrict Stage 3 reranking to Top-$K$ ($K \le 100$) candidates.
