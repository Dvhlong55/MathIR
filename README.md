# MathIR: Mathematical Information Retrieval System Based on Syntax Tree Structures and Formula Semantics

[![Course](https://img.shields.io/badge/Course-SC203%20Scientific%20Method-blue)](https://github.com/Dvhlong55/MathIR)
[![Institution](https://img.shields.io/badge/Institution-HCMUS%20APCS-orange)](https://www.fit.hcmus.edu.vn)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An empirical research project conducted for the **SC203: Scientific Method** course at the **Advanced Program in Computer Science (APCS)**, Faculty of Information Technology, **University of Science, VNU-HCM**.

---

## 👥 Research Team (Group 2)
* **Team Leader:** Đỗ Viết Hoàng Long ([@Dvhlong55](https://github.com/Dvhlong55)) - ID: `25125024`
* **Team Members:**
  * Trần Đức Toàn - ID: `25125091`
  * Phạm Trung Duy - ID: `25125010`
  * Nguyễn Nguyễn Gia Bảo - ID: `25125006`
* **Lecturer:** Assoc. Prof. Dr. Trần Minh Triết
* **Teaching Assistant (TA):** Viết Thám

---

## 🎯 Testable Hypotheses
* **$H_1$ (Semantic Canonicalization & Variable Renaming Invariance):** Mathematical expressions that are algebraically equivalent or differ merely in variable identifiers suffer from near-zero recall under sequential 1D text matching, but achieve high retrieval precision when normalized into canonical Operator Trees (OPT).
* **$H_2$ (Hierarchical 2D Layout & Commutative Invariance):** Tree Edit Distance (TED) and tree-based dense vector embeddings yield statistically superior Mean Average Precision (mAP) and Precision@K over sequential n-gram string matching.
* **$H_3$ (Multi-Modal Context-Formula Interdependence):** Fusing contextual text embeddings with formula structural embeddings retrieves the target problem significantly better than formula-only or text-only retrieval.

---

## 📁 Repository Structure
```text
MathIR/
├── baselines/                          # (Git-ignored) Local clones of MathBERT, TangentCFT, SymPy
├── demo/                               # Interactive local web UI (index.html)
├── reports/                            # Academic reports, research proposals & weekly submissions
│   ├── proposal/                       # Official Research Proposal (LaTeX & PDF)
│   └── weekly/                         # Weekly progress reports (w1, w3, w4, w5, w7...)
├── src/                                # Core implementation of MathIR
│   ├── preprocessing/                  # LaTeX/MathML parsing, VN->EN normalization, SymPy CAS
│   ├── representation/                 # SLT / OPT tree construction & AST parsing
│   ├── retrieval/                      # FAISS indexing, structural penalty gate & hybrid fusion
│   └── evaluation/                     # Metric computations (nDCG', mAP', P@10, significance tests)
├── tasks/                              # Weekly team task assignments (week7.md...)
├── tests/                              # Adversarial & diagnostic test suite (test_bottlenecks.py)
├── requirements.txt                    # Python dependencies
├── server.py                           # Local Hybrid MathIR benchmark server
└── README.md
```

---

## 📚 Core Baselines & Benchmarks

| Paradigm | Baseline | Paper | Repository |
| :--- | :--- | :--- | :--- |
| **Neural (Text BERT)** | MathBERT | Shen et al. (2021), *arXiv:2106.07340* | [tbs17/MathBERT](https://github.com/tbs17/MathBERT) |
| **Structural (OPT/SLT)** | Tangent-CFT / CFTED | Mansouri et al. (ICTIR '19; CEUR Vol-2696) | [dprl/TangentCFT](https://gitlab.com/dprl/TangentCFT) |
| **Symbolic (CAS)** | SymPy CAS | Meurer et al. (PeerJ CS, 2017) | [sympy/sympy](https://github.com/sympy/sympy) |

**Standard Benchmarks:** ARQMath (CLEF Lab), NTCIR-12 MathIR Task, and MIRB ([arXiv:2505.15585](https://arxiv.org/abs/2505.15585) / [j991222/mirb](https://github.com/j991222/mirb)).

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/Dvhlong55/MathIR.git
cd MathIR
pip install -r requirements.txt
```

### 2. Running Empirical Benchmark Suite
```bash
python tests/test_bottlenecks.py
```

### 3. Launching Interactive Benchmark Engine
```bash
python server.py 8000
# Open http://127.0.0.1:8000 in your browser
```
