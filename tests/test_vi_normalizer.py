import os
import sys
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Đảm bảo import được gói src khi chạy từ thư mục gốc MathIR
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.vi_normalizer import normalize_vi_to_en, LATEX_PATTERN
from src.retrieval.stage1_pipeline import MathBERTEmbedder


def verify_latex_integrity(original_query: str, normalized_query: str) -> bool:
    """Kiểm tra toàn bộ công thức LaTeX trong câu gốc và câu đã dịch trùng khớp 100%."""
    orig_formulas = LATEX_PATTERN.findall(original_query)
    norm_formulas = LATEX_PATTERN.findall(normalized_query)
    return orig_formulas == norm_formulas


def main():
    test_file = os.path.join(os.path.dirname(__file__), "vi_en_test_cases.json")
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    embedder = MathBERTEmbedder("tbs17/MathBERT")

    print(f"{'ID':<8} | {'Sim trước dịch':>15} | {'Sim sau dịch':>15} | {'LaTeX OK?'}")
    print("-" * 60)

    high_sim_count = 0
    latex_ok_count = 0

    for item in test_cases:
        cid = item["id"]
        q_vi = item["query_vi"]
        q_en_target = item["query_en"]

        q_translated = normalize_vi_to_en(q_vi)
        time.sleep(0.2)

        vec_vi = embedder.encode(q_vi)
        vec_en = embedder.encode(q_en_target)
        vec_trans = embedder.encode(q_translated)

        # Vì các vector đã chuẩn hóa L2 nên tích vô hướng chính là Cosine Similarity
        sim_before = float(np.dot(vec_vi[0], vec_en[0])) * 100.0
        sim_after = float(np.dot(vec_trans[0], vec_en[0])) * 100.0

        is_latex_ok = verify_latex_integrity(q_vi, q_translated)
        if sim_after >= 90.0:
            high_sim_count += 1
        if is_latex_ok:
            latex_ok_count += 1

        status_icon = "✓" if is_latex_ok else "✗"
        print(f"{cid:<8} | {sim_before:>13.1f}% | {sim_after:>13.1f}% | {status_icon}")

    print("-" * 60)
    print(f"Tổng kết: {high_sim_count}/{len(test_cases)} mẫu đạt Sim >= 90% | {latex_ok_count}/{len(test_cases)} mẫu bảo toàn LaTeX.")

    min_expected = max(8, int(0.8 * len(test_cases)))
    assert latex_ok_count == len(test_cases), "Lỗi: Có công thức LaTeX bị biến dạng sau khi dịch!"
    assert high_sim_count >= min_expected, f"Lỗi: Số mẫu đạt độ tương đồng >= 90% chưa đạt yêu cầu ({high_sim_count}/{min_expected})!"


if __name__ == "__main__":
    main()
