import os
import sys
import numpy as np

# Đảm bảo import được gói src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.vi_normalizer import (
    normalize_vi_to_en,
    expand_shorthand_and_clean,
    strip_untranslated_vi_words,
    LATEX_PATTERN,
)
from src.retrieval.stage1_pipeline import MathBERTEmbedder, encode_resilient_query


def test_shorthand_expansion():
    """Kiểm tra bung từ viết tắt học đường."""
    samples = [
        ("Tìm gtnn của $P = a + \\frac{1}{a}$", "giá trị nhỏ nhất"),
        ("cmr bđt $a^2 + b^2 \\ge 2ab$", "chứng minh rằng"),
        ("giải pt $x^2 - 5x + 6 = 0$", "phương trình"),
        ("giải hpt $\\begin{cases} x+y=1 \\end{cases}$", "hệ phương trình"),
    ]
    for vi_text, expected_phrase in samples:
        expanded = expand_shorthand_and_clean(vi_text)
        assert expected_phrase in expanded, f"Failed shorthand on: {vi_text} -> {expanded}"
    print("✓ Test 1: Bung từ viết tắt (shorthand expansion) thành công!")


def test_noise_prefix_stripping():
    """Kiểm tra loại bỏ lời dẫn khẩu ngữ tán gẫu đầu câu."""
    samples = [
        ("Mọi người ơi giúp mình giải pt $x^2 - 4 = 0$", "giải phương trình"),
        ("Cho mình hỏi bài toán này làm sao: tìm gtnn của $x^2$", "tìm giá trị nhỏ nhất"),
        ("Ai giải giúp em với: giải phương trình $2x = 4$", "giải phương trình"),
    ]
    for vi_text, expected_phrase in samples:
        cleaned = expand_shorthand_and_clean(vi_text)
        assert expected_phrase in cleaned, f"Failed noise stripping on: {vi_text} -> {cleaned}"
        assert not any(n in cleaned.lower() for n in ["mọi người ơi", "cho mình hỏi", "ai giải giúp"]), f"Noise remains in: {cleaned}"
    print("✓ Test 2: Lọc khẩu ngữ đầu câu (noise prefix stripping) thành công!")


def test_untranslated_vi_strip():
    """Kiểm tra lọc bỏ từ lỗi chính tả tiếng Việt còn sót."""
    dirty_en = "Solve the equation nghiem_sai_loi_chinh_ta MATHTOKEN0X"
    clean_en = strip_untranslated_vi_words(dirty_en)
    # Từ không dấu tiếng Anh được giữ, nếu có dấu tiếng Việt sẽ bị loại bỏ
    dirty_with_accents = "Solve the equation chứg_mih_rằg MATHTOKEN0X"
    clean_accent = strip_untranslated_vi_words(dirty_with_accents)
    assert "chứg_mih_rằg" not in clean_accent
    assert "MATHTOKEN0X" in clean_accent
    print("✓ Test 3: Lọc từ lỗi chính tả tiếng Việt sót lại thành công!")


def test_end_to_end_normalization():
    """Kiểm tra luồng chuẩn hóa từ câu viết tắt/khẩu ngữ sang tiếng Anh chuẩn."""
    q1 = "Mọi người ơi giúp mình tìm gtnn của $P = a + \\frac{1}{a}$ với $a \\ge 2$."
    out1 = normalize_vi_to_en(q1)
    assert "$P = a + \\frac{1}{a}$" in out1
    assert "$a \\ge 2$" in out1
    assert "minimum value" in out1.lower()

    q2 = "Giải pt $x^2 - 5x + 6 = 0$."
    out2 = normalize_vi_to_en(q2)
    assert "$x^2 - 5x + 6 = 0$" in out2
    assert "solve the equation" in out2.lower()
    print("✓ Test 4: Chuẩn hóa trọn gói câu viết tắt & khẩu ngữ thành công!")


def test_formula_anchored_embedding():
    """Kiểm tra kỹ thuật Neo Vector Công Thức (Formula-Anchored Embedding)."""
    embedder = MathBERTEmbedder()
    query = "Find the minimum value of $P = a + \\frac{1}{a}$."
    vec_resilient = encode_resilient_query(embedder, query, alpha=0.70)
    assert vec_resilient.shape == (1, 768)
    assert vec_resilient.dtype == np.float32

    # Vector phải chuẩn hóa L2 (norm xấp xỉ 1.0)
    norm = np.linalg.norm(vec_resilient)
    assert abs(norm - 1.0) < 1e-4, f"Vector norm = {norm}, expected 1.0"
    print("✓ Test 5: Neo Vector Công Thức (Formula-Anchored Embedding) thành công!")


def main():
    print("=" * 60)
    print(" Chạy kịch bản kiểm thử tính năng Upgrade 2 (Từ lạ, viết tắt & neo vector)")
    print("=" * 60)
    test_shorthand_expansion()
    test_noise_prefix_stripping()
    test_untranslated_vi_strip()
    test_end_to_end_normalization()
    test_formula_anchored_embedding()
    print("=" * 60)
    print(" Toàn bộ 5/5 kiểm thử của Upgrade 2 ĐÃ ĐẠT!")


if __name__ == "__main__":
    main()
