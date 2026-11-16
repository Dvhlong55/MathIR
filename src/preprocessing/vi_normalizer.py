import re
from typing import List, Tuple, Optional
from deep_translator import GoogleTranslator

# Từ điển thuật ngữ toán học chuẩn Việt - Anh
MATH_DICT = {
    "tìm giá trị nhỏ nhất của biểu thức": "Find the minimum value of the expression",
    "tìm giá trị lớn nhất của biểu thức": "Find the maximum value of the expression",
    "tìm giá trị nhỏ nhất": "Find the minimum value",
    "tìm giá trị lớn nhất": "Find the maximum value",
    "chứng minh bất đẳng thức": "Prove the inequality",
    "chứng minh rằng": "Prove that",
    "giải hệ phương trình": "Solve the system of equations",
    "giải bất phương trình": "Solve the inequality",
    "giải phương trình": "Solve the equation",
    "tính tích phân": "Evaluate the integral",
    "tính giới hạn": "Find the limit",
    "tính đạo hàm": "Find the derivative",
    "tính khoảng cách": "Find the distance",
    "viết phương trình": "Write the equation",
    "tìm tọa độ": "Find the coordinates",
    "tìm tất cả các": "Find all",
    "tính ma trận": "Calculate the matrix",
    "tính định thức": "Calculate the determinant",
    "tính xác suất": "Calculate the probability",
    "số nguyên dương": "positive integer",
    "số thực dương": "positive real number",
    "số tự nhiên": "natural number",
    "số nguyên": "integer",
    "số thực": "real number",
    "bất đẳng thức": "inequality",
    "hệ phương trình": "system of equations",
    "phương trình bậc hai": "quadratic equation",
    "phương trình": "equation",
    "biểu thức": "expression",
    "hàm số": "function",
    "dãy số": "sequence",
    "ma trận": "matrix",
    "định thức": "determinant",
    "vectơ": "vector",
    "véc-tơ": "vector",
    "mặt phẳng": "plane",
    "đường thẳng": "line",
    "hình chóp": "pyramid",
    "hình vuông": "square",
    "tam giác": "triangle",
    "đường tròn": "circle",
    "tiếp tuyến": "tangent line",
    "vuông góc": "perpendicular",
    "song song": "parallel",
    "trực tâm": "orthocenter",
    "trọng tâm": "centroid",
    "nội tiếp": "inscribed",
    "ngoại tiếp": "circumscribed",
    "xác suất": "probability",
    "tổ hợp": "combination",
    "chỉnh hợp": "permutation",
    "không gian mẫu": "sample space",
    "chia hết cho": "is divisible by",
    "với mọi": "for all",
    "tồn tại": "there exists",
    "sao cho": "such that",
    "thỏa mãn": "satisfying",
    "khi và chỉ khi": "if and only if",
    "trên đoạn": "on the interval",
    "chứng minh": "Prove",
    "tìm": "Find",
    "cho": "Given",
    "giải": "Solve",
    "tính": "Calculate",
}

# Bắt đầy đủ 4 dạng bao công thức LaTeX: $$...$$, $...$, \(...\), \[...\]
LATEX_PATTERN = re.compile(
    r'\$\$[\s\S]+?\$\$|\$[^$]+?\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]'
)

# Bắt lại token đánh dấu kể cả khi bộ dịch tự thêm khoảng trắng hoặc đổi chữ thường
PLACEHOLDER_RESTORE_PATTERN = re.compile(
    r'__\s*MATH_\s*(\d+)\s*__', flags=re.IGNORECASE
)


def mask_formulas(text: str) -> Tuple[str, List[str]]:
    """Tách tất cả khối công thức LaTeX và thay thế bằng placeholder __MATH_i__."""
    formulas: List[str] = []

    def _replace(match: re.Match) -> str:
        formulas.append(match.group(0))
        return f"__MATH_{len(formulas) - 1}__"

    masked_text = LATEX_PATTERN.sub(_replace, text)
    return masked_text, formulas


def apply_math_dictionary(text: str) -> str:
    """Thay thế các cụm thuật ngữ toán học cố định theo thứ tự cụm dài trước, từ đơn sau."""
    result = text
    sorted_keys = sorted(MATH_DICT.keys(), key=len, reverse=True)
    for vi_phrase in sorted_keys:
        en_phrase = MATH_DICT[vi_phrase]
        pattern = re.compile(rf'(?<!\w){re.escape(vi_phrase)}(?!\w)', flags=re.IGNORECASE)
        result = pattern.sub(en_phrase, result)
    return result


def translate_text(masked_text: str) -> str:
    """Dịch văn bản đã che công thức sang tiếng Anh bằng từ điển cứng + deep-translator."""
    pre_translated = apply_math_dictionary(masked_text)
    try:
        translated = GoogleTranslator(source='vi', target='en').translate(pre_translated)
        if translated:
            return translated
    except Exception:
        pass
    return pre_translated


def unmask_formulas(text: str, formulas: List[str]) -> str:
    """Phục hồi chính xác 100% các công thức LaTeX vào đúng vị trí."""
    def _restore(match: re.Match) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(formulas):
            return formulas[idx]
        return match.group(0)

    restored = PLACEHOLDER_RESTORE_PATTERN.sub(_restore, text)
    # Fallback: thay thế trực tiếp nếu regex không bắt hết
    for i, formula in enumerate(formulas):
        restored = restored.replace(f"__MATH_{i}__", formula)
    return restored


def normalize_vi_to_en(query: str) -> str:
    """Hàm chính: Chuẩn hóa câu truy vấn toán học Tiếng Việt sang Tiếng Anh."""
    if not query or not query.strip():
        return query
    masked, formulas = mask_formulas(query)
    translated = translate_text(masked)
    result = unmask_formulas(translated, formulas)
    return result
