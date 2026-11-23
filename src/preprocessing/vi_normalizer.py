import re
from typing import List, Tuple, Optional
from deep_translator import GoogleTranslator

# Bắt đầy đủ các dạng bao công thức LaTeX: $$, $, \(, \[, và \begin{env}...\end{env}
LATEX_PATTERN = re.compile(
    r'\$\$[\s\S]+?\$\$|\$[^$]+?\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]|\\begin\{([a-zA-Z0-9*]+)\}[\s\S]+?\\end\{\1\}'
)

# Bắt lại placeholder MATHTOKENiX (hoặc fallback __MATH_i__)
PLACEHOLDER_RESTORE_PATTERN = re.compile(
    r'MATHTOKEN(\d+)X|__\s*MATH_\s*(\d+)\s*__', flags=re.IGNORECASE
)

# 1. Bảng bung từ viết tắt toán học và mẫu lọc khẩu ngữ
VI_SHORTHAND_MAP = {
    r'\bgtnn\b': 'giá trị nhỏ nhất',
    r'\bgtln\b': 'giá trị lớn nhất',
    r'\bbđt\b': 'bất đẳng thức',
    r'\bbpt\b': 'bất phương trình',
    r'\bhpt\b': 'hệ phương trình',
    r'\bpt\b': 'phương trình',
    r'\bcmr\b': 'chứng minh rằng',
    r'\bc/m\b': 'chứng minh',
    r'\btxđ\b': 'tập xác định',
    r'\bđkxđ\b': 'điều kiện xác định',
    r'\bđk\b': 'điều kiện',
    r'\bntcn\b': 'nguyên tố cùng nhau',
    r'\bcsc\b': 'cấp số cộng',
    r'\bcsn\b': 'cấp số nhân',
    r'\bvt\b': 'vế trái',
    r'\bvp\b': 'vế phải',
    r'\bđpcm\b': 'điều phải chứng minh',
}

NOISE_PREFIX_PATTERN = re.compile(
    r'^(?:mọi người ơi|các bạn ơi|cho mình hỏi|cho em hỏi|giúp mình|giúp em|'
    r'ai giải giúp|làm ơn|hãy giúp|bài toán này làm sao|đề bài là)\s*[,:;.-]*\s*',
    flags=re.IGNORECASE,
)

VI_ACCENT_CHARS = r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]'

# 2. Bảng khử đa nghĩa tiếng Việt trước khi dịch
VI_DISAMBIGUATION = [
    (r'\btrên\s+đoạn\b', 'trên khoảng đóng'),
    (r'\btrên\s+nửa\s+khoảng\b', 'trên khoảng nửa đóng'),
    (r'\bnghiệm\s+nguyên\s+dương\b', 'nghiem_nguyen_duong'),
    (r'\bnghiệm\s+nguyên\b', 'nghiem_nguyen'),
    (r'\bnghiệm\s+thực\s+phân\s+biệt\b', 'nghiem_thuc_phan_biet'),
    (r'\bnghiệm\s+phân\s+biệt\b', 'nghiem_phan_biet'),
    (r'\bnghiệm\s+thực\b', 'nghiem_thuc'),
    (r'\bnghiệm\s+dương\b', 'nghiem_duong'),
    (r'\bnghiệm\s+duy\s+nhất\b', 'nghiem_duy_nhat'),
    (r'\bvô\s+nghiệm\b', 'vo_nghiem'),
    (r'\bcó\s+nghiệm\b', 'co_nghiem'),
    (r'\bnghiệm\b', 'nghiem_so'),
    (r'\bđồng\s+biến\b', 'tăng đơn điệu'),
    (r'\bnghịch\s+biến\b', 'giảm đơn điệu'),
    (r'\bnguyên\s+hàm\b', 'tích phân bất định'),
    (r'\btiếp\s+tuyến\b', 'đường tiếp tuyến'),
]

# 2. Bảng hậu chuẩn hóa tiếng Anh sau dịch (ép về chuẩn ngữ liệu MathBERT)
POST_EN_RULES = [
    # Đổi For/Give/Let/Suppose ở đầu câu về Given
    (r'^(?:For|Give|Let|Suppose)\s+(?=MATHTOKEN\d+X|__MATH_\d+__|positive\b|real\b|integer\b|all\b|the\b|a\b|an\b)', 'Given '),
    (r'^Given\s+that\s+(?=MATHTOKEN\d+X|__MATH_\d+__)', 'Given '),
    (r'\bGiven\s+function\b', 'Given the function'),
    (r'\bGiven\s+pyramid\b', 'Given pyramid'),
    (r'\blateral\s+edge\b', 'Lateral edge'),
    (r'\bthat\s+when\b', 'that when'),
    # Đổi smallest/largest value -> minimum/maximum value
    (r'\b(?:smallest|lowest|least)\s+value\b', 'minimum value'),
    (r'\b(?:largest|greatest|highest|biggest)\s+value\b', 'maximum value'),
    # Đổi động từ tính toán về đúng chuẩn học thuật
    (r'\b(?:Calculate|Compute|Count|Find)\s+(?:the\s+)?integral\b', 'Evaluate the integral'),
    (r'\b(?:Calculate|Compute|Count|Evaluate)\s+(?:the\s+)?limit\b', 'Find the limit'),
    (r'\b(?:Compute|Count|Evaluate)\s+(?:the\s+)?derivative\b', 'Calculate the derivative'),
    # Chuẩn hóa giới từ và khoảng/đoạn
    (r'\bon\s+(?:the\s+)?(?:closed\s+interval|half-open\s+interval|segment|paragraph|stage|period|section|range)\b', 'on the interval'),
    (r'\bdivisible\s+for\b', 'divisible by'),
    # Khôi phục các token khử đa nghĩa nghiệm sang tiếng Anh chuẩn
    (r'\bnghiem_nguyen_duong\b', 'positive integer roots'),
    (r'\bnghiem_nguyen\b', 'integer roots'),
    (r'\bnghiem_thuc_phan_biet\b', 'distinct real roots'),
    (r'\bnghiem_phan_biet\b', 'distinct roots'),
    (r'\bnghiem_thuc\b', 'real roots'),
    (r'\bnghiem_duong\b', 'positive roots'),
    (r'\bnghiem_duy_nhat\b', 'unique root'),
    (r'\bvo_nghiem\b', 'no solution'),
    (r'\bco_nghiem\b', 'has roots'),
    (r'\bnghiem_so\b', 'roots'),
    (r'\bhas\s+two\s+solutions\b', 'has two roots'),
    (r'\bhave\s+two\s+solutions\b', 'has two roots'),
    # Tự động chèn mạo từ "the" bị thiếu
    (r'\bof\s+(expression|equation|function|inequality|system|polynomial|sequence|integral)\b', r'of the \1'),
    (r'\b(Solve|Prove|Simplify|Evaluate)\s+(equation|inequality|expression|system|integral)\b', r'\1 the \2'),
]

# 3. Từ điển dự phòng khi ngoại tuyến (Offline fallback)
MATH_DICT = {
    "tìm giá trị nhỏ nhất của biểu thức": "Find the minimum value of the expression",
    "tìm giá trị lớn nhất của biểu thức": "Find the maximum value of the expression",
    "tìm giá trị nhỏ nhất": "Find the minimum value",
    "tìm giá trị lớn nhất": "Find the maximum value",
    "chứng minh bất đẳng thức": "Prove the inequality",
    "chứng minh rằng": "Prove that",
    "giải hệ phương trình": "Solve the system of equations",
    "giải bất phương trình": "Solve the inequality",
    "giải phương trình bậc hai": "Solve the quadratic equation",
    "giải phương trình": "Solve the equation",
    "tính tích phân": "Evaluate the integral",
    "tính giới hạn": "Find the limit",
    "tính đạo hàm": "Calculate the derivative",
    "tính khoảng cách": "Find the distance",
    "tìm tọa độ véc-tơ": "Find the coordinates of vector",
    "tìm tọa độ": "Find the coordinates",
    "tìm tất cả các": "Find all",
    "tính xác suất": "Calculate the probability",
    "khi chọn ngẫu nhiên": "when randomly choosing",
    "chọn ngẫu nhiên": "randomly choosing",
    "học sinh nữ": "female student",
    "học sinh": "students",
    "có đúng": "there is exactly",
    "biết có": "given there are",
    "cạnh bên": "lateral edge",
    "vuông góc với mặt phẳng": "is perpendicular to plane",
    "mặt phẳng": "plane",
    "có đáy": "with base",
    "là hình vuông cạnh": "being a square with side",
    "hình vuông cạnh": "square with side",
    "khoảng cách từ": "the distance from",
    "đến mặt phẳng": "to plane",
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
    "hình chóp": "pyramid",
    "hình vuông": "square",
    "tam giác": "triangle",
    "đường tròn": "circle",
    "vuông góc": "perpendicular",
    "song song": "parallel",
    "mặt phẳng": "plane",
    "chia hết cho": "is divisible by",
    "với mọi": "for all",
    "tồn tại": "there exists",
    "sao cho": "such that",
    "thỏa mãn": "satisfying",
    "trên khoảng đóng": "on the interval",
    "trên đoạn": "on the interval",
    "tại điểm": "at the point",
    "để khi": "that when",
    "để": "that",
    "trong": "among",
    "người": "people",
    "nữ": "females",
    "với": "with",
    "và": "and",
    "từ": "from",
    "đến": "to",
    "cạnh": "side",
    "chứng minh": "Prove",
    "tìm": "Find",
    "cho": "Given",
    "giải": "Solve",
    "tính": "Calculate",
}


def mask_formulas(text: str) -> Tuple[str, List[str]]:
    """Tách tất cả khối công thức LaTeX và thay thế bằng placeholder ' MATHTOKENiX '."""
    formulas: List[str] = []

    def _replace(match: re.Match) -> str:
        formulas.append(match.group(0))
        return f" MATHTOKEN{len(formulas) - 1}X "

    masked_text = LATEX_PATTERN.sub(_replace, text)
    return masked_text, formulas


def expand_shorthand_and_clean(query: str) -> str:
    """Làm sạch lời dẫn khẩu ngữ và bung toàn bộ từ viết tắt toán học tiếng Việt."""
    cleaned = NOISE_PREFIX_PATTERN.sub('', query.strip())
    for pattern, full_phrase in VI_SHORTHAND_MAP.items():
        cleaned = re.sub(pattern, full_phrase, cleaned, flags=re.IGNORECASE)
    return cleaned


def strip_untranslated_vi_words(en_masked_text: str) -> str:
    """Loại bỏ các từ gõ sai chính tả/từ lóng tiếng Việt không dịch được để tránh làm nhiễu MathBERT."""
    tokens = en_masked_text.split()
    clean_tokens = [
        t for t in tokens
        if not re.search(VI_ACCENT_CHARS, t, flags=re.IGNORECASE)
    ]
    return " ".join(clean_tokens)


def disambiguate_vi_math(text: str) -> str:
    """Khử đa nghĩa các thuật ngữ toán tiếng Việt trước khi dịch NMT."""
    result = text
    for pattern, repl in VI_DISAMBIGUATION:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def canonicalize_en_math(text: str) -> str:
    """Hậu chuẩn hóa tiếng Anh sau dịch theo phong cách chuẩn học thuật MathBERT."""
    result = text
    for pattern, repl in POST_EN_RULES:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


def apply_math_dictionary(text: str) -> str:
    """Thay thế từ điển toán học cố định khi chạy ngoại tuyến."""
    result = text
    sorted_keys = sorted(MATH_DICT.keys(), key=len, reverse=True)
    for vi_phrase in sorted_keys:
        en_phrase = MATH_DICT[vi_phrase]
        pattern = re.compile(rf'(?<!\w){re.escape(vi_phrase)}(?!\w)', flags=re.IGNORECASE)
        result = pattern.sub(en_phrase, result)
    return result


def translate_text(masked_text: str) -> str:
    """Dịch câu đã che công thức: Khử đa nghĩa -> NMT tiếng Việt nguyên câu -> Hậu chuẩn hóa tiếng Anh -> Lọc từ tiếng Việt sót lại."""
    disambiguated = disambiguate_vi_math(masked_text)
    try:
        translated = GoogleTranslator(source='vi', target='en').translate(disambiguated)
        if translated:
            return strip_untranslated_vi_words(canonicalize_en_math(translated))
    except Exception:
        pass
    # Dự phòng offline hoặc khi bị giới hạn request Google
    return strip_untranslated_vi_words(canonicalize_en_math(apply_math_dictionary(disambiguated)))


def sanitize_placeholder_spacing(text: str) -> str:
    """Chuẩn hóa khoảng trắng quanh placeholder MATHTOKEN trước khi unmask để tránh dính chữ."""
    # Tách chữ bị dính liền trước placeholder (VD: GivenMATHTOKEN0X -> Given MATHTOKEN0X)
    text = re.sub(r'([a-zA-Z0-9])(MATHTOKEN\d+X|__\s*MATH_\s*\d+\s*__)', r'\1 \2', text)
    # Tách chữ bị dính liền sau placeholder (VD: MATHTOKEN0Xfor -> MATHTOKEN0X for)
    text = re.sub(r'(MATHTOKEN\d+X|__\s*MATH_\s*\d+\s*__)([a-zA-Z0-9])', r'\1 \2', text)
    # Xóa khoảng trắng thừa trước dấu chấm, phẩy
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


sanitize_math_spacing = sanitize_placeholder_spacing


def unmask_formulas(text: str, formulas: List[str]) -> str:
    """Phục hồi chính xác 100% các công thức LaTeX vào đúng vị trí và chuẩn hóa khoảng trắng."""
    text = sanitize_placeholder_spacing(text)

    def _restore(match: re.Match) -> str:
        idx_str = match.group(1) if match.group(1) is not None else match.group(2)
        idx = int(idx_str)
        if 0 <= idx < len(formulas):
            return formulas[idx]
        return match.group(0)

    restored = PLACEHOLDER_RESTORE_PATTERN.sub(_restore, text)
    # Fallback thay thế trực tiếp nếu còn sót
    for i, formula in enumerate(formulas):
        restored = restored.replace(f"MATHTOKEN{i}X", formula)
        restored = restored.replace(f"__MATH_{i}__", formula)

    # Dọn dẹp khoảng trắng thừa quanh dấu câu cuối cùng
    restored = re.sub(r'\s+([.,;:?!])', r'\1', restored)
    return re.sub(r'[ \t]+', ' ', restored).strip()


def normalize_vi_to_en(query: str) -> str:
    """Hàm chính: Chuẩn hóa câu truy vấn toán học Tiếng Việt sang Tiếng Anh (v3)."""
    if not query or not query.strip():
        return query
    masked, formulas = mask_formulas(query)
    cleaned = expand_shorthand_and_clean(masked)
    translated = translate_text(cleaned)
    result = unmask_formulas(translated, formulas)
    return result

