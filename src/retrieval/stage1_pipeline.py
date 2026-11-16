from typing import List, Dict, Any, Optional
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.preprocessing.vi_normalizer import normalize_vi_to_en, LATEX_PATTERN
from src.retrieval.faiss_index import MathFAISSIndex


class MathBERTEmbedder:
    """Bộ trích xuất vector 768 chiều đã chuẩn hóa L2 từ mô hình MathBERT."""

    def __init__(self, model_name: str = "tbs17/MathBERT"):
        # Ưu tiên load từ baselines/mathbert nếu có sẵn cục bộ
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "baselines", "mathbert")
        path = local_path if os.path.isdir(local_path) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling có xét attention_mask
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        vec = (sum_embeddings / sum_mask).cpu().numpy().astype("float32")

        # Chuẩn hóa L2
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.clip(norm, 1e-12, None)
        return vec


def encode_resilient_query(
    embedder: MathBERTEmbedder,
    normalized_query: str,
    formulas: Optional[List[str]] = None,
    alpha: float = 0.70,
) -> np.ndarray:
    """Kỹ thuật Neo Vector Công Thức (Formula-Anchored Embedding):
    Kết hợp vector toàn câu (70% - 75%) và vector thuần công thức LaTeX (25% - 30%).
    Nếu phần lời văn ngắn (< 2 từ) hoặc gặp từ lạ, phần công thức sẽ neo giữ vector không bị lệch.
    """
    vec_full = embedder.encode(normalized_query)
    if formulas is None:
        formulas = LATEX_PATTERN.findall(normalized_query)
    if not formulas:
        return vec_full

    # Nếu sau khi lọc từ lỗi mà phần chữ còn lại quá ít (< 2 từ), tăng trọng số công thức lên 50%
    text_only = LATEX_PATTERN.sub("", normalized_query).strip()
    if len(text_only.split()) < 2:
        alpha = 0.50

    formulas_joined = " ".join(formulas)
    vec_math = embedder.encode(formulas_joined)

    combined = alpha * vec_full + (1.0 - alpha) * vec_math
    norm = np.linalg.norm(combined, axis=1, keepdims=True)
    return (combined / np.clip(norm, 1e-12, None)).astype("float32")


class Stage1RetrievalPipeline:
    """Pipeline Tầng 1 hoàn chỉnh: Câu hỏi tiếng Việt -> Dịch bảo toàn LaTeX -> MathBERT -> Top-100 FAISS."""

    def __init__(self, index_path: str, meta_path: str, model_name: str = "tbs17/MathBERT"):
        self.embedder = MathBERTEmbedder(model_name=model_name)
        self.faiss_index = MathFAISSIndex(dim=768)
        self.faiss_index.load(index_path, meta_path)

    def retrieve_top_k(self, raw_vi_query: str, k: int = 100, resilient: bool = True) -> Dict[str, Any]:
        normalized_en_query = normalize_vi_to_en(raw_vi_query)
        if resilient:
            query_vec = encode_resilient_query(self.embedder, normalized_en_query)
        else:
            query_vec = self.embedder.encode(normalized_en_query)
        candidates = self.faiss_index.search(query_vec, k=k)
        return {
            "raw_query": raw_vi_query,
            "normalized_query": normalized_en_query,
            "candidates": candidates,
        }
