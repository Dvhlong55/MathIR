from typing import List, Dict, Any, Optional
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.preprocessing.vi_normalizer import normalize_vi_to_en
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


class Stage1RetrievalPipeline:
    """Pipeline Tầng 1 hoàn chỉnh: Câu hỏi tiếng Việt -> Dịch bảo toàn LaTeX -> MathBERT -> Top-100 FAISS."""

    def __init__(self, index_path: str, meta_path: str, model_name: str = "tbs17/MathBERT"):
        self.embedder = MathBERTEmbedder(model_name=model_name)
        self.faiss_index = MathFAISSIndex(dim=768)
        self.faiss_index.load(index_path, meta_path)

    def retrieve_top_k(self, raw_vi_query: str, k: int = 100) -> Dict[str, Any]:
        normalized_en_query = normalize_vi_to_en(raw_vi_query)
        query_vec = self.embedder.encode(normalized_en_query)
        candidates = self.faiss_index.search(query_vec, k=k)
        return {
            "raw_query": raw_vi_query,
            "normalized_query": normalized_en_query,
            "candidates": candidates,
        }
