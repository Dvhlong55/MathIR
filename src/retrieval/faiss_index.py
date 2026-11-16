import json
import faiss
import numpy as np
from typing import List, Dict, Any


class MathFAISSIndex:
    """Bộ đánh chỉ mục và tìm kiếm vector tốc độ cao Top-K bằng FAISS IndexFlatIP."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def _ensure_l2_normalized_float32(self, vectors: np.ndarray) -> np.ndarray:
        """Đảm bảo mảng đầu vào có định dạng (N, dim) float32 liên tục và được chuẩn hóa L2."""
        arr = np.ascontiguousarray(vectors, dtype=np.float32).copy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Số chiều vector ({arr.shape[1]}) khác với cấu hình chỉ mục ({self.dim}).")
        faiss.normalize_L2(arr)
        return arr

    def build(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Chế độ A: Xây dựng chỉ mục từ mảng vector (N, 768) và danh sách metadata."""
        norm_vectors = self._ensure_l2_normalized_float32(vectors)
        if norm_vectors.shape[0] != len(metadata):
            raise ValueError("Số lượng vector và số phần tử metadata phải bằng nhau.")
        self.index.reset()
        self.index.add(norm_vectors)
        self.metadata = metadata

    def save(self, index_path: str, meta_path: str) -> None:
        """Lưu chỉ mục FAISS (.bin) và metadata (.json) xuống ổ cứng."""
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, meta_path: str) -> None:
        """Nạp lại chỉ mục FAISS (.bin) và metadata (.json) từ ổ cứng."""
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_vector: np.ndarray, k: int = 100) -> List[Dict[str, Any]]:
        """Chế độ B: Truy vấn Top-K ứng viên gần nhất, trả về đúng chuẩn 5 khóa."""
        norm_query = self._ensure_l2_normalized_float32(query_vector)
        scores, indices = self.index.search(norm_query, k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            results.append({
                "rank": rank,
                "doc_id": str(item.get("doc_id", f"doc_{idx:04d}")),
                "score": round(float(score), 6),
                "formula": str(item.get("formula", "")),
                "text": str(item.get("text", "")),
            })
        return results
