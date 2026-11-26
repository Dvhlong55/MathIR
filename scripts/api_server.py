#!/usr/bin/env python3
"""
Lightweight CORS-enabled REST API Server for MathIR Hybrid Search.
Ready to integrate with smp-math.com or local PDF/exam banks.

Usage:
    python scripts/api_server.py --port 8080
Endpoint:
    POST /search  {"query": "Cho x > 0, tìm ...", "top_k": 10}
    GET  /health
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.vi_normalizer import normalize_vi_to_en
from src.retrieval.faiss_index import MathFAISSIndex
from src.retrieval.hybrid_pipeline import CandidateItem, HybridMathIRPipeline
from src.retrieval.stage1_pipeline import MathBERTEmbedder
from src.retrieval.symbolic_verifier import TimeoutSymbolicVerifier, VerificationStatus


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class MathIRSearchEngine:
    def __init__(self, data_dir: str):
        corpus_path = os.path.join(data_dir, "corpus_formulas.jsonl")
        vecs_cache = os.path.join(data_dir, "corpus_vecs.npy")

        self.corpus = load_jsonl(corpus_path)
        self.embedder = MathBERTEmbedder()

        if os.path.exists(vecs_cache):
            corpus_vecs = np.load(vecs_cache)
        else:
            texts = [item["latex"] for item in self.corpus]
            corpus_vecs = np.vstack(
                [self.embedder.encode(texts[i : i + 64]) for i in range(0, len(texts), 64)]
            )
            np.save(vecs_cache, corpus_vecs)

        self.index = MathFAISSIndex(dim=corpus_vecs.shape[1])
        self.index.build(
            corpus_vecs,
            [
                {"idx": i, "doc_id": item["doc_id"], "formula": item["latex"]}
                for i, item in enumerate(self.corpus)
            ],
        )
        self.pipeline = HybridMathIRPipeline(lambda_penalty=0.5, cas_boost=0.25)
        self.verifier = TimeoutSymbolicVerifier(timeout_seconds=0.5)

    def search(self, query: str, top_k: int = 10) -> Dict:
        norm_query = normalize_vi_to_en(query)
        q_vec = self.embedder.encode([norm_query])
        hits = self.index.search(q_vec, k=max(top_k, 50))

        candidates: List[CandidateItem] = []
        for rank_i, hit in enumerate(hits):
            cas_eq = None
            if rank_i < 10:
                vres = self.verifier.verify_pair(norm_query, hit["formula"])
                cas_eq = vres.status == VerificationStatus.EQUIV_PROVEN
            candidates.append(
                CandidateItem(
                    doc_id=hit["doc_id"],
                    text=hit["formula"],
                    neural_score=float(hit["score"]),
                    cas_equivalent=cas_eq,
                )
            )

        ranked = self.pipeline.rerank(norm_query, candidates)[:top_k]
        return {
            "query": query,
            "normalized_query": norm_query,
            "results": [
                {
                    "doc_id": c.doc_id,
                    "latex": c.text,
                    "neural_score": round(c.neural_score, 4),
                    "final_score": round(c.final_score, 4),
                    "gate_reasons": c.gate_reasons,
                    "cas_equivalent": c.cas_equivalent,
                }
                for c in ranked
            ],
        }


ENGINE: MathIRSearchEngine = None  # type: ignore


class APIHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "corpus_size": len(ENGINE.corpus)})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/search":
            self._send_json(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw)
            query = data.get("query", "").strip()
            top_k = int(data.get("top_k", 10))
            if not query:
                self._send_json(400, {"error": "Missing 'query'"})
                return
            res = ENGINE.search(query, top_k=top_k)
            self._send_json(200, res)
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def main():
    global ENGINE
    parser = argparse.ArgumentParser(description="MathIR Hybrid Search API Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "processed"))
    args = parser.parse_args()

    print(f"Initializing MathIRSearchEngine from {args.data_dir}...")
    ENGINE = MathIRSearchEngine(args.data_dir)
    print(f"MathIR API Server listening on http://127.0.0.1:{args.port}")
    HTTPServer(("0.0.0.0", args.port), APIHandler).serve_forever()


if __name__ == "__main__":
    main()
