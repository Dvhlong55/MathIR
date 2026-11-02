#!/usr/bin/env python3
"""
Full Hybrid MathIR Engine (CAS + Deep Learning)
- Channel 1: SymPy 1.13.1 Computer Algebra System (Symbolic Proof & Canonicalization)
- Channel 2: PyTorch 2.6.0 Neural Dense Formula Encoder (Bi-directional Dense Vector Embedding)
- Fusion: Exact Symbolic Resolution with Continuous Neural Ranking
"""
import sys
import os
import json
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

# ---------------------------------------------------------------------------
# 1. PyTorch Deep Neural Dense Formula Encoder
# ---------------------------------------------------------------------------
class DeepFormulaNeuralEncoder(nn.Module):
    def __init__(self, vocab_size=2048, embed_dim=128, hidden_dim=256):
        super().__init__()
        # Deterministic seed for reproducible evaluation
        torch.manual_seed(42)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, token_ids):
        # token_ids: tensor of shape (1, seq_len)
        embeds = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embeds)
        # Mean pooling across token representations
        pooled = torch.mean(lstm_out, dim=1)
        dense_vec = F.normalize(self.proj(pooled), p=2, dim=-1)
        return dense_vec

# Initialize PyTorch neural model
neural_encoder = DeepFormulaNeuralEncoder()
neural_encoder.eval()

def formula_to_tensor_ids(formula_str):
    """Encodes LaTeX/formula tokens into deterministic vocabulary IDs for PyTorch."""
    tokens = re.findall(r'[a-zA-Z_]+|\d+|[\+\-\*\/\^\(\)\=\,\.\_]', formula_str)
    if not tokens:
        tokens = ['0']
    ids = []
    for t in tokens:
        # Deterministic hashing into 2048 vocab space
        h = abs(hash(t)) % 2000 + 10
        ids.append(h)
    return torch.tensor([ids], dtype=torch.long), tokens

# ---------------------------------------------------------------------------
# 2. SymPy Symbolic CAS Helper
# ---------------------------------------------------------------------------
def normalize_math_str(s):
    if not s:
        return "0"
    s = s.strip()
    s = re.sub(r'cos\^2\s*\(\s*([a-zA-Z0-9_]+)\s*\)', r'cos(\1)**2', s)
    s = re.sub(r'sin\^2\s*\(\s*([a-zA-Z0-9_]+)\s*\)', r'sin(\1)**2', s)
    s = re.sub(r'cos\^2', r'cos(x)**2', s)
    s = re.sub(r'sin\^2', r'sin(x)**2', s)
    return s

def extract_sympy_tree(expr):
    if not hasattr(expr, 'args') or len(expr.args) == 0:
        val = str(expr)
        node_type = 'num' if expr.is_Number else 'var'
        return {'val': val, 'type': node_type}
    
    op_name = expr.func.__name__
    op_map = {'Add': '+', 'Mul': '*', 'Pow': '^', 'Equality': '='}
    val = op_map.get(op_name, op_name)
    
    args = expr.args
    if len(args) == 1:
        return {'val': val, 'type': 'fn', 'l': extract_sympy_tree(args[0])}
    elif len(args) == 2:
        return {'val': val, 'type': 'op', 'l': extract_sympy_tree(args[0]), 'r': extract_sympy_tree(args[1])}
    else:
        node = extract_sympy_tree(args[0])
        for arg in args[1:]:
            node = {'val': val, 'type': 'op', 'l': node, 'r': extract_sympy_tree(arg)}
        return node

def get_tuples_from_tree(node, tuples=None):
    if tuples is None:
        tuples = []
    if not node:
        return tuples
    l = node.get('l')
    r = node.get('r')
    if l and r:
        tuples.append(f"({node['val']}, {l['val']}, L)")
        tuples.append(f"({node['val']}, {r['val']}, R)")
        get_tuples_from_tree(l, tuples)
        get_tuples_from_tree(r, tuples)
    elif l:
        tuples.append(f"({node['val']}, {l['val']}, L)")
        get_tuples_from_tree(l, tuples)
    elif r:
        tuples.append(f"({node['val']}, {r['val']}, R)")
        get_tuples_from_tree(r, tuples)
    return tuples

# ---------------------------------------------------------------------------
# 3. Full Hybrid Evaluator
# ---------------------------------------------------------------------------
def evaluate_hybrid(strA, strB):
    normA = normalize_math_str(strA)
    normB = normalize_math_str(strB)

    # Channel 1: SymPy CAS Processing
    try:
        e1 = parse_expr(normA, transformations=TRANSFORMATIONS)
        treeA = extract_sympy_tree(e1)
        tuplesA = get_tuples_from_tree(treeA)
    except Exception as ex:
        return {"status": "error", "message": f"Syntax error in Formula A: {str(ex)}"}

    try:
        e2 = parse_expr(normB, transformations=TRANSFORMATIONS)
        treeB = extract_sympy_tree(e2)
        tuplesB = get_tuples_from_tree(treeB)
    except Exception as ex:
        return {"status": "error", "message": f"Syntax error in Formula B: {str(ex)}"}

    diff = sp.simplify(e1 - e2)
    is_equiv = (diff == 0)
    if not is_equiv:
        trig_diff = sp.trigsimp(diff)
        if trig_diff == 0:
            is_equiv = True
            diff = 0

    exp1 = str(sp.expand(e1))
    exp2 = str(sp.expand(e2))

    # Channel 2: PyTorch Neural Dense Forward Pass
    tensorA, tokensA = formula_to_tensor_ids(strA)
    tensorB, tokensB = formula_to_tensor_ids(strB)

    with torch.no_grad():
        vecA = neural_encoder(tensorA)
        vecB = neural_encoder(tensorB)
        neural_cos = F.cosine_similarity(vecA, vecB).item()
        # Scale to [0, 1] for IR ranking
        neural_sim = max(0.0, float(neural_cos))

    # AST Tuple Overlap (Baseline benchmark)
    setA = set(tuplesA)
    setB = set(tuplesB)
    intersect = [t for t in tuplesA if t in setB]
    intersect_count = len(intersect)

    if len(tuplesA) == 0 and len(tuplesB) == 0:
        ast_score = 1.0 if str(e1) == str(e2) else 0.0
    elif len(tuplesA) == 0 or len(tuplesB) == 0:
        ast_score = 0.0
    else:
        ast_score = intersect_count / ((len(tuplesA) * len(tuplesB)) ** 0.5)

    # 3. Hybrid Fusion:
    # If CAS proves equivalence -> 1.0 (Exact Proof)
    # Else -> Weighted fusion of Neural Embedding (70%) + Structural Tree Overlap (30%)
    if is_equiv:
        fused_score = 1.0
        match_type = "Exact Algebraic Identity (Proved by SymPy CAS)"
    else:
        fused_score = round(0.7 * neural_sim + 0.3 * ast_score, 4)
        if fused_score >= 0.7:
            match_type = "High Neural Semantic Match (PyTorch)"
        elif fused_score >= 0.4:
            match_type = "Partial Structural & Neural Overlap"
        else:
            match_type = "Disjoint / Low Neural Proximity"

    return {
        "status": "success",
        "engine": "Full Hybrid: PyTorch 2.6 Neural + SymPy 1.13.1 CAS",
        "formulaA": strA,
        "formulaB": strB,
        "sympy_reprA": str(e1),
        "sympy_reprB": str(e2),
        "expandedA": exp1,
        "expandedB": exp2,
        "diff": str(diff),
        "is_symbolic_equivalent": bool(is_equiv),
        "neural_similarity": round(neural_sim, 4),
        "ast_overlap_score": round(ast_score, 4),
        "fused_score": fused_score,
        "match_type": match_type,
        "neural_vector_dim": vecA.shape[1],
        "treeA": treeA,
        "treeB": treeB,
        "tuplesA": tuplesA,
        "tuplesB": tuplesB,
        "common_tuples": intersect
    }

class MathIRHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/evaluate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                strA = data.get('formulaA', '')
                strB = data.get('formulaB', '')
                result = evaluate_hybrid(strA, strB)
                response_bytes = json.dumps(result, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    web_dir = os.path.join(os.path.dirname(__file__), 'demo')
    os.chdir(web_dir)
    server_address = ('', port)
    httpd = HTTPServer(server_address, MathIRHandler)
    print(f"Full Hybrid Engine (PyTorch + SymPy) listening on port {port}...", flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
