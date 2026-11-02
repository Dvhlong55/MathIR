#!/usr/bin/env python3
"""
Real MathIR Backend Server powered by SymPy 1.13.1 and AST Structural Analysis.
Serves static files and provides /api/evaluate endpoint with CORS support.
"""
import sys
import os
import json
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

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
    """Recursively converts a SymPy expression into a JSON tree for SVG rendering."""
    if not hasattr(expr, 'args') or len(expr.args) == 0:
        val = str(expr)
        node_type = 'num' if expr.is_Number else 'var'
        return {'val': val, 'type': node_type}
    
    op_name = expr.func.__name__
    # Human-friendly operator symbols
    op_map = {'Add': '+', 'Mul': '*', 'Pow': '^', 'Equality': '='}
    val = op_map.get(op_name, op_name)
    
    args = expr.args
    if len(args) == 1:
        return {'val': val, 'type': 'fn', 'l': extract_sympy_tree(args[0])}
    elif len(args) == 2:
        return {'val': val, 'type': 'op', 'l': extract_sympy_tree(args[0]), 'r': extract_sympy_tree(args[1])}
    else:
        # Multi-arg associative op: chain into binary tree
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

def evaluate(strA, strB):
    normA = normalize_math_str(strA)
    normB = normalize_math_str(strB)

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

    # 1. Real Symbolic CAS Evaluation (SymPy)
    diff = sp.simplify(e1 - e2)
    is_equiv = (diff == 0)
    if not is_equiv:
        trig_diff = sp.trigsimp(diff)
        if trig_diff == 0:
            is_equiv = True
            diff = 0

    exp1 = str(sp.expand(e1))
    exp2 = str(sp.expand(e2))

    # 2. Real Syntactic Tuple Overlap (Tangent-CFT Baseline)
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

    # 3. Hybrid Fusion
    if is_equiv:
        fused_score = 1.0
        match_type = "Exact Algebraic Identity (Proved by SymPy)"
    elif ast_score > 0:
        fused_score = round(ast_score, 4)
        match_type = "Partial Structural Match (Sub-tree Overlap)"
    else:
        fused_score = 0.0
        match_type = "Algebraically & Structurally Disjoint"

    return {
        "status": "success",
        "engine": "SymPy 1.13.1 + MathIR AST",
        "formulaA": strA,
        "formulaB": strB,
        "sympy_reprA": str(e1),
        "sympy_reprB": str(e2),
        "expandedA": exp1,
        "expandedB": exp2,
        "diff": str(diff),
        "is_symbolic_equivalent": bool(is_equiv),
        "ast_overlap_score": round(ast_score, 4),
        "fused_score": fused_score,
        "match_type": match_type,
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
                result = evaluate(strA, strB)
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
    print(f"MathIR Real Python SymPy Server listening on port {port}...", flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
