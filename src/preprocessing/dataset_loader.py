"""
Dataset loader and parser for MathIR benchmarks (ARQMath & NTCIR via MIRB HuggingFace).
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

This module handles:
1. Direct streaming / downloading of MathIR datasets from HuggingFace (hcju/mseformula, hcju/wikiformula, hcju/msedup).
2. Local caching in `data/raw/` to avoid redundant network requests.
3. Parsing corpus, queries, and qrels (relevance judgments).
4. Extracting and canonicalizing formula spans from question and answer context.
"""

import os
import re
import json
import ssl
import urllib.request
from typing import Dict, List, Optional, Tuple, Any


# Standard public HuggingFace datasets for MathIR from MIRB benchmark
DEFAULT_DATASETS = {
    "mseformula": {
        "repo_id": "hcju/mseformula",
        "description": "ARQMath Task 2 (Math Stack Exchange Formula Retrieval)",
        "files": {
            "corpus": "https://huggingface.co/datasets/hcju/mseformula/resolve/main/corpus.jsonl",
            "queries": "https://huggingface.co/datasets/hcju/mseformula/resolve/main/queries.jsonl",
            "qrels": "https://huggingface.co/datasets/hcju/mseformula/resolve/main/qrels/test.jsonl",
        }
    },
    "wikiformula": {
        "repo_id": "hcju/wikiformula",
        "description": "NTCIR-12 MathIR Wiki Formula Browsing Task",
        "files": {
            "corpus": "https://huggingface.co/datasets/hcju/wikiformula/resolve/main/corpus.jsonl",
            "queries": "https://huggingface.co/datasets/hcju/wikiformula/resolve/main/queries.jsonl",
            "qrels": "https://huggingface.co/datasets/hcju/wikiformula/resolve/main/qrels/test.jsonl",
        }
    }
}


def download_file_with_progress(url: str, dest_path: str) -> None:
    """Downloads a file directly from a URL with an unverified SSL fallback context.
    
    Ensures portability across diverse OS and Python distributions without
    requiring manual SSL certificate setup on Windows/MinGW.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MathIR-BenchmarkLoader/1.0 (HCMUS APCS SC203)"}
    )
    
    print(f"[MathIR] Downloading: {url} -> {dest_path}...")
    with urllib.request.urlopen(req, context=ctx) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1024 * 64
        total_downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            total_downloaded += len(chunk)
    print(f"[MathIR] Completed: {dest_path} ({total_downloaded / 1024:.1f} KB)")


UNICODE_MATH_MAP = {
    '≤': r'\le ',
    '≥': r'\ge ',
    '≠': r'\neq ',
    '≈': r'\approx ',
    '≡': r'\equiv ',
    '×': r'\times ',
    '·': r'\cdot ',
    '÷': r'\div ',
    '±': r'\pm ',
    '∓': r'\mp ',
    '∞': r'\infty ',
    '∈': r'\in ',
    '∉': r'\notin ',
    '⊂': r'\subset ',
    '⊆': r'\subseteq ',
    '∪': r'\cup ',
    '∩': r'\cap ',
    '∀': r'\forall ',
    '∃': r'\exists ',
    '→': r'\to ',
    '⇒': r'\Rightarrow ',
    '⇔': r'\Leftrightarrow ',
    '∑': r'\sum ',
    '∏': r'\prod ',
    '∫': r'\int ',
    '√': r'\sqrt ',
    'α': r'\alpha ',
    'β': r'\beta ',
    'γ': r'\gamma ',
    'θ': r'\theta ',
    'λ': r'\lambda ',
    'π': r'\pi ',
    'σ': r'\sigma ',
    'φ': r'\phi ',
    'ω': r'\omega ',
    'Δ': r'\Delta ',
    'Γ': r'\Gamma ',
    'Ω': r'\Omega ',
}


def clean_latex(latex_str: str) -> str:
    """Removes standard markdown/LaTeX inline delimiters ($...$, $$...$$, \\(...\\), \\[...]\\)
    and canonicalizes Unicode mathematical symbols to standard LaTeX macros.
    """
    if not latex_str:
        return ""
    s = latex_str.strip()
    s = re.sub(r'^\$\$+|\$\$+$', '', s)
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\\[|\\\]$', '', s)
    s = re.sub(r'^\\\(|\\\)$', '', s)
    
    # Replace Unicode math symbols with LaTeX commands for parser robustness
    for uchar, lmacro in UNICODE_MATH_MAP.items():
        if uchar in s:
            s = s.replace(uchar, lmacro)
            
    return s.strip()


def extract_formulas_from_text(text: str) -> List[str]:
    """Extracts all embedded LaTeX formulas from a text containing $...$ or $$...$$."""
    if not text:
        return []
    
    # 1. Match display math $$...$$ or \[...\]
    display_matches = re.findall(r'\$\$([\s\S]*?)\$\$', text)
    display_matches += re.findall(r'\\\[([\s\S]*?)\\\]', text)
    
    # Remove matched display blocks to avoid double counting inline
    clean_text = re.sub(r'\$\$[\s\S]*?\$\$', ' ', text)
    clean_text = re.sub(r'\\\[[\s\S]*?\\\]', ' ', clean_text)
    
    # 2. Match inline math $...$ or \(...\)
    inline_matches = re.findall(r'\$([^$]+)\$', clean_text)
    inline_matches += re.findall(r'\\\(([\s\S]*?)\\\)', clean_text)
    
    all_formulas = [clean_latex(f) for f in (display_matches + inline_matches)]
    return [f for f in all_formulas if len(f) > 0]


class MathBenchmarkLoader:
    """Manages loading, downloading, and caching of standardized MathIR benchmark datasets."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initializes loader.
        
        Args:
            cache_dir: Directory where raw JSONL files are stored. Defaults to `data/raw`.
        """
        if cache_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            cache_dir = os.path.join(project_root, "data", "raw")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_dataset_files(self, dataset_name: str) -> Dict[str, str]:
        """Ensures all files for the specified dataset are downloaded locally and returns their paths."""
        if dataset_name not in DEFAULT_DATASETS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DEFAULT_DATASETS.keys())}")
        
        ds_info = DEFAULT_DATASETS[dataset_name]
        dataset_cache = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(dataset_cache, exist_ok=True)
        
        local_paths = {}
        for split, url in ds_info["files"].items():
            filename = f"{split}.jsonl"
            dest_path = os.path.join(dataset_cache, filename)
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
                download_file_with_progress(url, dest_path)
            local_paths[split] = dest_path
        
        return local_paths

    def load_corpus(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Loads corpus items as structured dictionaries: {doc_id, latex, context}."""
        files = self.get_dataset_files(dataset_name)
        corpus = []
        with open(files["corpus"], "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_text = item.get("text", "")
                formulas = extract_formulas_from_text(raw_text)
                
                # If no $ delimiters were found, the raw text itself is the formula
                primary_latex = clean_latex(formulas[0]) if formulas else clean_latex(raw_text)
                
                corpus.append({
                    "doc_id": str(item.get("_id", "")),
                    "title": item.get("title", ""),
                    "latex": primary_latex,
                    "all_formulas": formulas,
                    "context": raw_text,
                    "source_dataset": dataset_name
                })
        return corpus

    def load_queries(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Loads query items: {query_id, formula, context, raw_text}."""
        files = self.get_dataset_files(dataset_name)
        queries = []
        with open(files["queries"], "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_text = item.get("text", "")
                
                # In ARQMath Task 2 / MIRB queries, the format typically starts with:
                # 'Formula: $...$\nContext: ...'
                formula = ""
                context = ""
                if "Formula:" in raw_text:
                    parts = raw_text.split("Context:", 1)
                    formula_part = parts[0].replace("Formula:", "").strip()
                    formulas = extract_formulas_from_text(formula_part)
                    formula = clean_latex(formulas[0]) if formulas else clean_latex(formula_part)
                    if len(parts) > 1:
                        context = parts[1].strip()
                else:
                    formulas = extract_formulas_from_text(raw_text)
                    formula = clean_latex(formulas[0]) if formulas else clean_latex(raw_text)
                    context = raw_text
                
                queries.append({
                    "query_id": str(item.get("_id", "")),
                    "formula": formula,
                    "context": context,
                    "raw_text": raw_text,
                    "source_dataset": dataset_name
                })
        return queries

    def load_qrels(self, dataset_name: str) -> Dict[str, Dict[str, float]]:
        """Loads ground truth relevance judgements: {query_id: {corpus_id: relevance_score}}."""
        files = self.get_dataset_files(dataset_name)
        qrels: Dict[str, Dict[str, float]] = {}
        with open(files["qrels"], "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rel = json.loads(line)
                qid = str(rel.get("query-id", ""))
                cid = str(rel.get("corpus-id", ""))
                score = float(rel.get("score", 1.0))
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][cid] = score
        return qrels
