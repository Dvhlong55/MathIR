"""Stage 3: Timeout-guarded SymPy CAS verification and bounded score boost.

Provides:
- LaTeX noise cleaning & early detection of unsupported informal syntax (e.g. ellipsis `...`).
- Windows-safe per-candidate timeout using a persistent spawned worker process.
- Parse/verification telemetry (`VerificationStats`) for ARQMath noisy-LaTeX auditing.
- Validation-tunable bounded score boost (`apply_bounded_boost`) replacing unconditional Rank-1 promotion.
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Dict, Iterable, List, Optional, Tuple


class VerificationStatus(str, Enum):
    """Outcome categories for Stage 3 symbolic verification."""

    EQUIV_PROVEN = "EQUIV_PROVEN"
    NOT_EQUIV = "NOT_EQUIV"
    UNSUPPORTED_SYNTAX = "UNSUPPORTED_SYNTAX"
    PARSE_ERROR = "PARSE_ERROR"
    TIMEOUT = "TIMEOUT"
    SIMPLIFY_ERROR = "SIMPLIFY_ERROR"


@dataclass
class VerificationResult:
    """Detailed result for a single (query, candidate) symbolic check."""

    status: VerificationStatus
    is_symbolic_equivalent: bool
    parse_success_query: bool
    parse_success_cand: bool
    diff_repr: str
    latency_ms: float
    error_detail: Optional[str] = None
    query_expr_repr: Optional[str] = None
    cand_expr_repr: Optional[str] = None


@dataclass
class VerificationStats:
    """Aggregated Stage 3 parse reliability and latency telemetry."""

    total: int = 0
    equiv_proven: int = 0
    not_equiv: int = 0
    unsupported_syntax: int = 0
    parse_error: int = 0
    timeout: int = 0
    simplify_error: int = 0
    query_parse_success: int = 0
    cand_parse_success: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    error_reasons: Counter = field(default_factory=Counter)

    def record(self, result: VerificationResult) -> None:
        """Records one `VerificationResult` into aggregate statistics."""
        self.total += 1
        self.latencies_ms.append(result.latency_ms)
        if result.parse_success_query:
            self.query_parse_success += 1
        if result.parse_success_cand:
            self.cand_parse_success += 1

        status_attr = {
            VerificationStatus.EQUIV_PROVEN: "equiv_proven",
            VerificationStatus.NOT_EQUIV: "not_equiv",
            VerificationStatus.UNSUPPORTED_SYNTAX: "unsupported_syntax",
            VerificationStatus.PARSE_ERROR: "parse_error",
            VerificationStatus.TIMEOUT: "timeout",
            VerificationStatus.SIMPLIFY_ERROR: "simplify_error",
        }[result.status]
        setattr(self, status_attr, getattr(self, status_attr) + 1)

        if result.error_detail:
            key = result.error_detail.split(":", 1)[0].strip()[:120]
            self.error_reasons[key] += 1

    def summary(self) -> Dict[str, Any]:
        """Returns parse success rates, outcome breakdown, and latency percentiles."""
        total = self.total or 1
        sorted_lat = sorted(self.latencies_ms)
        n = len(sorted_lat)

        def _percentile(p: float) -> float:
            if not sorted_lat:
                return 0.0
            idx = min(n - 1, max(0, int(math.ceil(p * n)) - 1))
            return round(sorted_lat[idx], 3)

        both_parsed = self.equiv_proven + self.not_equiv + self.simplify_error
        return {
            "total_pairs": self.total,
            "both_parse_success_rate": round(both_parsed / total, 4),
            "query_parse_success_rate": round(self.query_parse_success / total, 4),
            "cand_parse_success_rate": round(self.cand_parse_success / total, 4),
            "counts": {
                "equiv_proven": self.equiv_proven,
                "not_equiv": self.not_equiv,
                "unsupported_syntax": self.unsupported_syntax,
                "parse_error": self.parse_error,
                "timeout": self.timeout,
                "simplify_error": self.simplify_error,
            },
            "latency_ms": {
                "p50": _percentile(0.50),
                "p95": _percentile(0.95),
                "max": round(sorted_lat[-1], 3) if sorted_lat else 0.0,
            },
            "top_error_reasons": self.error_reasons.most_common(10),
        }


_ELLIPSIS_RE = re.compile(r"\.\.\.|\\(?:dots|ldots|cdots|vdots|ddots)\b")
_PRESENTATION_MACROS_RE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits|left|right|big|Big|bigg|Bigg)\b"
)
_LATEX_SPACING_RE = re.compile(r"\\[,;:!]|\\q?quad\b")
_INLINE_DOLLAR_RE = re.compile(r"\$\$([\s\S]*?)\$\$|\$([^$]+)\$")


def has_unsupported_syntax(raw_str: str) -> Optional[str]:
    """Detects empty inputs or informal notation (e.g. ellipsis `...`) before CAS parsing."""
    if not raw_str or not raw_str.strip():
        return "empty_input"
    if _ELLIPSIS_RE.search(raw_str):
        return "ellipsis_sequence"
    return None


_TRIG_POWER_RE = re.compile(
    r"\\(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh)\s*\^\s*\{?([0-9]+)\}?\s*\(\s*([^()]+)\s*\)"
)


def clean_latex_noise(expr_str: str) -> str:
    """Strips presentation-only LaTeX tokens while preserving valid LaTeX syntax."""
    s = expr_str.strip()
    s = _PRESENTATION_MACROS_RE.sub("", s)
    s = _LATEX_SPACING_RE.sub(" ", s)
    # Disambiguate trig powers like \sin^2(x) -> (\sin(x))^2 for Lark LaTeX grammar
    s = _TRIG_POWER_RE.sub(r"(\\\1(\3))^\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_math_candidates(raw_str: str) -> List[str]:
    """Extracts LaTeX/math segments from mixed text+formula strings."""
    s = raw_str.strip()
    s = re.sub(r"\\\[([\s\S]*?)\\\]", r"$\1$", s)
    s = re.sub(r"\\\(([\s\S]*?)\\\)", r"$\1$", s)
    matches = _INLINE_DOLLAR_RE.findall(s)
    if matches:
        blocks = [m[0] or m[1] for m in matches if (m[0] or m[1]).strip()]
        return sorted(blocks, key=len, reverse=True)
    stripped = re.sub(r"^\$+|\$+$", "", s).strip()
    return [stripped] if stripped else []


def _parse_latex_with_fallback(cand: str, preferred_backend: str = "lark") -> Tuple[Any, Optional[str]]:
    """Attempts SymPy `parse_latex` (with backend fallback) then `parse_expr`."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    cleaned = clean_latex_noise(cand)
    looks_like_latex = "\\" in cleaned or "{" in cleaned or "_" in cleaned

    parse_errors: List[str] = []
    if looks_like_latex:
        try:
            from sympy.parsing.latex import parse_latex
        except Exception as exc:
            parse_latex = None
            parse_errors.append(f"import_parse_latex:{exc}")

        if parse_latex is not None:
            available_backends: List[str] = []
            for candidate_backend in (preferred_backend, "lark", "antlr"):
                if candidate_backend in available_backends:
                    continue
                if candidate_backend == "lark" and importlib.util.find_spec("lark") is not None:
                    available_backends.append("lark")
                elif candidate_backend == "antlr" and importlib.util.find_spec("antlr4") is not None:
                    available_backends.append("antlr")

            for backend in available_backends:
                try:
                    parsed = parse_latex(cleaned, backend=backend)
                    if isinstance(parsed, sp.Basic):
                        return parsed, None
                    parse_errors.append(
                        f"parse_latex_{backend}:AmbiguousOrNonBasic:{type(parsed).__name__}"
                    )
                except TypeError:
                    try:
                        parsed = parse_latex(cleaned)
                        if isinstance(parsed, sp.Basic):
                            return parsed, None
                        parse_errors.append(f"parse_latex:NonBasic:{type(parsed).__name__}")
                    except Exception as exc:
                        parse_errors.append(f"parse_latex:{type(exc).__name__}:{exc}")
                        break
                except Exception as exc:
                    parse_errors.append(f"parse_latex_{backend}:{type(exc).__name__}:{exc}")

    # Normalize bare trig^2 shortcuts for plain-text parse_expr fallback
    plain = re.sub(r"cos\^2\s*\(\s*([a-zA-Z0-9_]+)\s*\)", r"cos(\1)**2", cleaned)
    plain = re.sub(r"sin\^2\s*\(\s*([a-zA-Z0-9_]+)\s*\)", r"sin(\1)**2", plain)
    plain = re.sub(r"\bcos\^2\b", "cos(x)**2", plain)
    plain = re.sub(r"\bsin\^2\b", "sin(x)**2", plain)
    try:
        return parse_expr(plain, transformations=transformations), None
    except Exception as exc:
        parse_errors.append(f"parse_expr:{type(exc).__name__}:{exc}")
        return None, " | ".join(parse_errors)


def parse_formula_safe(
    raw_str: str, preferred_backend: str = "lark"
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """Parses a raw math/text query into a SymPy object.

    Returns:
        `(sympy_expr, unsupported_reason, parse_error_detail)`
    """
    unsupported = has_unsupported_syntax(raw_str)
    if unsupported:
        return None, unsupported, None

    candidates = extract_math_candidates(raw_str)
    if not candidates:
        return None, "empty_input", None

    parsed_non_rel: List[Any] = []
    parsed_rel: List[Any] = []
    last_error: Optional[str] = None

    for cand in candidates:
        if has_unsupported_syntax(cand):
            return None, "ellipsis_sequence", None
        expr, err = _parse_latex_with_fallback(cand, preferred_backend=preferred_backend)
        if expr is None:
            last_error = err
            continue
        if getattr(expr, "is_Equality", False) and getattr(expr.lhs, "is_Symbol", False):
            expr = expr.rhs
        if getattr(expr, "is_Relational", False):
            parsed_rel.append(expr)
        else:
            parsed_non_rel.append(expr)

    if parsed_non_rel:
        return parsed_non_rel[0], None, None
    if parsed_rel:
        return parsed_rel[0], None, None
    return None, None, last_error or "parse_failed"


def _canonicalize_relational(rel: Any) -> Tuple[str, Any, Any]:
    """Normalizes `<` / `<=` into `>` / `>=` by swapping sides so `x > 0` matches `0 < x`."""
    op = rel.rel_op
    if op == "<":
        return ">", rel.rhs, rel.lhs
    if op == "<=":
        return ">=", rel.rhs, rel.lhs
    return op, rel.lhs, rel.rhs


def evaluate_pair_core(
    str_a: str, str_b: str, preferred_backend: str = "lark"
) -> VerificationResult:
    """Pure in-process SymPy comparison without process timeout wrapper."""
    import sympy as sp

    t0 = time.perf_counter()
    expr_a, unsup_a, err_a = parse_formula_safe(str_a, preferred_backend=preferred_backend)
    expr_b, unsup_b, err_b = parse_formula_safe(str_b, preferred_backend=preferred_backend)

    parse_ok_a = expr_a is not None
    parse_ok_b = expr_b is not None

    if unsup_a or unsup_b:
        reason = unsup_a or unsup_b
        return VerificationResult(
            status=VerificationStatus.UNSUPPORTED_SYNTAX,
            is_symbolic_equivalent=False,
            parse_success_query=parse_ok_a,
            parse_success_cand=parse_ok_b,
            diff_repr=f"Unsupported ({reason})",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_detail=f"unsupported:{reason}",
            query_expr_repr=str(expr_a) if parse_ok_a else None,
            cand_expr_repr=str(expr_b) if parse_ok_b else None,
        )

    if not parse_ok_a or not parse_ok_b:
        detail = err_a if not parse_ok_a else err_b
        return VerificationResult(
            status=VerificationStatus.PARSE_ERROR,
            is_symbolic_equivalent=False,
            parse_success_query=parse_ok_a,
            parse_success_cand=parse_ok_b,
            diff_repr="Parse Error",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_detail=f"parse_error:{detail}",
            query_expr_repr=str(expr_a) if parse_ok_a else None,
            cand_expr_repr=str(expr_b) if parse_ok_b else None,
        )

    try:
        rel_a = getattr(expr_a, "is_Relational", False)
        rel_b = getattr(expr_b, "is_Relational", False)

        if rel_a or rel_b:
            if not (rel_a and rel_b):
                return VerificationResult(
                    status=VerificationStatus.NOT_EQUIV,
                    is_symbolic_equivalent=False,
                    parse_success_query=True,
                    parse_success_cand=True,
                    diff_repr="Type mismatch: Relational vs Expression",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    query_expr_repr=str(expr_a),
                    cand_expr_repr=str(expr_b),
                )

            op_a, lhs_a, rhs_a = _canonicalize_relational(expr_a)
            op_b, lhs_b, rhs_b = _canonicalize_relational(expr_b)
            if op_a != op_b:
                return VerificationResult(
                    status=VerificationStatus.NOT_EQUIV,
                    is_symbolic_equivalent=False,
                    parse_success_query=True,
                    parse_success_cand=True,
                    diff_repr=f"Relation mismatch: {op_a} vs {op_b}",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    query_expr_repr=str(expr_a),
                    cand_expr_repr=str(expr_b),
                )

            if op_a in ("==", "!="):
                diff_rel = sp.simplify((lhs_a - rhs_a) - (lhs_b - rhs_b))
                alt_rel = sp.simplify((lhs_a - rhs_a) + (lhs_b - rhs_b))
                is_equiv = diff_rel == 0 or alt_rel == 0
                diff_str = "0" if is_equiv else str(diff_rel)
            else:
                diff_rel = sp.simplify((lhs_a - rhs_a) - (lhs_b - rhs_b))
                is_equiv = diff_rel == 0
                diff_str = "0" if is_equiv else str(diff_rel)

            return VerificationResult(
                status=VerificationStatus.EQUIV_PROVEN if is_equiv else VerificationStatus.NOT_EQUIV,
                is_symbolic_equivalent=bool(is_equiv),
                parse_success_query=True,
                parse_success_cand=True,
                diff_repr=diff_str,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                query_expr_repr=str(expr_a),
                cand_expr_repr=str(expr_b),
            )

        diff = sp.simplify(expr_a - expr_b)
        is_equiv = diff == 0
        if not is_equiv:
            trig_diff = sp.trigsimp(diff)
            if trig_diff == 0:
                is_equiv = True
                diff = 0

        return VerificationResult(
            status=VerificationStatus.EQUIV_PROVEN if is_equiv else VerificationStatus.NOT_EQUIV,
            is_symbolic_equivalent=bool(is_equiv),
            parse_success_query=True,
            parse_success_cand=True,
            diff_repr=str(diff),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            query_expr_repr=str(expr_a),
            cand_expr_repr=str(expr_b),
        )
    except Exception as exc:
        return VerificationResult(
            status=VerificationStatus.SIMPLIFY_ERROR,
            is_symbolic_equivalent=False,
            parse_success_query=True,
            parse_success_cand=True,
            diff_repr=f"Simplify Error: {exc}",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error_detail=f"simplify_error:{type(exc).__name__}:{exc}",
            query_expr_repr=str(expr_a),
            cand_expr_repr=str(expr_b),
        )


def _worker_loop(
    req_conn: Connection, resp_conn: Connection, preferred_backend: str
) -> None:
    """Persistent child worker loop so Windows avoids process-spawn overhead per candidate."""
    # Pre-import SymPy & warm both LaTeX and plain-expr parsers before signaling readiness
    import sympy as sp
    try:
        from sympy.parsing.latex import parse_latex

        parse_latex("x", backend=preferred_backend)
    except Exception:
        pass
    try:
        evaluate_pair_core("(x + 1)^2", "x^2 + 2*x + 1", preferred_backend=preferred_backend)
        sp.simplify(sp.Symbol("x") + 1)
    except Exception:
        pass
    resp_conn.send("READY")

    while True:
        try:
            payload = req_conn.recv()
        except EOFError:
            break
        if payload is None:
            break
        str_a, str_b = payload
        result = evaluate_pair_core(str_a, str_b, preferred_backend=preferred_backend)
        resp_conn.send(result)


class TimeoutSymbolicVerifier:
    """Windows-safe Stage 3 SymPy verifier with per-candidate timeout and telemetry."""

    def __init__(
        self,
        timeout_seconds: float = 0.75,
        latex_backend: str = "lark",
        worker_startup_timeout: float = 15.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.latex_backend = latex_backend
        self.worker_startup_timeout = float(worker_startup_timeout)
        self.stats = VerificationStats()
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: Optional[multiprocessing.Process] = None
        self._req_send: Optional[Connection] = None
        self._resp_recv: Optional[Connection] = None

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._cleanup_worker(terminate=False)
        req_recv, req_send = self._ctx.Pipe(duplex=False)
        resp_recv, resp_send = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_worker_loop,
            args=(req_recv, resp_send, self.latex_backend),
            daemon=True,
        )
        proc.start()
        req_recv.close()
        resp_send.close()
        if not resp_recv.poll(self.worker_startup_timeout) or resp_recv.recv() != "READY":
            proc.terminate()
            proc.join(timeout=1.0)
            req_send.close()
            resp_recv.close()
            raise RuntimeError("SymPy worker process failed to initialize within startup timeout")
        self._proc = proc
        self._req_send = req_send
        self._resp_recv = resp_recv

    def _cleanup_worker(self, terminate: bool = True) -> None:
        if self._req_send is not None:
            try:
                if not terminate:
                    self._req_send.send(None)
            except Exception:
                pass
            try:
                self._req_send.close()
            except Exception:
                pass
            self._req_send = None

        if self._resp_recv is not None:
            try:
                self._resp_recv.close()
            except Exception:
                pass
            self._resp_recv = None

        if self._proc is not None:
            if terminate and self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=1.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=1.0)
            self._proc = None

    def verify_pair(self, query_formula: str, cand_formula: str) -> VerificationResult:
        """Verifies algebraic equivalence within `timeout_seconds` and records stats."""
        # Fast-path syntax check in parent avoids IPC wait on obvious unsupported strings
        unsup_q = has_unsupported_syntax(query_formula)
        unsup_c = has_unsupported_syntax(cand_formula)
        if unsup_q or unsup_c:
            reason = unsup_q or unsup_c
            res = VerificationResult(
                status=VerificationStatus.UNSUPPORTED_SYNTAX,
                is_symbolic_equivalent=False,
                parse_success_query=unsup_q is None,
                parse_success_cand=unsup_c is None,
                diff_repr=f"Unsupported ({reason})",
                latency_ms=0.0,
                error_detail=f"unsupported:{reason}",
            )
            self.stats.record(res)
            return res

        self._ensure_worker()
        assert self._req_send is not None and self._resp_recv is not None
        t0 = time.perf_counter()

        try:
            self._req_send.send((query_formula, cand_formula))
            if self._resp_recv.poll(self.timeout_seconds):
                res = self._resp_recv.recv()
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._cleanup_worker(terminate=True)
                res = VerificationResult(
                    status=VerificationStatus.TIMEOUT,
                    is_symbolic_equivalent=False,
                    parse_success_query=False,
                    parse_success_cand=False,
                    diff_repr=f"Timeout after {self.timeout_seconds:.2f}s",
                    latency_ms=elapsed_ms,
                    error_detail=f"timeout:>{self.timeout_seconds:.2f}s",
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._cleanup_worker(terminate=True)
            res = VerificationResult(
                status=VerificationStatus.SIMPLIFY_ERROR,
                is_symbolic_equivalent=False,
                parse_success_query=False,
                parse_success_cand=False,
                diff_repr=f"Worker Error: {exc}",
                latency_ms=elapsed_ms,
                error_detail=f"worker_error:{type(exc).__name__}:{exc}",
            )

        self.stats.record(res)
        return res

    def check_single_parse(self, raw_str: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Audits parse success on a single ARQMath LaTeX expression without pair diff."""
        expr, unsup, err = parse_formula_safe(raw_str, preferred_backend=self.latex_backend)
        return expr is not None, unsup, err

    def rerank_candidates(
        self,
        query_formula: str,
        candidates: Iterable[Dict[str, Any]],
        formula_key: str = "formula",
        score_key: str = "stage12_score",
        boost_mode: str = "residual",
        beta: float = 0.2,
        min_context_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Applies Stage 3 verification + bounded score boost to Stage 1+2 candidates."""
        reranked: List[Dict[str, Any]] = []
        for item in candidates:
            entry = dict(item)
            base_score = float(entry.get(score_key, 0.0))
            cand_formula = str(entry.get(formula_key, ""))
            ver_res = self.verify_pair(query_formula, cand_formula)
            final_score = apply_bounded_boost(
                base_score=base_score,
                is_symbolic_equivalent=ver_res.is_symbolic_equivalent,
                mode=boost_mode,
                beta=beta,
                min_context_score=min_context_score,
            )
            entry["stage3_status"] = ver_res.status.value
            entry["is_symbolic_equivalent"] = ver_res.is_symbolic_equivalent
            entry["stage3_latency_ms"] = round(ver_res.latency_ms, 3)
            entry["final_score"] = final_score
            reranked.append(entry)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    def close(self) -> None:
        """Gracefully shuts down the background SymPy worker process."""
        self._cleanup_worker(terminate=False)

    def __enter__(self) -> "TimeoutSymbolicVerifier":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def apply_bounded_boost(
    base_score: float,
    is_symbolic_equivalent: bool,
    mode: str = "residual",
    beta: float = 0.2,
    min_context_score: float = 0.0,
) -> float:
    """Applies a bounded validation-tuned score boost for Delta=0 pairs.

    Modes:
        - `residual`: `s + beta * (1 - s)` when equivalent and `s >= min_context_score`.
          Preserves contextual ordering from Stage 1+2 while keeping scores in `[0, 1]`.
        - `clipped_additive`: `min(1.0, s + beta)` when equivalent and `s >= min_context_score`.
        - `hard_rank1`: `1.0` when equivalent (ablation baseline reproducing `server.py`).
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0.0, 1.0]")
    if not (0.0 <= min_context_score <= 1.0):
        raise ValueError("min_context_score must be in [0.0, 1.0]")

    clamped = max(0.0, min(1.0, float(base_score)))
    if not is_symbolic_equivalent:
        return round(clamped, 6)

    if mode == "hard_rank1":
        return 1.0
    if clamped < min_context_score:
        return round(clamped, 6)

    if mode == "residual":
        boosted = clamped + beta * (1.0 - clamped)
    elif mode == "clipped_additive":
        boosted = min(1.0, clamped + beta)
    else:
        raise ValueError(f"Unknown boost mode: {mode}")

    return round(max(0.0, min(1.0, boosted)), 6)
