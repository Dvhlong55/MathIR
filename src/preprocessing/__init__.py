"""Preprocessing package for MathIR."""

from .canonicalizer import FormulaCanonicalizer
from .dataset_loader import (
    MathBenchmarkLoader,
    clean_latex,
    extract_formulas_from_text,
    DEFAULT_DATASETS,
)
from .formula_sampler import (
    FormulaSampler,
    tag_formula_features,
)
from .vi_normalizer import (
    normalize_vi_to_en,
    mask_formulas,
    unmask_formulas,
    apply_math_dictionary,
    expand_shorthand_and_clean,
    strip_untranslated_vi_words,
    disambiguate_vi_math,
    canonicalize_en_math,
    LATEX_PATTERN,
)

__all__ = [
    "FormulaCanonicalizer",
    "MathBenchmarkLoader",
    "FormulaSampler",
    "clean_latex",
    "extract_formulas_from_text",
    "tag_formula_features",
    "DEFAULT_DATASETS",
    "normalize_vi_to_en",
    "mask_formulas",
    "unmask_formulas",
    "apply_math_dictionary",
    "expand_shorthand_and_clean",
    "strip_untranslated_vi_words",
    "disambiguate_vi_math",
    "canonicalize_en_math",
    "LATEX_PATTERN",
]
