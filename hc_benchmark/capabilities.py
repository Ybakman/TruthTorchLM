"""Capabilities inventory — everything the benchmark supports, in main + sub categories.

One command that answers "what's in here?" across all four axes of the protocol
(**G × D → M → V**): generators, datasets, UQ methods, and evaluation metrics — each
grouped into main categories and sub-categories.

It is **derived from the code**, not hand-maintained, so it can't drift: generators come
from the `generators` registry, datasets and metrics from `availability`, and the method
catalog is validated against the real exports by a torch-gated test. Dependency-light —
runs without the ML stack.

    from hc_benchmark.capabilities import print_capabilities
    print_capabilities()                 # everything
    print_capabilities(axis="D")         # just datasets

or from the shell:

    python -m hc_benchmark.capabilities            # all axes
    python -m hc_benchmark.capabilities --axis M   # methods only
"""

import importlib.util as _ilu
from pathlib import Path as _Path

from .generators import Access, GENERATORS, Role

__all__ = ["capabilities", "print_capabilities", "METHOD_CATALOG"]


def _load_availability():
    """Load TruthTorchLM/availability.py directly (it's pure lists) so this stays torch-free.

    Importing it through the package would run TruthTorchLM/__init__, which pulls in torch.
    """
    path = _Path(__file__).resolve().parents[1] / "src/TruthTorchLM/availability.py"
    spec = _ilu.spec_from_file_location("_ttlm_availability", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_av = _load_availability()


# --- M: method catalog ------------------------------------------------------------------
# The method classes import torch, so they can't be introspected here without the stack.
# This catalog is the declared inventory; tests/test_capabilities.py (torch-gated) asserts
# it matches the real truth_methods exports, access levels, and REQUIRES_TRAINING flags,
# so it stays honest.
#
# fields: (name, family, source, prep)   prep in {inference, needs-training, needs-fit}
METHOD_CATALOG = [
    # Verbalized (VB) -- stated confidence
    ("VerbalizedConfidence", "VB · self-verbalized", "upstream TTLM", "inference"),
    ("PTrue", "VB · self-surrogate (prob-read)", "upstream TTLM", "inference"),
    ("CrossExamination", "VB · external judge", "upstream TTLM", "inference"),
    ("MultiLLMCollab", "VB · multi-LLM", "upstream TTLM", "inference"),
    # Self-Consistency (SC) -- O(N) sampling
    ("DiscreteSemanticEntropy", "SC · cluster entropy", "jlko/semantic_uncertainty (Nature 2024)", "inference"),
    ("NumSemanticSetUncertainty", "SC · cluster count", "upstream TTLM", "inference"),
    ("KernelLanguageEntropy", "SC · cluster entropy", "upstream TTLM", "inference"),
    ("LexicalSimilarity", "SC · similarity-only", "zlin7/UQ-NLG (TMLR)", "inference"),
    ("EigV", "SC · graph-spectral", "zlin7/UQ-NLG (TMLR) [alias of SumEigen]", "inference"),
    ("SelfDetection", "SC · self-check", "upstream TTLM", "inference"),
    ("SPUQ", "SC · perturbation", "intuit-ai-research/SPUQ (EACL 2024)", "inference"),
    ("IUQ", "SC · long-form claim-level", "louisfanhz/IUQ (ACL 2026)", "inference"),
    ("NeighborhoodConsistencyBelief", "SC · neighborhood", "zjunlp/belief (ACL 2026)", "inference"),
    # Proxy / Surrogate (P)
    ("DisAAD", "P · distilled proxy", "huizi-Cui/DisAAD (ACL 2026)", "needs-training"),
    # Pre-generation input gate (not a TruthMethod)
    ("OOD-PCA gate", "Input gate · density/OOD", "toastedqu/rag_safety_pca (ACL 2026)", "needs-fit"),
]

# --- D: dataset categorization ----------------------------------------------------------
# Format buckets, built from the availability lists so they track the loaders.
_MCQ = set(_av.MCQ_DATASETS)
_LONGFORM = set(_av.LONG_FORM_HEALTH_DATASETS) | set(_av.LONG_FORM_AVAILABLE_DATASETS)
_EXTRACTIVE = {"squad_v2", "narrative_qa", "wikipedia_factual"}
_MULTIHOP = {"hotpot_qa", "gsm8k"}


def _dataset_format(name: str) -> str:
    if name in _MCQ:
        return "MCQ (constrained output)"
    if name in _LONGFORM:
        return "Long-form (claim-level)"
    if name in _EXTRACTIVE:
        return "Extractive / contextual"
    if name in _MULTIHOP:
        return "Multi-hop / reasoning"
    return "Free-form QA"


def _dataset_area(name: str) -> str:
    if name in set(_av.HEALTH_DATASETS):
        return "Health"
    if name == "gsm8k":
        return "Math"
    return "General"


def _all_datasets():
    # single-turn QA datasets + the long-form ones
    return sorted(set(_av.AVAILABLE_DATASETS) | _LONGFORM)


# --- assembly ---------------------------------------------------------------------------


def _group(items, key):
    out = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


def capabilities() -> dict:
    """The full nested inventory: {axis: {main_category: {sub_category: [items]}}}."""
    # G -- generators, from the registry
    specs = list(GENERATORS.values())
    g_by_provider = {}
    for prov in ("openai", "anthropic", "gemini", "open"):
        members = [s for s in specs if s.provider == prov]
        if members:
            g_by_provider[prov] = {
                "targets": [s.name for s in members if Role.TARGET in s.roles],
                "proxies": [s.name for s in members if Role.PROXY in s.roles],
            }
    G = {
        "By provider": g_by_provider,
        "By access": {
            "black-box (API targets)": [s.name for s in specs if s.access is Access.BLACK_BOX],
            "white-box (open; reference line + proxy)":
                [s.name for s in specs if s.access is Access.WHITE_BOX],
        },
        "Reasoning (LRM) targets": [s.name for s in specs if s.is_reasoning],
    }

    # D -- datasets
    datasets = _all_datasets()
    D = {
        "By area": _group(datasets, _dataset_area),
        "By format": _group(datasets, _dataset_format),
        "By scope": {
            "single-turn QA": datasets,
            "dialogue / multi-turn": ["(none yet — Q4 stub)"],
        },
    }

    # M -- methods
    M = {
        "By family": _group(METHOD_CATALOG, lambda m: m[1].split(" · ")[0]),
        "By preparation": _group(METHOD_CATALOG, lambda m: m[3]),
    }

    # V -- evaluation metrics, from availability
    V = {
        "Discrimination": _av.DISCRIMINATION_METRICS,
        "Threshold-based": _av.THRESHOLD_METRICS,
        "Calibration error": _av.CALIBRATION_METRICS,
        "Safety-weighted": _av.SAFETY_METRICS,
        "Latency (protocol §5)": ["marginal_ms (p50/p95/p99)", "overhead_ratio",
                                  "SLA pass/fail", "serial vs concurrent"],
    }

    return {"G · Generators": G, "D · Datasets": D, "M · UQ methods": M,
            "V · Metrics": V}


def _print_node(node, indent=1):
    pad = "  " * indent
    if isinstance(node, dict):
        for k, v in node.items():
            print(f"{pad}{k}:")
            _print_node(v, indent + 1)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, tuple):  # a method catalog row
                name, family, source, prep = item
                tag = "" if prep == "inference" else f"  [{prep}]"
                print(f"{pad}- {name}{tag}   ({source})")
            else:
                print(f"{pad}- {item}")
    else:
        print(f"{pad}- {node}")


def print_capabilities(axis: str = None) -> None:
    """Print the inventory. ``axis`` in {'G','D','M','V'} limits to one axis."""
    caps = capabilities()
    n_ds = len(_all_datasets())
    n_gen = len(GENERATORS)
    n_meth = len(METHOD_CATALOG)
    print(f"\nTruthTorchLM-HC capabilities — {n_gen} generators · {n_ds} datasets · "
          f"{n_meth} methods\n" + "=" * 70)
    for axis_name, node in caps.items():
        if axis and not axis_name.upper().startswith(axis.upper()):
            continue
        print(f"\n{axis_name}")
        _print_node(node)
    print()


def _main():
    import argparse

    p = argparse.ArgumentParser(description="List everything the benchmark supports.")
    p.add_argument("--axis", choices=["G", "D", "M", "V"], default=None,
                   help="Limit to one axis: G generators, D datasets, M methods, V metrics.")
    args = p.parse_args()
    print_capabilities(axis=args.axis)


if __name__ == "__main__":
    _main()
