"""Shared pytest configuration.

Puts ``src/`` on the path so the tests run against a working tree without an editable
install.

It also allows the pure-numpy layers -- the calibration, safety, and access-level
modules -- to be tested **without the full ML stack installed**. Importing the
``TruthTorchLM`` package normally pulls in torch, transformers, litellm, and outlines,
which is a heavy prerequisite for testing functions whose only dependency is numpy. When
those imports fail, we register a minimal stand-in package so the leaf modules can still
be loaded by their real dotted names. When the full stack *is* present (the GPU cluster,
CI), nothing is stubbed and the tests exercise the genuine import path.
"""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Leaf modules that depend on numpy alone and are therefore testable standalone.
_STANDALONE_MODULES = {
    "TruthTorchLM.utils.calibration_metrics": SRC / "TruthTorchLM/utils/calibration_metrics.py",
    "TruthTorchLM.utils.safety_metrics": SRC / "TruthTorchLM/utils/safety_metrics.py",
    "TruthTorchLM.utils.access_level": SRC / "TruthTorchLM/utils/access_level.py",
    "TruthTorchLM.instrumentation.timing": SRC / "TruthTorchLM/instrumentation/timing.py",
    "TruthTorchLM.instrumentation.stats": SRC / "TruthTorchLM/instrumentation/stats.py",
    "TruthTorchLM.evaluators.correctness_evaluator": SRC
    / "TruthTorchLM/evaluators/correctness_evaluator.py",
    "TruthTorchLM.evaluators.mcq_match": SRC / "TruthTorchLM/evaluators/mcq_match.py",
    "TruthTorchLM.utils.hc_datasets": SRC / "TruthTorchLM/utils/hc_datasets.py",
}


def _install_stub_package() -> None:
    """Register TruthTorchLM / TruthTorchLM.utils as namespace shells, then load leaves."""
    for name, path in (
        ("TruthTorchLM", SRC / "TruthTorchLM"),
        ("TruthTorchLM.utils", SRC / "TruthTorchLM/utils"),
        ("TruthTorchLM.instrumentation", SRC / "TruthTorchLM/instrumentation"),
        ("TruthTorchLM.evaluators", SRC / "TruthTorchLM/evaluators"),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg
    sys.modules["TruthTorchLM"].utils = sys.modules["TruthTorchLM.utils"]
    sys.modules["TruthTorchLM"].instrumentation = sys.modules["TruthTorchLM.instrumentation"]
    sys.modules["TruthTorchLM"].evaluators = sys.modules["TruthTorchLM.evaluators"]

    for dotted, path in _STANDALONE_MODULES.items():
        if dotted in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(dotted, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = module
        spec.loader.exec_module(module)
        parent, leaf = dotted.rsplit(".", 1)
        setattr(sys.modules[parent], leaf, module)


try:  # pragma: no cover - depends on the local environment
    import TruthTorchLM  # noqa: F401
except Exception as exc:  # noqa: BLE001 - any missing heavy dependency
    print(
        f"[conftest] Full TruthTorchLM import unavailable ({type(exc).__name__}: {exc}); "
        "falling back to standalone loading of the pure-numpy metric modules.",
        file=sys.stderr,
    )
    _install_stub_package()
