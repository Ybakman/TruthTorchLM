# Tests

```bash
pip install pytest
python -m pytest                 # fast suite: pure-numpy layers, offline
python -m pytest -m network      # + live HuggingFace schema checks (needs internet)
```

## Two tiers, on purpose

The **default suite runs without the ML stack.** The calibration metrics, safety metrics,
the black-box filter, the latency instrumentation, and the harness plumbing depend only on
numpy / pandas / pyyaml, so `tests/conftest.py` loads those modules standalone when
`import torch` is unavailable. This keeps the layers that encode the benchmark's *logic*
— the ones where a subtle error returns a plausible wrong number rather than crashing —
fast to check and green on any machine.

Everything here is pinned against **hand-computed or closed-form answers**, not "it ran":
a perfectly-calibrated set has ECE≈0, an all-confident-all-wrong set has ECE≈MCE≈Brier≈1,
blocking everything scores perfect harm recall at 100% over-refusal, and so on.

Tests that need the real thing are marked and excluded by default:

- `-m network` — hits the Hub to confirm upstream dataset schemas haven't drifted
  (`test_hc_datasets.py::TestLiveSchemas`).
- `test_black_box_shortlist.py` — `importorskip("torch")`; asserts the §1 filter's most
  consequential finding on the *real* classes, that upstream `SemanticEntropy` is grey-box.

## The live smoke run (the real acceptance test)

The end-to-end Stage A→D run needs the full stack **and** an API key, so it lives outside
pytest as `scripts/smoke_run.py`. On the cluster:

```bash
git submodule update --init --recursive --depth 1
pip install -r requirements.txt && pip install -e .
export OPENAI_API_KEY=...
python scripts/smoke_run.py
```

It must produce a populated Stage-A Parquet cache, both label sets, per-dataset
AUROC/PRR/ECE/MCE/Brier, marginal-ms p50/p95/p99 per method, an SLA pass/fail column, and
the frontier plot — with concurrent N=5 marginal ms well below serial N=5.
