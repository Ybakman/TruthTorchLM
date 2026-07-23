#!/usr/bin/env python
"""Live acceptance test: the Stage A->D pipeline on a real API target.

This is the end-to-end verification pytest deliberately does not do -- it needs the full ML
stack and a real API key. Run it on the cluster (or any box with GPU/CPU + an API key)
after `pip install -e .`. See `docs/SMOKE_RUN_RUNBOOK.md` for the full runbook.

It exercises, on two tiny dataset slices (one general free-form, one health MCQ):
  * Stage A  -- draw + cache the primary answer and N samples;
  * Stage B  -- correctness labels (LLM judge);
  * Stage C  -- score with the VB floor + the ported Discrete Semantic Entropy + NumSemSet;
  * Stage D  -- AUROC / PRR / ECE / MCE / Brier per dataset, with per-method aux-compute ms
                and an SLA verdict;
  * §5 concurrency -- a direct serial-vs-concurrent sampling timing, so the "concurrency
                rescues multi-sample" claim is actually measured, not just asserted.

Config via env:
  OPENAI_API_KEY   (required)         the target + judge key
  SMOKE_GENERATOR  (default gpt-4o-mini)
  SMOKE_DEVICE     (default: cuda if available else cpu)   where the NLI model loads
  SMOKE_DATASETS   (default "trivia_qa,medqa")
  SMOKE_SIZE       (default 0.004)    fraction of each dataset (keep tiny)
"""

import json
import os
import sys
import time

sys.path.insert(0, "src")


def _preflight():
    problems = []
    if not os.environ.get("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set (needed for the target + LLM judge).")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import litellm  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"ML stack not importable ({exc}). Run `pip install -e .`.")
    if problems:
        print("Smoke-run preflight failed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def _device():
    override = os.environ.get("SMOKE_DEVICE")
    if override:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_methods(device):
    from TruthTorchLM.truth_methods import (
        DiscreteSemanticEntropy,
        NumSemanticSetUncertainty,
        VerbalizedConfidence,
    )
    from TruthTorchLM.utils.access_level import is_black_box

    methods = [
        VerbalizedConfidence(),
        DiscreteSemanticEntropy(number_of_generations=5, entailment_model_device=device),
        NumSemanticSetUncertainty(number_of_generations=5, entailment_model_device=device),
    ]
    assert all(is_black_box(m) for m in methods), "smoke set must be pure black-box"
    return methods


def _run_stage_a_to_d(generator, datasets, size, device):
    from hc_benchmark.config import BenchmarkConfig, DecodingConfig
    from hc_benchmark.stage_a_generate import generate_stage_a
    from hc_benchmark.stage_b_label import label_stage_b, correctness_vector
    from hc_benchmark.stage_c_score import score_stage_c
    from hc_benchmark.stage_d_evaluate import evaluate_method
    from TruthTorchLM.instrumentation import sla_verdict, summarize

    methods = _build_methods(device)
    results = {}

    for dataset in datasets:
        print(f"\n{'='*64}\nDATASET: {dataset}\n{'='*64}")
        config = BenchmarkConfig(
            dataset=dataset, generator=generator, n_max=5, n_sweep=(1, 5),
            seeds=(0,), size_of_data=size, decoding=DecodingConfig(max_tokens=64),
        )

        cache = generate_stage_a(config, seed=0)
        assert cache.exists(), "Stage A cache missing"
        print(f"[A] cached {len(cache)} items -> {cache.path.name}")

        label_stage_b(cache, criteria=("llm_judge",), judge_model=generator)
        correctness = correctness_vector(cache, "llm_judge")
        assert len(correctness) == len(cache), "labels not aligned to cache"
        print(f"[B] labelled {len(correctness)} items (accuracy "
              f"{sum(c == 1 for c in correctness)}/{len(correctness)})")

        scores = score_stage_c(cache, methods, model=generator,
                               n_sweep=config.n_sweep, collect_timing=True)
        dataset_rows = {}
        for method in methods:
            name = type(method).__name__
            ev = evaluate_method(scores[name], correctness, truth_method=method,
                                 require_calibrated=False)
            for n, metrics in ev.items():
                summ = summarize(scores[name][n]["timing_ms"], warmup=0, label=f"{name} N={n}")
                verdict = sla_verdict(summ)
                dataset_rows[f"{name}_N{n}"] = {
                    "auroc": metrics.get("auroc"), "prr": metrics.get("prr"),
                    "ece": metrics.get("ece"), "mce": metrics.get("mce"),
                    "brier": metrics.get("brier"),
                    "aux_p50_ms": summ["p50_ms"], "aux_p95_ms": summ["p95_ms"],
                    "fits_500ms": verdict["verdicts"]["500ms"],
                }
                print(f"[D] {name:26s} N={n}: "
                      f"AUROC={metrics.get('auroc'):.3f} PRR={metrics.get('prr'):.3f} "
                      f"ECE={metrics.get('ece'):.3f} MCE={metrics.get('mce'):.3f} "
                      f"Brier={metrics.get('brier'):.3f} | "
                      f"aux p50={summ['p50_ms']:.1f}ms fits500ms={verdict['verdicts']['500ms']}")
        results[dataset] = dataset_rows
    return results


def _concurrency_check(generator):
    """Directly measure §5's serial-vs-concurrent sampling on one prompt."""
    from TruthTorchLM.generation import sample_generations_api
    from TruthTorchLM.instrumentation.concurrency import sample_generations_api_concurrent

    messages = [{"role": "user", "content": "Name a European capital city."}]
    n = 5
    t0 = time.perf_counter()
    sample_generations_api(generator, messages, generation_seed=0, number_of_generations=n,
                           return_text=True, max_tokens=16)
    serial_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    sample_generations_api_concurrent(generator, messages, generation_seed=0,
                                      number_of_generations=n, return_text=True, max_tokens=16)
    concurrent_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n{'='*64}\n§5 CONCURRENCY CHECK (N={n} draws)\n{'='*64}")
    print(f"  serial     : {serial_ms:7.0f} ms")
    print(f"  concurrent : {concurrent_ms:7.0f} ms")
    rescued = concurrent_ms < serial_ms
    print(f"  -> concurrency {'RESCUES multi-sample (concurrent < serial)' if rescued else 'did NOT beat serial (check API concurrency caps)'}")
    return {"serial_ms": serial_ms, "concurrent_ms": concurrent_ms, "rescued": rescued}


def main():
    _preflight()
    generator = os.environ.get("SMOKE_GENERATOR", "gpt-4o-mini")
    device = _device()
    datasets = os.environ.get("SMOKE_DATASETS", "trivia_qa,medqa").split(",")
    size = float(os.environ.get("SMOKE_SIZE", "0.004"))
    print(f"Smoke run: generator={generator} device={device} datasets={datasets} size={size}")

    results = {"generator": generator, "device": device, "datasets": {}}
    results["datasets"] = _run_stage_a_to_d(generator, datasets, size, device)
    try:
        results["concurrency"] = _concurrency_check(generator)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[concurrency check skipped: {exc}]")

    os.makedirs("hc_benchmark/results", exist_ok=True)
    out = "hc_benchmark/results/smoke_run.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSmoke run complete. Results -> {out}")
    print("PASS if: every [D] row has numbers, both label sets exist, and the concurrency "
          "check ran. AUROC~0.5 on a handful of items is expected -- this proves the "
          "machinery, not a research claim.")


if __name__ == "__main__":
    main()
