"""End-to-end Stage A->D driver.

    python -m hc_benchmark.run --config hc_benchmark/configs/smoke.yaml

Chains the four stages for one config, over its seed list, and writes a results bundle.
This is orchestration glue: the real work lives in the stage modules, and the point of
the driver is that the *whole pipeline* runs from a single YAML so a run is reproducible
from its config alone (whose hash keys every artifact).

Methods are constructed here rather than in config because a TruthMethod may need a loaded
entailment model or a proxy checkpoint -- an object, not a string. Edit ``build_methods``
for your method set; the default is the black-box pair used by the smoke test.
"""

import argparse
import json
from pathlib import Path

from .config import load_config
from .stage_a_generate import generate_stage_a
from .stage_b_label import label_stage_b, correctness_vector
from .stage_c_score import score_stage_c
from .stage_d_evaluate import aggregate_seeds, evaluate_method


def build_methods(entailment_device: str = "cuda"):
    """The UQ methods to score. Black-box only for round one (see the §1 filter).

    - VerbalizedConfidence: the cheap single-pass VB floor.
    - DiscreteSemanticEntropy: the text-only SC workhorse (Farquhar et al. Nature 2024) --
      the authentic replacement for grey-box SemanticEntropy in this regime.
    - NumSemanticSetUncertainty: cluster *count*, kept as a cheaper SC reference point
      (DSE is the cluster-assignment *entropy* over the same clustering).

    ``entailment_device`` is where the NLI clustering model loads; pass "cpu" for a
    CPU-only smoke run.
    """
    from TruthTorchLM.truth_methods import (
        DiscreteSemanticEntropy,
        EigV,
        LexicalSimilarity,
        NumSemanticSetUncertainty,
        VerbalizedConfidence,
    )
    from TruthTorchLM.utils.access_level import is_black_box

    methods = [
        VerbalizedConfidence(),                                     # VB single-pass floor
        LexicalSimilarity(number_of_generations=5),                # SC, similarity-only (no NLI)
        DiscreteSemanticEntropy(number_of_generations=5, entailment_model_device=entailment_device),
        EigV(number_of_generations=5, entailment_model_device=entailment_device),      # SC, graph-spectral
        NumSemanticSetUncertainty(number_of_generations=5, entailment_model_device=entailment_device),
    ]

    non_bb = [type(m).__name__ for m in methods if not is_black_box(m)]
    if non_bb:
        raise ValueError(
            f"Round one is pure black-box, but {non_bb} require more than generated text. "
            f"Remove them or run with the reference-line flag."
        )
    return methods


def run(config_path: str, cache_root="hc_benchmark/cache", results_root="hc_benchmark/results"):
    config = load_config(config_path)
    print(f"=== hc_benchmark run: {config.dataset}/{config.generator} "
          f"[{config.content_hash()}] ===")

    methods = build_methods()
    per_seed_by_criterion = {c: {type(m).__name__: [] for m in methods}
                             for c in config.correctness_criteria}

    for seed in config.seeds:
        cache = generate_stage_a(config, seed=seed, cache_root=cache_root)
        label_stage_b(cache, criteria=config.correctness_criteria,
                      bleurt_threshold=config.bleurt_threshold)
        scores = score_stage_c(cache, methods, model=config.generator,
                               n_sweep=config.n_sweep, collect_timing=True)

        for criterion in config.correctness_criteria:
            correctness = correctness_vector(cache, criterion)
            for method in methods:
                name = type(method).__name__
                ev = evaluate_method(scores[name], correctness, truth_method=method)
                per_seed_by_criterion[criterion][name].append(ev)

    # Aggregate seeds -> mean +/- std, per method per criterion (protocol §6).
    bundle = {"config": config.to_dict(), "results": {}}
    for criterion, by_method in per_seed_by_criterion.items():
        bundle["results"][criterion] = {
            name: aggregate_seeds(per_seed) for name, per_seed in by_method.items()
        }

    out_dir = Path(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{config.dataset}_{config.generator}_{config.content_hash()}.json"
    out_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"=== wrote {out_path} ===")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Run the Stage A-D UQ benchmark for one config.")
    parser.add_argument("--config", required=True, help="Path to a hc_benchmark YAML config.")
    parser.add_argument("--cache-root", default="hc_benchmark/cache")
    parser.add_argument("--results-root", default="hc_benchmark/results")
    args = parser.parse_args()
    run(args.config, cache_root=args.cache_root, results_root=args.results_root)


if __name__ == "__main__":
    main()
