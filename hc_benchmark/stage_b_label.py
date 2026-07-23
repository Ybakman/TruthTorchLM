"""Stage B -- correctness labels (protocol §3, §7).

Annotate each primary answer with a binary reliable/unreliable label -- the ground truth
the UQ score is graded against. The protocol is emphatic that this choice is not
incidental: "the correctness criterion materially moves AUROC" because it *defines the
labels*. So Stage B runs **both** criteria and persists both, and Stage D reports AUROC
sensitivity across them rather than either silently picking one.

* ``llm_judge`` -- TruthTorchLM's ``ModelJudge`` (LLM-as-judge). For health, hand it a
  clinical rubric via ``judge_system_prompt`` and spot-validate against a human.
* ``bleurt`` -- BLEURT-20 similarity thresholded at 0.5, matching DisAAD's ``--thres_gt
  0.5`` so our biomedical numbers line up with theirs.

Labels are keyed by criterion, e.g. ``correct_llm_judge`` / ``correct_bleurt``, and
written back next to the cache.
"""

import json
from pathlib import Path

from tqdm import tqdm

__all__ = ["label_stage_b", "load_labels"]


def _labels_path(cache) -> Path:
    return cache.path.with_suffix(".labels.json")


def label_stage_b(
    cache,
    criteria=("llm_judge", "bleurt"),
    judge_model: str = "gpt-4o-mini",
    bleurt_threshold: float = 0.5,
    judge_system_prompt: str = None,
    _judge_fn=None,
    _bleurt_fn=None,
    overwrite: bool = False,
):
    """Compute and persist correctness labels for a Stage-A cache.

    ``_judge_fn`` / ``_bleurt_fn`` are injection points so the orchestration is testable
    without loading a judge model or BLEURT. Each takes (question, answer, ground_truths)
    and returns a 0/1 int.
    """
    path = _labels_path(cache)
    if path.exists() and not overwrite:
        print(f"[Stage B] labels exist: {path.name} -- skipping.")
        return load_labels(cache)

    items = cache.read()
    judge_fn = _judge_fn or (_default_judge(judge_model, judge_system_prompt)
                             if "llm_judge" in criteria else None)
    bleurt_fn = _bleurt_fn or (_default_bleurt(bleurt_threshold)
                               if "bleurt" in criteria else None)

    labels = {}
    for item in tqdm(items, desc="[Stage B] labelling"):
        q, ans, gts = item["question"], item["primary_answer"], item["ground_truths"]
        entry = {}
        if "llm_judge" in criteria:
            entry["correct_llm_judge"] = int(judge_fn(q, ans, gts))
        if "bleurt" in criteria:
            entry["correct_bleurt"] = int(bleurt_fn(q, ans, gts))
        labels[str(item["item_id"])] = entry

    payload = {
        "criteria": list(criteria),
        "judge_model": judge_model,
        "bleurt_threshold": bleurt_threshold,
        "labels": labels,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[Stage B] wrote {len(labels)} labels -> {path.name}")
    return payload


def load_labels(cache) -> dict:
    path = _labels_path(cache)
    if not path.exists():
        raise FileNotFoundError(f"No Stage-B labels at {path}. Run label_stage_b first.")
    return json.loads(path.read_text())


def correctness_vector(cache, criterion: str) -> list:
    """The per-item 0/1 label list for one criterion, aligned to cache read order."""
    key = f"correct_{criterion}"
    labels = load_labels(cache)["labels"]
    items = cache.read()
    return [labels[str(item["item_id"])][key] for item in items]


def _default_judge(judge_model, judge_system_prompt):
    from TruthTorchLM.evaluators import ModelJudge

    judge = ModelJudge(model=judge_model) if judge_system_prompt is None else ModelJudge(
        model=judge_model, system_prompt=judge_system_prompt
    )

    def _fn(question, answer, ground_truths):
        return judge(question, answer, ground_truths, "")

    return _fn


def _default_bleurt(threshold):
    """BLEURT-20 scorer thresholded at `threshold`. Loaded lazily -- it pulls torch."""
    from evaluate import load as load_metric

    bleurt = load_metric("bleurt", "BLEURT-20", module_type="metric")

    def _fn(question, answer, ground_truths):
        if not ground_truths:
            return 0
        scores = bleurt.compute(
            predictions=[answer] * len(ground_truths),
            references=[str(g) for g in ground_truths],
        )["scores"]
        return int(max(scores) >= threshold)

    return _fn
