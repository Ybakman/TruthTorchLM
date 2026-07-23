"""Neighbor-Consistency Belief (NCB) — structural belief robustness over a neighborhood.

Official source: **Xu, Zhao, Yao, et al., "Illusions of Confidence? Diagnosing LLM
Truthfulness via Neighborhood Consistency", ACL 2026
([arXiv:2601.05905](https://arxiv.org/abs/2601.05905))**, `zjunlp/belief` (pinned at
`third_party/belief-neighborhood-consistency`). This is a faithful port of that repo's
`calc_belief_score.py::calculate_belief_metrics`.

**The measurement is inference-only.** The repo also ships Structure-Aware Training (SAT),
but that is an *intervention* to improve robustness — not the UQ signal. NCB itself needs
no training; it is a scoring function over samples, so it belongs with the round-two
inference methods, not the DisAAD-style train-on-server bucket.

**The idea (and why it is not plain self-consistency).** Point-wise self-consistency can
create an "illusion of knowing": a model can answer the original question with perfect
consistency yet collapse on conceptually-adjacent questions. NCB multiplies the original
question's self-consistency ``P(y)`` by the model's accuracy across a *neighborhood* of
related questions ``p_i``:

    NCB = P(y) · aggregate(p_1, ..., p_k)

with three aggregations from the source: geometric mean (default), arithmetic mean, and a
robust geometric mean (drop the single worst neighbor). A belief that is self-consistent
but fragile across its neighborhood scores low. NCB is also **gated on the original answer
being correct** — belief robustness is only defined when the model actually knows the
fact; a wrong dominant answer is ``valid=False`` with score 0.

**Requirement worth stating.** NCB needs a *neighborhood* — related questions with known
answers — which is a dataset artifact (the paper's fact-belief data supplies them). It is
therefore applicable to belief-structured datasets, not plain short-QA, and does not read
the shared Stage-A sample cache. Pure black-box (text + entity extraction only).
"""

from collections import Counter
from typing import List, Optional

import numpy as np

__all__ = [
    "EPSILON",
    "dominant_answer_probability",
    "neighbor_accuracy",
    "aggregate_neighbor_probs",
    "neighbor_consistency_belief",
    "is_dominant_correct",
]

# Matches the source's clip constant, used to keep log() finite.
EPSILON = 1e-10


def dominant_answer_probability(entities: List[str]):
    """P(y): majority-vote fraction over the original question's sampled entity-answers.

    Port of Step 1: cluster the N samples by (normalized) entity, take the fraction in the
    largest cluster. Returns ``(dominant_answer, p_y)``; ``p_y`` is None for no samples.
    """
    if not entities:
        return None, None
    counts = Counter(entities)
    dominant_answer, dominant_count = counts.most_common(1)[0]
    return dominant_answer, dominant_count / len(entities)


def is_dominant_correct(dominant_answer: str, golden_answer: str, match_type: str = "loose") -> bool:
    """The validity gate: is the model's dominant answer the correct one?

    'loose' = substring containment either way (the source's default in run_all.sh);
    'strict' = exact match.
    """
    if match_type == "loose":
        if not dominant_answer or not golden_answer:
            return dominant_answer == golden_answer
        return (dominant_answer in golden_answer) or (golden_answer in dominant_answer)
    return dominant_answer == golden_answer


def neighbor_accuracy(responses: List[str], correct_answer: str) -> float:
    """p_i: fraction of a neighbor question's N samples that match its correct answer.

    Port of Step 2: exact, case-insensitive, stripped match. No samples -> 0.0.
    """
    if not responses:
        return 0.0
    correct = (correct_answer or "").strip().lower()
    correct_count = sum(1 for r in responses if (r or "").strip().lower() == correct)
    return correct_count / len(responses)


def aggregate_neighbor_probs(neighbor_probs: List[float], p_y: float,
                             neighbor_agg: str = "geo_mean") -> float:
    """Combine P(y) with the neighbor accuracies into the NCB score (Step 3).

    Exact port of the three log-space aggregations, with the same EPSILON clipping:
      geo_mean         -> P(y) · GeometricMean(p_i)
      arith_mean       -> P(y) · ArithmeticMean(p_i)
      robust_geo_mean  -> drop the single lowest p_i, then geometric mean
    """
    p_y_safe = max(p_y, EPSILON)
    if not neighbor_probs:
        return 0.0

    probs = list(neighbor_probs)
    probs_safe = np.clip(probs, EPSILON, 1.0)

    if neighbor_agg == "arith_mean":
        mean_neighbors = float(np.mean(probs))  # original probs; p_i >= 0
        log_score = np.log(p_y_safe) + np.log(max(mean_neighbors, EPSILON))
    elif neighbor_agg == "robust_geo_mean":
        if len(probs) > 1:
            min_idx = int(np.argmin(probs))
            filtered = [p for i, p in enumerate(probs) if i != min_idx]
        else:
            filtered = probs
        filtered_safe = np.clip(filtered, EPSILON, 1.0)
        log_score = np.log(p_y_safe) + float(np.mean(np.log(filtered_safe)))
    elif neighbor_agg == "geo_mean":
        log_score = np.log(p_y_safe) + float(np.mean(np.log(probs_safe)))
    else:
        raise ValueError(f"Unknown neighbor_agg '{neighbor_agg}'.")

    return float(np.exp(log_score))


def neighbor_consistency_belief(
    entities: List[str],
    golden_answer: str,
    neighbor_responses: List[List[str]],
    neighbor_correct_answers: List[str],
    match_type: str = "loose",
    neighbor_agg: str = "geo_mean",
) -> dict:
    """Full NCB for one item (faithful port of `calculate_belief_metrics`'s per-item body).

    ``entities`` are the original question's N sampled answers (entity-normalized upstream);
    ``neighbor_responses[i]`` / ``neighbor_correct_answers[i]`` are the i-th neighbor
    question's samples and its known answer.

    Returns the source's belief_result dict; ``truth_value`` mirrors ``score`` (higher =
    more robust belief = more truthful). ``valid=False`` (wrong dominant answer or no
    samples) yields score 0.
    """
    dominant, p_y = dominant_answer_probability(entities)
    if p_y is None:
        return {"valid": False, "reason": "no_samples", "score": 0.0, "truth_value": 0.0}

    if not is_dominant_correct(dominant, golden_answer, match_type):
        return {
            "valid": False, "reason": "wrong_answer",
            "dominant_answer": dominant, "golden_answer": golden_answer,
            "p_y": float(p_y), "score": 0.0, "truth_value": 0.0,
        }

    neighbor_probs = [
        neighbor_accuracy(resp, ans)
        for resp, ans in zip(neighbor_responses, neighbor_correct_answers)
    ]
    score = aggregate_neighbor_probs(neighbor_probs, p_y, neighbor_agg)
    return {
        "valid": True,
        "score": score,
        "truth_value": score,
        "p_y": float(p_y),
        "neighbor_probs": [float(p) for p in neighbor_probs],
    }
