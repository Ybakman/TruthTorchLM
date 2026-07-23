"""IUQ — Interrogative Uncertainty Quantification for long-form LLM generation.

Official source: **Fan, Duan & Xu, "IUQ: Interrogative Uncertainty Quantification for
Long-Form Large Language Model Generation", ACL 2026
([arXiv:2604.15109](https://arxiv.org/abs/2604.15109))**, `louisfanhz/IUQ` (pinned at
`third_party/IUQ`, MIT). This is a faithful port of that repo's claim-level scoring — the
prompts are vendored verbatim and the score formulas match `main.py` / `schemas.py`.

**Why this doesn't fit the scalar `TruthMethod` (or the per-claim `ClaimCheckMethod`)**:
IUQ is a *generation-level, long-form* method. Two of its three quantities cannot be
computed from one isolated claim:

* **supportness** — cross-sample consistency: the fraction of the N *other* diverse
  generations that support the claim (needs the whole sample set), and
* **impact** — contradiction of the claim's interrogation answers with the *accumulated
  claim context*, with exponential-decay **error propagation across the ordered claim
  sequence** (needs all prior claims).

TTLM's `ClaimCheckMethod.check_claim` sees one claim + `text_so_far` at a time, so IUQ
lives here as its own pipeline rather than being forced into that interface. The final
per-claim score is ``IUQ = supportness × impact``; faithfulness (``1 − mean contradiction``)
is reported alongside.

The pipeline stages (claim extraction → question generation → responding → contradiction
scoring → supportness voting) are LLM calls; they are wired here and exercised in a live
long-form run. The **pure scoring math** below is what had to be ported exactly, and is
unit-tested against hand-computed values.
"""

import re
from typing import Callable, List

import numpy as np

__all__ = [
    "INTERROGATOR_PROMPTS",
    "RESPONDER_PROMPTS",
    "EVALUATOR_PROMPTS",
    "compute_impacts",
    "compute_faithfulness",
    "iuq_score",
    "parse_contradiction_percentage",
    "parse_support_vote",
    "InterrogativeUQ",
]

# --- Prompts, vendored verbatim from third_party/IUQ/prompts.py --------------------------

INTERROGATOR_PROMPTS = {
    "q_from_single_claim_system_prompt": "You are a helpful assistant.",
    "q_from_single_claim_user_prompt": (
        "Given context and a claim, generate one specific, clear question that has its "
        "answer contained in the claim. The generated question must be self-contained and "
        "related to the context.\nReturn only the question, with no additional text.\n\n"
        "Context: {context}\n\nClaim: {claim}"
    ),
}

RESPONDER_PROMPTS = {
    "respond": (
        "Answer the following question based on the given context. Format your answer in "
        "one sentence:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer: "
    ),
    "contradiction": (
        "You will be given a statement and a context. Suppose the statement is TRUE, how "
        "much of the context will you change to keep it consistent with the statement?\n"
        "Your final answer should be a percentage number between 0 and 100, representing "
        "the percentage of the context you will change.\n\n<Statement>\n{statement}\n"
        "</Statement>\n\n<Context>\n{context}\n</Context>\n\nReturn your answer as a "
        "percentage number ONLY, with no additional text."
    ),
}

EVALUATOR_PROMPTS = {
    "from_generations_system_prompt": (
        "You are a meticulous fact-checker who checks the correctness of claims based on a "
        "given passage."
    ),
    "from_generations_user_prompt_strict": (
        'Is the following claim supported by the given passage?\n\n<Claim>\n{claim}\n'
        '</Claim>\n\n<Passage>\n{passage}\n</Passage>\n\nReturn "True" if the claim is '
        'supported by the passage, return "False" otherwise. Return ONLY the result, '
        "nothing else."
    ),
}


# --- Pure scoring, faithful ports of schemas.py / main.py --------------------------------


def compute_impacts(contradictions: List[float], with_error_propagation: bool = True) -> np.ndarray:
    """Per-claim impact from the claims' mean contradiction scores.

    Port of ``GenerationSample.gather_impacts``. ``contradictions[i]`` is the mean
    contradiction over claim *i*'s interrogation answers (in claim order). Without error
    propagation, impact is simply ``1 − contradiction``. With it, contradictions leak
    forward through the ordered claim sequence via an exponential-decay convolution, then
    are mapped through ``exp(−·)`` so a claim downstream of a contradicted one is penalised:

        weights = convolve(contradictions, exp(-arange(n)))[:n]
        impact  = exp(-weights)
    """
    impacts = np.asarray(contradictions, dtype=float)
    if impacts.size == 0:
        return impacts
    if not with_error_propagation:
        return 1 - impacts
    weight_func = np.exp(-np.arange(len(impacts)))
    weights = np.convolve(impacts, weight_func)[: len(impacts)]
    return 1 / np.exp(weights)


def compute_faithfulness(contradictions: List[float]) -> np.ndarray:
    """Per-claim faithfulness = ``1 − mean contradiction`` (port of
    ``gather_claim_level_faithfulness``)."""
    return 1 - np.asarray(contradictions, dtype=float)


def iuq_score(supportness: float, impact: float):
    """The IUQ claim score: ``supportness × impact`` (port of ``compute_uq_scores``).

    ``None`` supportness (unscored claim) propagates to ``None``, matching the source.
    Higher = more supported and more faithful = more certain; as a TTLM truth value it is
    used directly (its complement is the uncertainty).
    """
    if supportness is None:
        return None
    return float(supportness) * float(impact)


def parse_contradiction_percentage(response: str) -> float:
    """Parse the contradiction judge's "0-100" reply into a [0,1] score.

    Port of the parsing in ``phase_evaluate_faithfulness``: take the last sentence, find
    the first integer, divide by 100 (out-of-range or unparseable -> 0.0).
    """
    response = (response or "").strip()
    sentences = [s for s in re.split(r"[.!?]+\s*", response) if s]
    if sentences:
        response = sentences[-1]
    match = re.search(r"(\d+)[.,%]?", response)
    if not match:
        return 0.0
    pct = int(match.group(1))
    return pct / 100.0 if 0 <= pct <= 100 else 0.0


def parse_support_vote(response: str) -> bool:
    """Parse the supportness judge's reply into a boolean vote.

    Port of ``phase_evaluate_supportness``: a reply containing "true"/"yes" is a support
    vote.
    """
    r = (response or "").lower().strip()
    return ("true" in r) or ("yes" in r)


class InterrogativeUQ:
    """Long-form, claim-level IUQ (Fan et al., ACL 2026). Pure black-box (text only).

    Scores a set of ``claims`` extracted from a generation, given the N diverse
    generations of the same prompt. ``chat`` is a callable ``(messages, temperature) ->
    text`` (e.g. wrapping ``litellm.completion`` or a local model); injected so this stays
    testable and backend-agnostic. Claim extraction and question generation can be done
    upstream (TTLM's decomposition) or via ``chat``.
    """

    def __init__(self, chat: Callable, n_answers: int = 3, with_error_propagation: bool = True):
        self.chat = chat
        self.n_answers = n_answers
        self.with_error_propagation = with_error_propagation

    # -- per-claim LLM stages ------------------------------------------------------------

    def supportness(self, claim: str, diverse_generations: List[str]) -> float:
        """Fraction of the N diverse generations that support the claim."""
        votes = []
        for passage in diverse_generations:
            prompt = EVALUATOR_PROMPTS["from_generations_user_prompt_strict"].format(
                claim=claim, passage=passage
            )
            reply = self.chat(
                [
                    {"role": "system",
                     "content": EVALUATOR_PROMPTS["from_generations_system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                0.0,
            )
            votes.append(parse_support_vote(reply))
        return float(np.mean(votes)) if votes else 0.0

    def contradiction(self, statement: str, context: str) -> float:
        """Contradiction of a statement with the accumulated claim context, in [0,1]."""
        prompt = RESPONDER_PROMPTS["contradiction"].format(statement=statement, context=context)
        reply = self.chat([{"role": "user", "content": prompt}], 0.0)
        return parse_contradiction_percentage(reply)

    def question_for_claim(self, claim: str, context: str) -> str:
        prompt = INTERROGATOR_PROMPTS["q_from_single_claim_user_prompt"].format(
            context=context, claim=claim
        )
        return self.chat(
            [
                {"role": "system",
                 "content": INTERROGATOR_PROMPTS["q_from_single_claim_system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            0.0,
        ).strip()

    def answer(self, question: str, context: str, temperature: float = 1.0) -> str:
        prompt = RESPONDER_PROMPTS["respond"].format(context=context, question=question)
        return self.chat([{"role": "user", "content": prompt}], temperature).strip()

    # -- the generation-level scoring ----------------------------------------------------

    def score_generation(self, claims: List[str], diverse_generations: List[str],
                         context: str = "") -> List[dict]:
        """Full IUQ for every claim of one generation.

        For each claim: interrogate (one question), answer it ``n_answers`` times, score
        each answer's contradiction with the accumulated claim context, then combine into
        supportness × impact. The accumulated context is the claims up to and including the
        current one, reversed — matching ``phase_evaluate_faithfulness``'s
        ``"\\n".join(all_claims[claim_idx::-1])``.
        """
        per_claim_contradiction = []
        supportness_scores = []
        for claim_idx, claim in enumerate(claims):
            claim_context = "\n".join(claims[claim_idx::-1])
            question = self.question_for_claim(claim, context)
            answer_contradictions = [
                self.contradiction(self.answer(question, context), claim_context)
                for _ in range(self.n_answers)
            ]
            per_claim_contradiction.append(float(np.mean(answer_contradictions)))
            supportness_scores.append(self.supportness(claim, diverse_generations))

        impacts = compute_impacts(per_claim_contradiction, self.with_error_propagation)
        faithfulness = compute_faithfulness(per_claim_contradiction)

        results = []
        for i, claim in enumerate(claims):
            results.append(
                {
                    "claim": claim,
                    "supportness_score": supportness_scores[i],
                    "impact": float(impacts[i]),
                    "faithfulness_score": float(faithfulness[i]),
                    # higher IUQ = more supported + faithful = more certain
                    "truth_value": iuq_score(supportness_scores[i], float(impacts[i])),
                    "interrogative_uncertainty": iuq_score(supportness_scores[i], float(impacts[i])),
                }
            )
        return results
