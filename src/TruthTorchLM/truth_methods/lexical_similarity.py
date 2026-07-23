"""Lexical Similarity (LeS) — mean pairwise ROUGE-L across samples.

Official source: **Lin, Trivedi & Sun, "Generating with Confidence: Uncertainty
Quantification for Black-box LLMs", TMLR ([arXiv:2305.19187](https://arxiv.org/abs/2305.19187))**,
implemented in `zlin7/UQ-NLG` (pinned at `third_party/UQ-NLG`). This is a faithful port of
that repo's ``_compute_lexical_sim`` over the ROUGE-L similarity matrix built by
``_get_lexical_similarities_sample``.

LeS is the *similarity-only* consistency baseline: no NLI model, no clustering, no token
likelihoods — just the average ROUGE-L overlap between every pair of sampled answers. If
the samples reword the same content they overlap highly (confident); if they diverge, the
mean similarity drops (uncertain). Text-only, so it passes the §1 black-box filter.

In the source, the returned *uncertainty* is ``-mean_pairwise_rougeL`` (higher similarity =
lower uncertainty). TTLM's truth value is oriented the other way (higher = more truthful),
so we return the mean similarity directly as the truth value — the same quantity, its
natural sign for this framework.

Port note: the source builds a similarity matrix over *unique* answers and averages via a
generation→unique-answer ``mapping``; that is a caching optimization. Averaging ROUGE-L
over the raw ordered generation pairs (identical answers scoring 1.0) is arithmetically
identical, and is what we do here for clarity.
"""

import itertools
from typing import Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .truth_method import TruthMethod
from ..generation import sample_generations_hf_local, sample_generations_api


def _mean_pairwise_rougeL(generated_texts: list, rouge_scorer) -> float:
    """Average ROUGE-L f-measure over all ordered pairs of generations (i != j).

    Matches ``_compute_lexical_sim``: identical answers score 1.0, a single generation
    yields 1.0 by convention (no pairs → maximally self-consistent).
    """
    n = len(generated_texts)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    # Cache pairwise scores; the matrix is symmetric, so compute each unordered pair once.
    for a, b in itertools.combinations(range(n), 2):
        score = rouge_scorer.compute(
            predictions=[generated_texts[a]],
            references=[generated_texts[b]],
            rouge_types=["rougeL"],
        )["rougeL"]
        total += 2 * float(score)  # both ordered directions (i,j) and (j,i)
        count += 2
    return total / count if count else 1.0


class LexicalSimilarity(TruthMethod):
    """Similarity-only consistency: mean pairwise ROUGE-L (Lin et al. 2023). Pure black-box."""

    REQUIRES_SAMPLED_TEXT = True

    def __init__(self, number_of_generations: int = 5, batch_generation: bool = True,
                 rouge_scorer=None):
        super().__init__()
        self.number_of_generations = number_of_generations
        self.batch_generation = batch_generation
        # Loaded lazily so importing the class doesn't pull in `evaluate`/`rouge_score`.
        self._rouge_scorer = rouge_scorer

    @property
    def rouge_scorer(self):
        if self._rouge_scorer is None:
            import evaluate

            # Same scorer the source uses: HF `evaluate` rouge -> rougeL f-measure.
            self._rouge_scorer = evaluate.load("rouge")
        return self._rouge_scorer

    def _lexical_similarity(self, sampled_generations_dict: dict) -> dict:
        generated_texts = sampled_generations_dict["generated_texts"][: self.number_of_generations]
        mean_sim = _mean_pairwise_rougeL(generated_texts, self.rouge_scorer)
        return {
            # higher similarity = more self-consistent = more truthful
            "truth_value": mean_sim,
            "lexical_similarity": mean_sim,
            "generated_texts": generated_texts,
        }

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        messages: list = [],
        context: str = "",
        **kwargs,
    ):
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(
                model=model, input_text=input_text, tokenizer=tokenizer,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_generations,
                return_text=True, batch_generation=self.batch_generation, **kwargs,
            )
        return self._lexical_similarity(sampled_generations_dict)

    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs,
    ):
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(
                model=model, messages=messages, generation_seed=generation_seed,
                number_of_generations=self.number_of_generations, return_text=True, **kwargs,
            )
        return self._lexical_similarity(sampled_generations_dict)
