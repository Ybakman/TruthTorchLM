"""Discrete Semantic Entropy (DSE) — the text-only semantic-entropy variant.

Official source: **Farquhar, Kossen, Kuhn & Gal, "Detecting hallucinations in large
language models using semantic entropy", Nature 2024**, implemented in
`jlko/semantic_uncertainty` (pinned at `third_party/semantic_uncertainty`). This is a
faithful port of that repo's ``cluster_assignment_entropy(semantic_ids)`` over semantic
clusters produced by its ``get_semantic_ids`` (bidirectional NLI entailment, non-strict).

Why this method and not upstream ``SemanticEntropy``: upstream TTLM's ``SemanticEntropy``
sets ``REQUIRES_SAMPLED_LOGPROBS = True`` — it weights clusters by token log-likelihoods,
which a black-box API target does not expose, so it fails the benchmark's §1 filter. The
**discrete** variant estimates the categorical distribution over cluster *assignments*
directly from the samples and takes its entropy — no token likelihoods, text only. That is
exactly the black-box regime this benchmark lives in.

Why this method and not ``NumSemanticSetUncertainty``: that method returns the cluster
*count* |C|. DSE returns the *entropy of the assignment distribution* — it distinguishes
"5 samples split 4/1 across 2 clusters" (low entropy, fairly certain) from "5 samples
split 3/2" or "1/1/1/1/1" (higher entropy). The count collapses those; the entropy keeps
them, which is the whole point of the Nature 2024 estimator.

The clustering itself reuses TTLM's ``bidirectional_entailment_clustering`` (semantic
method: two texts share a cluster iff neither entails a contradiction of the other),
which implements the same non-strict bidirectional rule as ``get_semantic_ids``. Only the
uncertainty *aggregation* is ported here.
"""

import numpy as np
import torch
from typing import Union
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DebertaForSequenceClassification,
    DebertaTokenizer,
)

from TruthTorchLM.utils import bidirectional_entailment_clustering
from .truth_method import TruthMethod
from ..generation import sample_generations_hf_local, sample_generations_api


def cluster_assignment_entropy(semantic_ids: list) -> float:
    """Entropy of the categorical distribution over cluster assignments.

    Verbatim port of ``jlko/semantic_uncertainty``'s function of the same name:
    estimate p over clusters as the assignment frequencies, return ``-(p * log p).sum()``.
    Spread mass across many clusters -> larger entropy; concentrated -> smaller.
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    # nonzero guard: empty clusters contribute 0 (0 * log 0 -> 0), matching the source
    # which only ever sees occupied ids because bincount indexes contiguous 0..K-1.
    nonzero = probabilities[probabilities > 0]
    entropy = -(nonzero * np.log(nonzero)).sum()
    return float(entropy)


def _semantic_ids_from_clusters(clusters: list, generated_texts: list) -> list:
    """Map TTLM's cluster lists (list[list[str]]) to a per-sample semantic-id list.

    ``get_semantic_ids`` returns one id per generation, in generation order; TTLM's
    clustering returns groups of texts. This converts the latter to the former so the
    ported entropy sees exactly the input its source expects — one id per sample, in the
    original sample order.
    """
    text_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for text in cluster:
            # first assignment wins if a text appears verbatim in two clusters (it won't,
            # but be defensive rather than silently overwrite)
            text_to_cluster.setdefault(text, cluster_id)
    return [text_to_cluster[text] for text in generated_texts]


class DiscreteSemanticEntropy(TruthMethod):
    """Text-only semantic entropy (Farquhar et al., Nature 2024; discrete variant).

    Pure black-box: needs sampled text only, no token log-probabilities.
    """

    REQUIRES_SAMPLED_TEXT = True
    # Deliberately NOT REQUIRES_SAMPLED_LOGPROBS -- that is what makes this admissible
    # under the §1 filter where upstream SemanticEntropy is not.

    def __init__(
        self,
        number_of_generations: int = 5,
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device: str = "cuda",
        method_for_similarity: str = "semantic",
        batch_generation: bool = True,
    ):
        super().__init__()

        if (model_for_entailment is None or tokenizer_for_entailment is None) \
                and method_for_similarity == "semantic":
            # Same NLI backbone the rest of the SC family uses (microsoft/deberta-large-mnli),
            # matching the entailment step in get_semantic_ids.
            model_for_entailment = DebertaForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained(
                "microsoft/deberta-large-mnli"
            )

        if method_for_similarity not in ("semantic", "jaccard"):
            raise ValueError("method_for_similarity must be 'semantic' or 'jaccard'.")

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.method_for_similarity = method_for_similarity
        self.number_of_generations = number_of_generations
        self.batch_generation = batch_generation

    def _discrete_semantic_entropy(self, sampled_generations_dict: dict, question: str) -> dict:
        generated_texts = sampled_generations_dict["generated_texts"][: self.number_of_generations]

        if len(generated_texts) == 0:
            # Nothing to cluster: maximally uncertain by convention (no signal).
            return {"truth_value": float("-inf"), "discrete_semantic_entropy": float("inf"),
                    "clusters": [], "semantic_ids": [], "generated_texts": generated_texts}

        clusters = bidirectional_entailment_clustering(
            self.model_for_entailment,
            self.tokenizer_for_entailment,
            question,               # question as the clustering "context", as elsewhere in TTLM
            generated_texts,
            method=self.method_for_similarity,
        )
        semantic_ids = _semantic_ids_from_clusters(clusters, generated_texts)
        entropy = cluster_assignment_entropy(semantic_ids)

        return {
            # higher truth value = more certain, so negate the entropy (uncertainty)
            "truth_value": -entropy,
            "discrete_semantic_entropy": entropy,
            "num_semantic_clusters": len(clusters),
            "semantic_ids": semantic_ids,
            "clusters": clusters,
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
                model=model,
                input_text=input_text,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_generations,
                return_text=True,
                batch_generation=self.batch_generation,
                **kwargs,
            )
        return self._discrete_semantic_entropy(sampled_generations_dict, question)

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
                model=model,
                messages=messages,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_generations,
                return_text=True,
                **kwargs,
            )
        return self._discrete_semantic_entropy(sampled_generations_dict, question)
