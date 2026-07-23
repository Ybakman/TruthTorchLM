"""DisAAD — proxy-based black-box UQ via distribution-aligned adversarial distillation.

Official source: **Cui, Ma, Wang, Gao & Zhang, "Estimating the Black-box LLM Uncertainty
with Distribution-Aligned Adversarial Distillation", ACL 2026
([2026.acl-long.1979](https://aclanthology.org/2026.acl-long.1979/),
[arXiv:2605.05777](https://arxiv.org/abs/2605.05777))**, `huizi-Cui/DisAAD` (pinned at
`third_party/DisAAD`). **The source declares no license**, so its code is used only as an
unmodified submodule — never copied. Proxy *training* is driven by shelling out to the
official scripts (see `hc_benchmark/disaad_train.py`); this module is the **Stage-2
inference** side: given a trained proxy, score a target response by the proxy's evidential
uncertainty.

DisAAD is a two-stage method:

* **Stage 1 (offline, per target).** Collect the black-box target's outputs across diverse
  prompts, then adversarially distil a small open **proxy** (student) that reproduces the
  target's (teacher's) high-probability output regions. Runs on the GPU cluster via the
  official scripts — *setup only here, no local training* (see the training entry point).
* **Stage 2 (online).** Run the target's response through the distilled proxy and read
  uncertainty off the proxy's logits via **evidential deep learning** (a Dirichlet view of
  the top-k logits as evidence). One small proxy forward pass, **decoupled** from the
  target — protocol §5's "Pob via proxy": roughly constant ms regardless of how slow the
  target is, so its overhead *ratio* shrinks on a slow/LRM target.

The evidential measures below are re-implemented from the standard EDL formulation (the
same one in the source's MIT-headed `scripts/metrics.py`), so they are testable in
isolation; the proxy forward pass needs the trained checkpoint and runs on the cluster.
"""

from typing import Union

import numpy as np
import torch
from scipy.special import digamma, softmax
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .truth_method import TruthMethod

__all__ = [
    "evidential_epistemic",
    "evidential_aleatoric",
    "max_softmax_probability",
    "softmax_entropy",
    "DisAAD",
]


def _topk(logits: np.ndarray, k: int) -> np.ndarray:
    if len(logits) < k:
        raise ValueError(f"logit vector length {len(logits)} < k={k}.")
    idx = np.argpartition(logits, -k)[-k:]
    return np.asarray(logits)[idx]


def evidential_epistemic(logits: np.ndarray, k: int) -> float:
    """EU: epistemic (evidence-scarcity) uncertainty from the top-k logits.

    Port of `metrics.get_eu(mode='eu')`: ``k / (sum(max(0, top-k logits)) + k)``. Reading
    the top-k logits as Dirichlet evidence, this is large when total evidence is small
    (the proxy is unsure) and small when evidence is abundant.
    """
    top_values = _topk(np.asarray(logits, dtype=float), k)
    return float(k / (np.sum(np.maximum(0.0, top_values)) + k))


def evidential_aleatoric(logits: np.ndarray, k: int) -> float:
    """AU: aleatoric (expected data) uncertainty of the Dirichlet over the top-k logits.

    Port of `metrics.get_eu(mode='au')`: with evidence ``alpha`` = top-k logits and
    ``alpha_0 = sum(alpha)``,
    ``AU = -sum( (alpha_i/alpha_0) * (digamma(alpha_i+1) - digamma(alpha_0+1)) )`` — the
    standard EDL expected entropy.
    """
    alpha = _topk(np.asarray(logits, dtype=float), k).reshape(1, -1)
    alpha_0 = alpha.sum(axis=1, keepdims=True)
    result = -(alpha / alpha_0) * (digamma(alpha + 1) - digamma(alpha_0 + 1))
    return float(result.sum(axis=1)[0])


def max_softmax_probability(logits: np.ndarray) -> float:
    """MSP: the maximum softmax probability (a confidence baseline; higher = more certain)."""
    return float(np.max(softmax(np.asarray(logits, dtype=float))))


def softmax_entropy(logits: np.ndarray) -> float:
    """Shannon entropy of the softmax distribution (higher = more uncertain)."""
    probs = softmax(np.asarray(logits, dtype=float))
    return float(-np.sum(probs * np.log(probs + 1e-10)))


class DisAAD(TruthMethod):
    """Proxy-based UQ (Cui et al., ACL 2026), Stage-2 inference. Pure black-box (text only).

    Requires a **trained proxy** (the distilled student) — a HuggingFace CausalLM +
    tokenizer produced by the offline training step. Scores the target's response by the
    proxy's per-token evidential uncertainty, aggregated over the response.

    Single-instance and decoupled: it consumes the target's already-produced answer and
    runs one proxy forward pass; it makes **no** extra target calls.
    """

    REQUIRES_SAMPLED_TEXT = False
    NUM_TARGET_CALLS = 1  # only the user's answer; the proxy pass is auxiliary compute

    # Readiness signalling: DisAAD is the one method that needs an offline training step
    # (a distilled proxy) before it can score. The benchmark's readiness report and the
    # error messages below use this so users learn a proxy is needed *before* a run, and
    # can tell when one is trained and ready.
    REQUIRES_TRAINING = True
    TRAINING_HINT = (
        "Train a proxy on the GPU server: "
        "`from hc_benchmark.disaad_train import DisAADTrainingConfig, train_proxy; "
        "train_proxy(DisAADTrainingConfig(teacher_model=..., student_model=...))`, "
        "then load it with `DisAAD.from_pretrained(proxy_path)`."
    )

    def __init__(
        self,
        proxy_model: PreTrainedModel = None,
        proxy_tokenizer: PreTrainedTokenizer = None,
        mode: str = "au",
        top_k: int = 10,
        device: str = "cuda",
        proxy_path: str = None,
    ):
        super().__init__()
        if mode not in ("au", "eu", "msp", "entropy"):
            raise ValueError(f"mode must be one of au/eu/msp/entropy, got '{mode}'.")
        self.proxy_model = proxy_model
        self.proxy_tokenizer = proxy_tokenizer
        self.mode = mode
        self.top_k = top_k
        self.device = device
        self.proxy_path = proxy_path

    def is_ready(self) -> bool:
        """True if a trained proxy is loaded (or a ready proxy is on disk at ``proxy_path``).

        Lets callers check readiness *before* a run instead of hitting an error mid-scoring.
        """
        if self.proxy_model is not None and self.proxy_tokenizer is not None:
            return True
        if self.proxy_path is not None:
            return self.is_trained(self.proxy_path)
        return False

    @staticmethod
    def is_trained(proxy_path: str) -> bool:
        """True if ``proxy_path`` holds a completed proxy (has the training-ready manifest)."""
        from hc_benchmark.disaad_train import is_proxy_trained

        return is_proxy_trained(proxy_path)

    @classmethod
    def from_pretrained(cls, proxy_path: str, mode: str = "au", top_k: int = 10,
                        device: str = "cuda", allow_unverified: bool = False, **kwargs):
        """Load a trained proxy checkpoint into a scorer (run on the cluster).

        Refuses a proxy directory that lacks the training-ready manifest — that usually
        means training didn't finish, or the path is wrong. Pass ``allow_unverified=True``
        to load a proxy trained outside this harness (no manifest).
        """
        if not allow_unverified and not cls.is_trained(proxy_path):
            raise RuntimeError(
                f"No DisAAD training manifest at '{proxy_path}' — the proxy is not marked "
                f"trained/ready. If training is still running or didn't finish, wait for it "
                f"to complete. {cls.TRAINING_HINT} "
                f"To load a proxy trained outside this harness, pass allow_unverified=True."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(proxy_path, **kwargs).to(device)
        tokenizer = AutoTokenizer.from_pretrained(proxy_path)
        return cls(proxy_model=model, proxy_tokenizer=tokenizer, mode=mode, top_k=top_k,
                   device=device, proxy_path=proxy_path)

    def _token_uncertainty(self, logits_vector: np.ndarray) -> float:
        if self.mode == "au":
            return evidential_aleatoric(logits_vector, self.top_k)
        if self.mode == "eu":
            return evidential_epistemic(logits_vector, self.top_k)
        if self.mode == "msp":
            return -max_softmax_probability(logits_vector)  # negate: MSP is confidence
        return softmax_entropy(logits_vector)

    def _score_with_proxy(self, prompt_text: str, generated_text: str) -> dict:
        """Run the proxy over prompt+response and aggregate per-token evidential uncertainty."""
        if self.proxy_model is None or self.proxy_tokenizer is None:
            raise RuntimeError(
                "DisAAD needs a trained proxy. Train one on the cluster "
                "(hc_benchmark/disaad_train.py) and load it via DisAAD.from_pretrained(path). "
                "Round one ships the setup only; no proxy is trained locally."
            )
        tok = self.proxy_tokenizer
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
        full_ids = tok(prompt_text + generated_text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            logits = self.proxy_model(full_ids).logits[0]  # (seq, vocab)
        # score the generated span: positions predicting each response token
        start = prompt_ids.shape[1] - 1
        response_logits = logits[start:-1].float().cpu().numpy()
        if len(response_logits) == 0:
            response_logits = logits[-1:].float().cpu().numpy()
        per_token = [self._token_uncertainty(v) for v in response_logits]
        uncertainty = float(np.mean(per_token)) if per_token else 0.0
        return {
            "truth_value": -uncertainty,          # higher = more certain
            "disaad_uncertainty": uncertainty,
            "mode": self.mode,
            "num_response_tokens": len(per_token),
        }

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
        # The *target* is a black-box API, but scoring runs on our local proxy -- that is
        # the whole point of DisAAD. Reconstruct the prompt text from the messages.
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        return self._score_with_proxy(prompt_text, generated_text)

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
        return self._score_with_proxy(input_text, generated_text)
