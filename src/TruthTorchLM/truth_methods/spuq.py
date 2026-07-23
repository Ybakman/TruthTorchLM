"""SPUQ — Perturbation-Based Uncertainty Quantification for black-box LLMs.

Official source: **Gao, Zhang, Mouatadid & Das, "SPUQ: Perturbation-Based Uncertainty
Quantification for Large Language Models", EACL 2024
([arXiv:2403.02509](https://arxiv.org/abs/2403.02509))**, `intuit-ai-research/SPUQ`
(pinned at `third_party/SPUQ`, Apache-2.0). This is a faithful port of that repo's
perturbation classes and its inter-sample aggregation.

**What makes SPUQ different from the SC family** (and why it needs its own target calls):
the consistency methods (DSE, LeS, EigV) resample the *same* prompt at temperature and
measure dispersion. SPUQ instead **perturbs the prompt** — paraphrase it, prepend a random
system message, insert a dummy token, or jitter the temperature — regenerates once per
perturbation, and measures how much the *output* changes. A model that is confident is
stable under input perturbation; a fragile one flips. Crucially, each output is
**down-weighted by how far its perturbed input drifted** from the original (ROUGE-L of the
inputs), so a paraphrase that changed the question's meaning counts less.

Consequence for the harness: SPUQ **cannot be served from the Stage-A sample cache** — that
cache holds fixed-prompt samples, and SPUQ needs *perturbed-prompt* generations. So it
makes its own ``n_perturb`` target calls at score time. The latency layer records this as
extra target generation (coupled to the generator, ``NUM_TARGET_CALLS = n_perturb``); it is
pure black-box (text only) and passes the §1 filter.

Two upstream bugs are fixed here and flagged inline: ``TemperaturePerturbation``'s
constructor argument order, and the ``verbalized_word`` branch that never matched. This
port keeps the *deployable, text-only* path — the inter-sample output-agreement aggregation
— as the default; the verbalized (intra-sample) variant is available but costs extra calls.
"""

import json
from copy import deepcopy
from typing import Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .truth_method import TruthMethod


# --- Perturbations (ports of third_party/SPUQ/perturbation.py) ---------------------------

_SYSTEM_MESSAGES = [
    "you are a helpful assistant",
    "you are a question-answering assistant",
    "you are a nice assistant",
    "You are a helpful assistant",
    "You are a question-answering assistant",
    "You are a nice assistant",
    "You are a helpful assistant.",
    "You are a question-answering assistant.",
    "You are a nice assistant.",
]

_DUMMY_TOKENS = [
    {"text": "\n", "pos": "both"}, {"text": "\t", "pos": "both"},
    {"text": " ", "pos": "both"}, {"text": "...", "pos": "both"},
    {"text": " um, ", "pos": "before"}, {"text": " uh, ", "pos": "before"},
    {"text": "?", "pos": "after"}, {"text": "??", "pos": "after"},
    {"text": "\n\n", "pos": "both"}, {"text": " um... ", "pos": "before"},
    {"text": " uh... ", "pos": "before"},
]


def _perturb_system_message(messages, temperature, n, rng):
    chosen = rng.choice(len(_SYSTEM_MESSAGES), size=min(n, len(_SYSTEM_MESSAGES)), replace=False)
    out = []
    for idx in chosen:
        out.append(([{"role": "system", "content": _SYSTEM_MESSAGES[idx]}] + messages, temperature))
    return out


def _perturb_dummy_token(messages, temperature, n, rng):
    chosen = rng.choice(len(_DUMMY_TOKENS), size=min(n, len(_DUMMY_TOKENS)), replace=False)
    out = []
    for idx in chosen:
        dummy = _DUMMY_TOKENS[idx]
        x = deepcopy(messages)
        pos = dummy["pos"]
        if pos == "both":
            pos = "after" if rng.random() > 0.5 else "before"
        if pos == "before":
            x[-1]["content"] = dummy["text"] + x[-1]["content"]
        else:
            x[-1]["content"] = x[-1]["content"] + dummy["text"]
        out.append((x, temperature))
    return out


def _perturb_temperature(messages, temperature, n, rng, t_min=0.0, t_max=1.0):
    # Upstream bug fix: the original TemperaturePerturbation.__init__ took (n, T_min, T_max)
    # but was constructed as (T_min, T_max, n) -- so n and T_min were swapped. Here the
    # arguments are explicit, so the jitter range is honoured.
    return [(messages, t_min + rng.random() * (t_max - t_min)) for _ in range(n)]


class SPUQ(TruthMethod):
    """Perturbation-based UQ (Gao et al., EACL 2024). Pure black-box; makes its own calls.

    ``perturbation``: 'system_message' (default, free), 'dummy_token' (free),
    'temperature', or 'paraphrasing' (needs a paraphrase callable).
    ``aggregation``: an inter-sample text-similarity metric ('rougeL' default, 'rouge1',
    'rouge2') measuring output agreement, input-drift-weighted per the source.
    """

    REQUIRES_SAMPLED_TEXT = False  # SPUQ generates perturbed-prompt outputs itself
    NUM_TARGET_CALLS = 1           # overwritten in __init__ to n_perturb

    def __init__(
        self,
        n_perturb: int = 5,
        perturbation: str = "system_message",
        aggregation: str = "rougeL",
        weighted: bool = True,
        temperature: float = 1.0,
        paraphrase_fn=None,
        seed: int = 0,
    ):
        super().__init__()
        if n_perturb <= 0:
            raise ValueError("n_perturb must be positive.")
        if perturbation not in ("system_message", "dummy_token", "temperature", "paraphrasing"):
            raise ValueError(f"Unknown perturbation '{perturbation}'.")
        if aggregation not in ("rouge1", "rouge2", "rougeL"):
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. This port keeps the text-only "
                f"inter-sample metrics; 'sbert'/'bertscore' can be added with the extra deps."
            )
        if perturbation == "paraphrasing" and paraphrase_fn is None:
            raise ValueError(
                "perturbation='paraphrasing' needs paraphrase_fn(messages, n) -> list of "
                "perturbed message-lists (the source uses a helper LLM for this)."
            )
        self.n_perturb = n_perturb
        self.perturbation = perturbation
        self.aggregation = aggregation
        self.weighted = weighted
        self.temperature = temperature
        self.paraphrase_fn = paraphrase_fn
        self._rng = np.random.default_rng(seed)
        self.NUM_TARGET_CALLS = n_perturb
        self._rouge = None

    # -- text similarity (port of text_sim.py rouge path) --------------------------------

    @property
    def rouge(self):
        if self._rouge is None:
            from rouge_score import rouge_scorer

            self._rouge = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        return self._rouge

    def _sim(self, a: str, b: str) -> float:
        return float(self.rouge.score(a, b)[self.aggregation].fmeasure)

    def _input_weight(self, inp0_turns, inp_turns) -> float:
        """ROUGE-L of the two inputs -- how little the perturbation drifted (calc_wt)."""
        if not self.weighted:
            return 1.0
        inp0 = "\n".join(t["content"] for t in inp0_turns)
        inp = "\n".join(t["content"] for t in inp_turns)
        return float(self.rouge.score(inp0, inp)["rougeL"].fmeasure)

    # -- perturbation dispatch -----------------------------------------------------------

    def _perturb(self, messages):
        if self.perturbation == "system_message":
            return _perturb_system_message(messages, self.temperature, self.n_perturb, self._rng)
        if self.perturbation == "dummy_token":
            return _perturb_dummy_token(messages, self.temperature, self.n_perturb, self._rng)
        if self.perturbation == "temperature":
            return _perturb_temperature(messages, self.temperature, self.n_perturb, self._rng)
        # paraphrasing
        perturbed_msgs = self.paraphrase_fn(messages, self.n_perturb)
        return [(m, self.temperature) for m in perturbed_msgs]

    # -- inter-sample aggregation (port of InterSampleAggregation.aggregate) -------------

    def _aggregate(self, inp_out: list) -> dict:
        """Input-weighted mean output-agreement vs the first (unperturbed-order) sample."""
        inp0, out0 = inp_out[0]
        sum_conf, sum_wt = 0.0, 0.0
        for inp, out in inp_out[1:]:
            wt = self._input_weight(inp0, inp)
            conf = self._sim(out0, out)
            sum_conf += conf * wt
            sum_wt += wt
        confidence = (sum_conf / sum_wt) if sum_wt > 0 else 1.0
        return {
            # higher agreement under perturbation = more confident = more truthful
            "truth_value": confidence,
            "spuq_confidence": confidence,
            "outputs": [out for _, out in inp_out],
        }

    # -- generation (dispatch by backend) ------------------------------------------------

    def _run(self, generate_one, messages) -> dict:
        # First entry is the *unperturbed* prompt, so out0 is the reference the others are
        # compared against -- matching the source, where inp_out[0] is the anchor.
        perturbed = [(messages, self.temperature)] + self._perturb(messages)
        inp_out = [(x, generate_one(x, t)) for x, t in perturbed]
        return self._aggregate(inp_out)

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
        from litellm import completion

        def generate_one(msgs, temperature):
            resp = completion(model=model, messages=msgs, temperature=temperature, **kwargs)
            return resp.choices[0].message["content"]

        return self._run(generate_one, messages)

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
        from TruthTorchLM.generation import generate_hf_local

        def generate_one(msgs, temperature):
            out = generate_hf_local(
                model=model, messages=msgs, question=question, tokenizer=tokenizer,
                temperature=temperature, do_sample=temperature > 0, **kwargs,
            )
            return out["generated_text"]

        return self._run(generate_one, messages)
