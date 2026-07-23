"""Concurrent multi-sample drawing for API targets (benchmark protocol §5).

Upstream ``sample_generations_api`` draws its N samples in a plain ``for`` loop, so the
serial-vs-concurrent question the protocol asks -- does concurrency rescue multi-sample
methods, or does the N× penalty survive? -- cannot be asked at all without this module.

Multi-sample draws are *independent*, so under concurrency N samples cost roughly the
**tail of N parallel generations** rather than N × one generation. Chain-of-interaction
turns are *dependent* (each conditions on the previous) and therefore cannot use this
path at any N; that asymmetry is why the protocol treats serial-multi-turn as the
worst-case latency and moving the chain onto a cheap proxy as the only available lever.

**The samples are identical to the serial path.** All N seeds are drawn up front, in
order, from the same seeded RNG the serial loop uses, then dispatched. So concurrent and
serial runs produce the same generations, and the comparison between them isolates
latency rather than confounding it with different draws -- which also keeps protocol §6's
"same generations for every method" control intact across the two execution modes.
"""

import copy
import random
from concurrent.futures import ThreadPoolExecutor

from litellm import completion

from .timing import Stage, stage

__all__ = ["sample_generations_api_concurrent", "DEFAULT_MAX_WORKERS"]

#: Conservative default. Real API accounts have per-minute request and token caps, and
#: exceeding them produces 429s whose retry delay would be measured as method latency --
#: turning a rate-limit incident into a fabricated performance result.
DEFAULT_MAX_WORKERS = 8


def sample_generations_api_concurrent(
    model: str,
    messages: list,
    generation_seed: int = None,
    number_of_generations: int = 0,
    return_text: bool = False,
    return_logits: bool = False,
    return_logprobs: bool = False,
    return_attentions: bool = False,
    return_activations: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs,
):
    """Concurrent drop-in for ``TruthTorchLM.generation.sample_generations_api``.

    Same signature, same return shape (``generated_texts`` / ``logprobs`` / ``tokens``,
    index-aligned), so callers can switch execution mode without touching anything else.
    """
    if number_of_generations == 0 or (not return_text and not return_logprobs):
        return None

    if generation_seed is not None:
        random.seed(generation_seed)

    kwargs = copy.deepcopy(kwargs)
    kwargs.pop("logprobs", None)
    kwargs.pop("seed", None)

    # Drawn here, in order, before any dispatch -- this is what makes the concurrent path
    # reproduce the serial path's samples exactly.
    seeds = [random.randint(0, 1000000) for _ in range(number_of_generations)]

    def _one(index: int):
        call_kwargs = dict(kwargs)
        call_kwargs["seed"] = seeds[index]
        return index, completion(
            model=model, messages=messages, logprobs=return_logprobs, **call_kwargs
        )

    responses = [None] * number_of_generations
    workers = max(1, min(max_workers, number_of_generations))

    # Timed as one span: under concurrency the cost is the tail of the parallel batch,
    # not the sum of the parts, and summing per-call durations here would report the
    # serial number while running concurrently -- exactly the confusion §5 warns about.
    with stage(
        Stage.EXTRA_GENERATION,
        label="sample_generations_api_concurrent",
        n=number_of_generations,
        execution="concurrent",
        max_workers=workers,
    ):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, response in pool.map(_one, range(number_of_generations)):
                responses[index] = response

    generated_texts = []
    logprobs_list = []
    token_lists = []
    for response in responses:
        if return_text:
            generated_texts.append(response.choices[0].message["content"])
        if return_logprobs:
            logprobs_list.append(
                [token["logprob"] for token in response.choices[0].logprobs["content"]]
            )
            token_lists.append(
                [token["token"] for token in response.choices[0].logprobs["content"]]
            )

    return {
        "generated_texts": generated_texts,
        "logprobs": logprobs_list,
        "tokens": token_lists,
    }
