"""Stage C -- score (protocol §7, and the §5 latency measurement).

Each UQ method reads the Stage-A cache, emits one uncertainty score per item, and logs
its own timing record -- swept over ``N in {1,3,5,10,20}`` by *truncating* the cached
sample list. Because every N reuses the identical cached draws, the N-axis of the
frontier reflects the sample budget alone, not re-sampling noise.

The key move is that a cached sample list is handed to each ``TruthMethod`` as a
pre-built ``sampled_generations_dict``. Upstream methods already accept that argument and
skip generation when it is supplied (see ``SemanticEntropy.forward_api``), so the target
is never called again at score time -- exactly what the shared-generation control
requires. The only thing the method spends here is its **auxiliary compute** (NLI
clustering, embedding, proxy pass), which is Stage 3 of the §5 decomposition and the
number this stage is built to measure.
"""

from tqdm import tqdm

from TruthTorchLM.instrumentation.timing import Stage, capture, stage

__all__ = ["score_stage_c", "MethodScores"]


class MethodScores(dict):
    """Scores + timing for one method at one N, across all items."""


def _sampled_dict_from_cache(item, n):
    """Build the ``sampled_generations_dict`` a TruthMethod expects from cached text.

    Only ``generated_texts`` is populated: the pure black-box methods need sampled text
    and nothing else. A method that also wants logprobs is grey-box, excluded from this
    benchmark by the §1 filter, and would (correctly) find them absent here.
    """
    samples = item["samples"][:n]
    return {"generated_texts": samples, "logprobs": [], "tokens": []}


def score_stage_c(
    cache,
    truth_methods,
    model,
    n_sweep=(1, 3, 5, 10, 20),
    tokenizer=None,
    collect_timing: bool = True,
):
    """Score every cached item with every method at every N in the sweep.

    Returns ``{method_name: {n: MethodScores}}``. Each ``MethodScores`` holds
    ``truth_values``, ``normalized_truth_values``, and (if timing) one auxiliary-compute
    ``timing_ms`` per item -- the query-independent floor §5 asks to be made visible,
    isolated from the shared sampling cost.
    """
    items = cache.read()  # full n_max draws; per-N truncation happens per item below
    results = {}

    for method in truth_methods:
        name = type(method).__name__
        results[name] = {}
        for n in n_sweep:
            if n > getattr(method, "number_of_generations", n):
                # A method fixed at k samples cannot use more; record it once at its k.
                pass
            scores = MethodScores(truth_values=[], normalized_truth_values=[], timing_ms=[])
            for item in tqdm(items, desc=f"[Stage C] {name} N={n}", leave=False):
                sampled = _sampled_dict_from_cache(item, n)
                messages = _messages_for(item)

                ctx = capture(method=name, n=n) if collect_timing else _null()
                with ctx as rec:
                    with stage(Stage.AUXILIARY_COMPUTE, label=name):
                        out = method(
                            model=model,
                            input_text="",
                            generated_text=item["primary_answer"],
                            question=item["question"],
                            messages=messages,
                            context=item.get("context", ""),
                            sampled_generations_dict=sampled,
                            tokenizer=tokenizer,
                        )
                scores["truth_values"].append(out["truth_value"])
                scores["normalized_truth_values"].append(out["normalized_truth_value"])
                if collect_timing and rec is not None:
                    scores["timing_ms"].append(rec.stage_ms(Stage.AUXILIARY_COMPUTE))
            results[name][n] = scores
    return results


def _messages_for(item):
    return [{"role": "user", "content": item["question"]}]


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
