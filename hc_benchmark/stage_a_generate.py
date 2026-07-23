"""Stage A -- generate and cache (protocol §7).

For each (item, target) draw the primary answer plus ``n_max`` samples once, and persist
them. This is the single most important control in the whole benchmark: every method and
every N in the sweep consumes this identical cache, so generation cost and generation
variance are removed as confounds between methods (protocol §6).

Maps onto DisAAD's ``generate_black.py`` (the "gene" pass), so results are comparable to
theirs.
"""

from tqdm import tqdm

from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.utils.dataset_utils import get_dataset

from .cache import GenerationCache

__all__ = ["generate_stage_a", "generate_stage_a_local"]


def _build_messages(item, user_prompt, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    if item.get("context"):
        content = ("Context: {context}\n" + user_prompt).format(
            context=item["context"], question=item["question"]
        )
    else:
        content = user_prompt.format(question=item["question"])
    messages.append({"role": "user", "content": content})
    return messages


def generate_stage_a(
    config,
    seed: int,
    cache_root: str = "hc_benchmark/cache",
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_BENCHMARK_PROMPT,
    overwrite: bool = False,
    _generate_fn=None,
    _sample_fn=None,
):
    """Produce and cache primary answer + ``n_max`` samples for one (config, seed) shard.

    ``_generate_fn`` / ``_sample_fn`` are injection points for testing without a live
    model: they default to the real API primitives. Only the ``litellm`` backend is wired
    here; the ``huggingface`` path raises with a pointer, because a local target also
    needs a loaded model + tokenizer handed in, which belongs to the caller's launch
    script rather than this orchestrator.
    """
    cache = GenerationCache(cache_root, config, seed)
    if cache.exists() and not overwrite:
        print(f"[Stage A] cache hit: {cache.path.name} ({len(cache)} items) -- skipping.")
        return cache

    backend = config.resolved_backend()
    if backend == "litellm":
        generate_fn = _generate_fn or _default_api_generate
        sample_fn = _sample_fn or _default_api_sample
    else:
        raise NotImplementedError(
            "Stage A's huggingface backend needs a loaded model+tokenizer passed from the "
            "launch script. Use generate_stage_a_local(...) with those objects, or the "
            "litellm backend for API targets."
        )

    # Resolve the friendly generator name (e.g. 'claude-opus-4-8') to the provider-qualified
    # model string LiteLLM expects (e.g. 'anthropic/claude-opus-4-8'). See generators.py.
    model = config.model_string()

    dataset = get_dataset(config.dataset, size_of_data=config.size_of_data, seed=seed)
    dec = config.decoding

    items = []
    for i, data in enumerate(tqdm(dataset, desc=f"[Stage A] {config.dataset}/{config.generator} seed{seed}")):
        messages = _build_messages(data, user_prompt, system_prompt)

        primary = generate_fn(
            model=model, messages=messages,
            temperature=dec.temperature, top_p=dec.top_p, max_tokens=dec.max_tokens,
            seed=seed,
        )
        samples = sample_fn(
            model=model, messages=messages, number_of_generations=config.n_max,
            temperature=dec.sample_temperature, top_p=dec.sample_top_p,
            max_tokens=dec.max_tokens, generation_seed=seed,
        )

        items.append(
            {
                "item_id": data.get("item_id", i),
                "question": data["question"],
                "context": data.get("context", ""),
                "ground_truths": data["ground_truths"],
                "primary_answer": primary,
                "samples": samples,
                "stratum": data.get("stratum"),
                "outcome_type": data.get("outcome_type", "factual_error"),
            }
        )

    cache.write(items)
    print(f"[Stage A] wrote {len(items)} items -> {cache.path.name}")
    return cache


def generate_stage_a_local(
    config,
    seed: int,
    model,
    tokenizer,
    cache_root: str = "hc_benchmark/cache",
    user_prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_BENCHMARK_PROMPT,
    overwrite: bool = False,
):
    """Stage A for an **open (HuggingFace) target** — the white-box reference-line path.

    The API path (:func:`generate_stage_a`) can't cover this: a local target needs a
    loaded ``model`` + ``tokenizer`` passed in from the launch script (they don't live in
    a YAML). This function takes them and drives TTLM's own HF generation primitives, so
    the cache it writes is identical in shape to the API path — every downstream stage is
    backend-agnostic.

    Open targets give a small, controlled generation time ``g`` on our own hardware, and
    are the only targets whose own logits are available — which is what makes the §6
    white-box reference line possible. This function produces the *text* cache; a
    white-box scorer reads the same target's logits separately.
    """
    from TruthTorchLM.generation import generate_hf_local, sample_generations_hf_local

    cache = GenerationCache(cache_root, config, seed)
    if cache.exists() and not overwrite:
        print(f"[Stage A/local] cache hit: {cache.path.name} ({len(cache)} items) -- skipping.")
        return cache

    if config.resolved_backend() != "huggingface":
        raise ValueError(
            f"generate_stage_a_local is for the huggingface backend, but config resolves to "
            f"'{config.resolved_backend()}'. Use generate_stage_a for API targets."
        )

    dataset = get_dataset(config.dataset, size_of_data=config.size_of_data, seed=seed)
    dec = config.decoding

    items = []
    for i, data in enumerate(tqdm(dataset, desc=f"[Stage A/local] {config.dataset}/{config.generator} seed{seed}")):
        messages = _build_messages(data, user_prompt, system_prompt)

        # Primary answer. generate_hf_local also returns the chat-templated `text`, which
        # is exactly the prompt the sampler must condition on so its draws match the
        # primary's context.
        primary_out = generate_hf_local(
            model=model, messages=messages, question=data["question"], tokenizer=tokenizer,
            temperature=dec.temperature, top_p=dec.top_p, max_new_tokens=dec.max_tokens,
            do_sample=dec.temperature > 0,
        )
        sampled = sample_generations_hf_local(
            model=model, input_text=primary_out["text"], tokenizer=tokenizer,
            generation_seed=seed, number_of_generations=config.n_max, return_text=True,
            batch_generation=True, temperature=dec.sample_temperature, top_p=dec.sample_top_p,
            max_new_tokens=dec.max_tokens, do_sample=True,
        )

        items.append(
            {
                "item_id": data.get("item_id", i),
                "question": data["question"],
                "context": data.get("context", ""),
                "ground_truths": data["ground_truths"],
                "primary_answer": primary_out["generated_text"],
                "samples": (sampled or {}).get("generated_texts", []),
                "stratum": data.get("stratum"),
                "outcome_type": data.get("outcome_type", "factual_error"),
            }
        )

    cache.write(items)
    print(f"[Stage A/local] wrote {len(items)} items -> {cache.path.name}")
    return cache


def _default_api_generate(model, messages, **kwargs):
    from TruthTorchLM.generation import generate_api

    out = generate_api(model=model, messages=messages, **kwargs)
    return out["generated_text"]


def _default_api_sample(model, messages, number_of_generations, generation_seed, **kwargs):
    from TruthTorchLM.generation import sample_generations_api

    out = sample_generations_api(
        model=model, messages=messages, number_of_generations=number_of_generations,
        return_text=True, generation_seed=generation_seed, **kwargs
    )
    return out["generated_texts"] if out else []
