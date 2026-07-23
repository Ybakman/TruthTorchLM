#!/usr/bin/env python
"""Confirm a closed API target actually behaves as a pure black-box generator.

The protocol's G table lists "Anthropic (Claude), Gemini via LiteLLM — verify text-only
BB" as a to-do. This is that verification: for each configured API provider it (1) checks
the key is set, (2) makes one real generation call through the same LiteLLM path the
harness uses, and (3) confirms the response is text-only — no token log-probabilities —
which is what makes the target admissible under the benchmark's §1 constraint.

Needs the provider API keys and litellm installed. It is intentionally tiny (one call per
provider) — it checks the wiring, not model quality.

    python scripts/verify_provider.py                  # all configured providers
    python scripts/verify_provider.py anthropic gemini # a subset
"""

import os
import sys

sys.path.insert(0, "src")

from hc_benchmark.generators import Access, GENERATORS, generators_for


def _one_per_provider(providers):
    """Pick one black-box spec per requested provider (smallest/cheapest first)."""
    chosen = {}
    for spec in generators_for(access=Access.BLACK_BOX):
        if providers and spec.provider not in providers:
            continue
        # Prefer the cheaper tier for a wiring check.
        if spec.provider not in chosen or "small" in spec.size_hint:
            chosen[spec.provider] = spec
    return chosen


def verify(spec):
    from litellm import completion

    if spec.env_var and not os.environ.get(spec.env_var):
        return False, f"{spec.env_var} not set"

    try:
        resp = completion(
            model=spec.model_id,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=8,
            logprobs=True,  # ask for logprobs -- a true black-box API won't return them
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"call failed: {type(exc).__name__}: {exc}"

    text = resp.choices[0].message["content"]
    # The admissibility check: a pure black-box target returns text and no usable
    # token log-probabilities, even when we ask for them.
    logprobs = getattr(resp.choices[0], "logprobs", None)
    has_logprobs = bool(logprobs and getattr(logprobs, "content", None))

    verdict = "TEXT-ONLY (black-box OK)" if not has_logprobs else \
        "RETURNED LOGPROBS -- grey-box, not pure black-box"
    return (not has_logprobs), f"got {text!r}; {verdict}"


def main():
    providers = set(sys.argv[1:]) or None
    chosen = _one_per_provider(providers)
    if not chosen:
        sys.exit(f"No black-box generators for {providers or 'any provider'} in the registry.")

    print("Verifying pure-black-box behaviour of API targets (one call each):\n")
    all_ok = True
    for provider, spec in sorted(chosen.items()):
        ok, detail = verify(spec)
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {provider:10s} {spec.name:22s} {detail}")

    print("\n" + ("All configured providers behave as pure black-box targets."
                  if all_ok else
                  "Some providers are unavailable or not text-only -- see above."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
