"""Generator registry — the G axis of the benchmark (protocol §2, G table).

TruthTorchLM already ships both backends the benchmark needs, so adding a target is
mostly configuration, not code: **LiteLLM** for closed APIs (OpenAI, Anthropic, Gemini)
and **HuggingFace** for open models we host ourselves. This module is the single place
that configuration lives, so a spec and a config file can't drift on model IDs.

Three things the protocol's G table asks for that are encoded here rather than left implicit:

* **Access level.** Closed APIs are *black-box* targets (text only — the benchmark's whole
  premise). Open models we host are *white-box-capable*: usable as the reference line and
  as the small-proxy substrate the P family needs. `access` records which.
* **LRM (reasoning) vs plain LLM,** and **reasoning-trace handling.** A reasoning model
  emits a trace before its answer. Whether a UQ method sees the trace or only the final
  answer changes both its accuracy and its latency (protocol §5: the worst matrix cell is
  SC × closed LRM, re-running the whole trace N times). `is_reasoning` flags it and
  `reasoning_trace` records the policy; the harness must honour it, not guess.
* **Role.** A model can be a *target* (the thing we score), a *proxy* (a small open model
  the P family routes through), or *both*. `roles` records it.

Model IDs verified against the providers in July 2026. They are the current-generation
defaults; edit the registry as new models ship — nothing else in the harness hard-codes a
model string.

**Nothing here calls a model.** Building a spec does not load weights or hit an API; the
registry is inert data that Stage A consumes. That keeps it importable in a minimal env
and testable without credentials.
"""

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Access",
    "Role",
    "ReasoningTrace",
    "GeneratorSpec",
    "GENERATORS",
    "get_generator",
    "generators_for",
]


class Access(str, Enum):
    """Whether a target exposes only its text (black-box) or its internals (white-box)."""

    BLACK_BOX = "black-box"    # closed API: generated text only — the benchmark's regime
    WHITE_BOX = "white-box"    # open weights: enables the reference line + proxy substrate


class Role(str, Enum):
    TARGET = "target"  # a model whose responses we score
    PROXY = "proxy"    # a small open model the P family routes through (DisAAD, ref proxy)


class ReasoningTrace(str, Enum):
    """How a reasoning model's chain-of-thought is handled when it is the target.

    NONE   -- not a reasoning model; nothing to handle.
    ANSWER_ONLY -- strip the trace; UQ methods see only the final answer. The cheap,
                   deployment-realistic default for a guardrail that never sees the trace.
    INCLUDE_TRACE -- the trace is part of the scored text. Costs the most (SC re-runs it
                   N times) and is usually only meaningful on open LRMs we host.
    """

    NONE = "none"
    ANSWER_ONLY = "answer_only"
    INCLUDE_TRACE = "include_trace"


@dataclass(frozen=True)
class GeneratorSpec:
    """One configured generator (target and/or proxy)."""

    name: str                       # friendly key used in configs and result tables
    model_id: str                   # LiteLLM model string, or HuggingFace repo id
    backend: str                    # "litellm" (API) or "huggingface" (local)
    provider: str                   # openai | anthropic | gemini | open
    access: Access
    roles: tuple = (Role.TARGET,)
    is_reasoning: bool = False      # LRM (o-series, thinking, DeepSeek-R1, ...)
    reasoning_trace: ReasoningTrace = ReasoningTrace.NONE
    env_var: str = None             # API key env var (None for local models)
    size_hint: str = ""             # rough scale, for the proxy-slot picker and notes
    notes: str = ""

    def is_black_box(self) -> bool:
        return self.access is Access.BLACK_BOX

    def can_be_proxy(self) -> bool:
        return Role.PROXY in self.roles


# ---------------------------------------------------------------------------
# The registry. Ordered by provider; closed APIs first (the black-box targets),
# then open models (reference line + proxy substrate).
# ---------------------------------------------------------------------------

_SPECS = [
    # --- OpenAI (closed, black-box) -- already tested upstream ---------------
    GeneratorSpec("gpt-4o", "gpt-4o", "litellm", "openai", Access.BLACK_BOX,
                  env_var="OPENAI_API_KEY", size_hint="frontier",
                  notes="GPT-4-class API target; confirms the black-box story on a true API."),
    GeneratorSpec("gpt-4o-mini", "gpt-4o-mini", "litellm", "openai", Access.BLACK_BOX,
                  env_var="OPENAI_API_KEY", size_hint="small-frontier",
                  notes="Cheap default for smoke tests and the LLM judge."),

    # --- Anthropic (closed, black-box) -- Claude via LiteLLM ----------------
    # LiteLLM routes Anthropic through the `anthropic/` prefix; the bare model ids are
    # the current-generation strings (see the claude-api reference).
    GeneratorSpec("claude-opus-4-8", "anthropic/claude-opus-4-8", "litellm", "anthropic",
                  Access.BLACK_BOX, is_reasoning=True,
                  reasoning_trace=ReasoningTrace.ANSWER_ONLY,
                  env_var="ANTHROPIC_API_KEY", size_hint="frontier",
                  notes="Most capable Claude; thinking model -- guardrail sees the answer, "
                        "not the (never-returned raw) trace. The costly SC x closed-LRM cell."),
    GeneratorSpec("claude-sonnet-5", "anthropic/claude-sonnet-5", "litellm", "anthropic",
                  Access.BLACK_BOX, is_reasoning=True,
                  reasoning_trace=ReasoningTrace.ANSWER_ONLY,
                  env_var="ANTHROPIC_API_KEY", size_hint="frontier",
                  notes="Near-Opus quality at Sonnet cost; adaptive thinking."),
    GeneratorSpec("claude-haiku-4-5", "anthropic/claude-haiku-4-5", "litellm", "anthropic",
                  Access.BLACK_BOX, env_var="ANTHROPIC_API_KEY", size_hint="small-frontier",
                  notes="Fast/cheap Claude tier; a plausible real coaching-agent target."),

    # --- Google Gemini (closed, black-box) -- via LiteLLM `gemini/` prefix ---
    # gemini-2.5-pro is GA-stable through Oct 2026; the 3.x flash line is the newer default.
    GeneratorSpec("gemini-2.5-pro", "gemini/gemini-2.5-pro", "litellm", "gemini",
                  Access.BLACK_BOX, is_reasoning=True,
                  reasoning_trace=ReasoningTrace.ANSWER_ONLY,
                  env_var="GEMINI_API_KEY", size_hint="frontier",
                  notes="Gemini structured-reasoning target; GA through 2026-10-16."),
    GeneratorSpec("gemini-3.6-flash", "gemini/gemini-3.6-flash", "litellm", "gemini",
                  Access.BLACK_BOX, env_var="GEMINI_API_KEY", size_hint="small-frontier",
                  notes="Current fast Gemini (gemini-flash-latest family). Verify the exact "
                        "id against ai.google.dev before a real run -- the flash line versions "
                        "quickly."),

    # --- Open models (white-box) -- reference line + targets ----------------
    GeneratorSpec("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.TARGET,), size_hint="8B",
                  notes="Open target we host -> enables the white-box reference line."),
    GeneratorSpec("llama-3.3-70b", "meta-llama/Llama-3.3-70B-Instruct", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.TARGET,), size_hint="70B",
                  notes="Large open target; the open-model stand-in for a frontier target."),
    GeneratorSpec("qwen3-8b", "Qwen/Qwen3-8B", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.TARGET, Role.PROXY), is_reasoning=True,
                  reasoning_trace=ReasoningTrace.ANSWER_ONLY, size_hint="8B",
                  notes="Qwen3 with a thinking mode; open LRM for reasoning-trace experiments."),
    GeneratorSpec("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.TARGET, Role.PROXY), size_hint="7B",
                  notes="Open target and a candidate proxy substrate."),
    GeneratorSpec("deepseek-r1-distill-qwen-7b",
                  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.TARGET,), is_reasoning=True,
                  reasoning_trace=ReasoningTrace.INCLUDE_TRACE, size_hint="7B",
                  notes="Open reasoning model; the open-LRM case where the trace is scorable."),

    # --- Proxy substrate (small open) -- the P family's model slot ----------
    # Matches DisAAD's own proxy scale (llama3.2-3b / llama3.2-1b in their scripts).
    GeneratorSpec("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.PROXY, Role.TARGET), size_hint="3B",
                  notes="Small proxy substrate for DisAAD / the reference proxy (P family). "
                        "Same scale as DisAAD's distilled proxy."),
    GeneratorSpec("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", "huggingface", "open",
                  Access.WHITE_BOX, roles=(Role.PROXY,), size_hint="1B",
                  notes="Smallest proxy substrate; the cheapest P-family option."),
]

GENERATORS = {spec.name: spec for spec in _SPECS}


def get_generator(name: str) -> GeneratorSpec:
    """Look up a generator spec by friendly name, with a helpful error on a miss."""
    if name in GENERATORS:
        return GENERATORS[name]
    # Also accept a raw model id, so a config can name a model not yet in the registry.
    for spec in GENERATORS.values():
        if spec.model_id == name:
            return spec
    raise KeyError(
        f"Unknown generator '{name}'. Known: {sorted(GENERATORS)}. "
        f"Add a GeneratorSpec to hc_benchmark/generators.py, or pass a raw model id "
        f"with an explicit backend in the config."
    )


def generators_for(provider: str = None, access: Access = None, role: Role = None,
                   reasoning: bool = None) -> list:
    """Filter the registry -- used to build provider config sweeps and the proxy slot."""
    out = []
    for spec in GENERATORS.values():
        if provider is not None and spec.provider != provider:
            continue
        if access is not None and spec.access is not access:
            continue
        if role is not None and role not in spec.roles:
            continue
        if reasoning is not None and spec.is_reasoning is not reasoning:
            continue
        out.append(spec)
    return out
