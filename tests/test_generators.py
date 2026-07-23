"""The generator registry — the G axis (protocol §2, G table).

Pins the two properties the harness relies on: friendly names resolve to the exact
provider-qualified model strings LiteLLM/HF expect, and the access-level / role / reasoning
metadata the protocol asks for is present and internally consistent. No model is ever
loaded or called here — the registry is inert data.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    dotted = "hc_benchmark.generators"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        import types

        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/generators.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load()
GENERATORS = gen.GENERATORS
Access = gen.Access
Role = gen.Role
ReasoningTrace = gen.ReasoningTrace


class TestModelStringResolution:
    def test_anthropic_friendly_name_resolves_to_litellm_prefix(self):
        """A config saying `claude-opus-4-8` must reach LiteLLM as `anthropic/claude-opus-4-8`."""
        assert GENERATORS["claude-opus-4-8"].model_id == "anthropic/claude-opus-4-8"
        assert GENERATORS["claude-sonnet-5"].model_id == "anthropic/claude-sonnet-5"
        assert GENERATORS["claude-haiku-4-5"].model_id == "anthropic/claude-haiku-4-5"

    def test_gemini_friendly_name_resolves_to_litellm_prefix(self):
        assert GENERATORS["gemini-2.5-pro"].model_id == "gemini/gemini-2.5-pro"
        assert GENERATORS["gemini-3.6-flash"].model_id == "gemini/gemini-3.6-flash"

    def test_openai_ids_are_bare(self):
        # LiteLLM takes OpenAI models without a provider prefix.
        assert GENERATORS["gpt-4o-mini"].model_id == "gpt-4o-mini"

    def test_open_models_use_huggingface_repo_ids(self):
        assert GENERATORS["llama-3.1-8b"].model_id == "meta-llama/Llama-3.1-8B-Instruct"
        assert GENERATORS["llama-3.2-3b"].model_id == "meta-llama/Llama-3.2-3B-Instruct"
        assert all("/" in GENERATORS[n].model_id
                   for n in GENERATORS if GENERATORS[n].backend == "huggingface")

    def test_get_generator_accepts_a_raw_model_id_too(self):
        spec = gen.get_generator("anthropic/claude-opus-4-8")
        assert spec.name == "claude-opus-4-8"

    def test_unknown_generator_raises_with_guidance(self):
        with pytest.raises(KeyError, match="Unknown generator"):
            gen.get_generator("gpt-9-ultra")


class TestAccessAndRoleInvariants:
    def test_closed_apis_are_black_box_and_open_models_are_white_box(self):
        for spec in GENERATORS.values():
            if spec.backend == "litellm":
                assert spec.access is Access.BLACK_BOX, spec.name
            elif spec.backend == "huggingface":
                assert spec.access is Access.WHITE_BOX, spec.name

    def test_only_open_models_can_be_proxies(self):
        """The P family routes through a small open model, never a closed API."""
        for spec in GENERATORS.values():
            if Role.PROXY in spec.roles:
                assert spec.access is Access.WHITE_BOX, spec.name

    def test_there_is_a_small_proxy_substrate(self):
        """DisAAD and the reference proxy need a small open model to route through."""
        proxies = gen.generators_for(role=Role.PROXY)
        assert proxies
        assert any(s.size_hint in ("1B", "3B") for s in proxies)

    def test_reasoning_flag_and_trace_policy_agree(self):
        for spec in GENERATORS.values():
            if spec.is_reasoning:
                assert spec.reasoning_trace in (ReasoningTrace.ANSWER_ONLY,
                                                ReasoningTrace.INCLUDE_TRACE), spec.name
            else:
                assert spec.reasoning_trace is ReasoningTrace.NONE, spec.name

    def test_api_specs_declare_their_key_env_var(self):
        for spec in GENERATORS.values():
            if spec.backend == "litellm":
                assert spec.env_var, spec.name


class TestFilters:
    def test_generators_for_provider(self):
        anthropic = gen.generators_for(provider="anthropic")
        assert {s.name for s in anthropic} == {
            "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"
        }

    def test_generators_for_black_box_targets(self):
        bb = gen.generators_for(access=Access.BLACK_BOX)
        providers = {s.provider for s in bb}
        assert providers == {"openai", "anthropic", "gemini"}

    def test_helpers_on_the_spec(self):
        assert GENERATORS["claude-opus-4-8"].is_black_box() is True
        assert GENERATORS["llama-3.2-1b"].can_be_proxy() is True
        assert GENERATORS["gpt-4o"].can_be_proxy() is False
