"""Routing action space (protocol Q6) — the two-level decision taxonomy.

Pins the structure the routing engine selects over: three primary actions (approve /
clarify / abstain), abstain's fan-out into handoffs, the cost ordering (cheapest-adequate
objective), and the safety invariants (which decisions never ship an unreviewed answer).
Pure enum/dataclass — loads standalone.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    dotted = "hc_benchmark.routing_actions"
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "hc_benchmark" not in sys.modules:
        pkg = types.ModuleType("hc_benchmark")
        pkg.__path__ = [str(REPO_ROOT / "hc_benchmark")]
        sys.modules["hc_benchmark"] = pkg
    spec = importlib.util.spec_from_file_location(dotted, REPO_ROOT / "hc_benchmark/routing_actions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


r = _load()
PrimaryAction = r.PrimaryAction
AbstainHandoff = r.AbstainHandoff


class TestTaxonomy:
    def test_three_primary_actions(self):
        assert {a.value for a in PrimaryAction} == {"approve", "clarify", "abstain"}

    def test_abstain_handoffs_are_rag_and_the_two_escalations(self):
        assert {h.value for h in AbstainHandoff} == {
            "rag", "escalate_larger_llm", "escalate_human",
        }
        assert r.ABSTAIN_HANDOFFS[0] is AbstainHandoff.RAG  # cheapest handoff first

    def test_only_abstain_carries_a_handoff(self):
        for d in r.decision_space():
            if d.handoff is not None:
                assert d.primary is PrimaryAction.ABSTAIN

    def test_a_handoff_under_a_non_abstain_primary_is_rejected(self):
        with pytest.raises(ValueError, match="ABSTAIN"):
            r.RoutingDecision(PrimaryAction.APPROVE, AbstainHandoff.RAG,
                              cost_tier=0, ships_answer=True, is_safe_stop=False,
                              description="x", typical_trigger="x")

    def test_every_primary_is_represented_and_abstain_has_plain_plus_handoffs(self):
        primaries = [d.primary for d in r.decision_space()]
        assert PrimaryAction.APPROVE in primaries and PrimaryAction.CLARIFY in primaries
        abstains = [d for d in r.decision_space() if d.primary is PrimaryAction.ABSTAIN]
        # plain abstain (handoff None) + one per handoff
        assert sum(1 for d in abstains if d.handoff is None) == 1
        assert {d.handoff for d in abstains if d.handoff} == set(AbstainHandoff)

    def test_labels(self):
        labels = {d.label for d in r.decision_space()}
        assert "approve" in labels and "clarify" in labels and "abstain" in labels
        assert "abstain→escalate_human" in labels and "abstain→rag" in labels


class TestCostOrdering:
    def test_approve_cheapest_human_most_expensive(self):
        ordered = r.decisions_by_cost()
        assert ordered[0].primary is PrimaryAction.APPROVE
        assert ordered[-1].handoff is AbstainHandoff.ESCALATE_HUMAN

    def test_cost_monotonic_non_decreasing(self):
        tiers = [d.cost_tier for d in r.decisions_by_cost()]
        assert tiers == sorted(tiers)

    def test_handoffs_cost_more_than_in_place_primaries(self):
        in_place = max(d.cost_tier for d in r.decision_space() if d.handoff is None)
        handoffs = min(d.cost_tier for d in r.decision_space() if d.handoff is not None)
        assert handoffs > in_place

    def test_human_costs_more_than_larger_llm_costs_more_than_rag(self):
        by = {d.handoff: d.cost_tier for d in r.decision_space() if d.handoff}
        assert by[AbstainHandoff.ESCALATE_HUMAN] > by[AbstainHandoff.ESCALATE_LARGER_LLM]
        assert by[AbstainHandoff.ESCALATE_LARGER_LLM] >= by[AbstainHandoff.RAG]


class TestSafetyInvariants:
    def test_approve_delivers_the_candidate_unchanged(self):
        approve = next(d for d in r.decision_space() if d.primary is PrimaryAction.APPROVE)
        assert approve.ships_answer and not approve.is_safe_stop
        assert "unchanged" in approve.description

    def test_clarify_is_a_direct_response_not_a_safe_stop(self):
        """clarify is a normal conversational turn: the model responds directly asking for
        clarification, then answers -- it is NOT a safety mechanism."""
        clarify = next(d for d in r.decision_space() if d.primary is PrimaryAction.CLARIFY)
        assert clarify.ships_answer is True
        assert clarify.is_safe_stop is False

    def test_only_plain_abstain_and_human_are_safe_stops(self):
        safe = {d.label for d in r.decision_space() if d.is_safe_stop}
        assert safe == {"abstain", "abstain→escalate_human"}

    def test_the_regeneration_handoffs_ship_a_re_gated_answer(self):
        for label in ("abstain→rag", "abstain→escalate_larger_llm"):
            d = next(x for x in r.decision_space() if x.label == label)
            assert d.ships_answer and not d.is_safe_stop  # answer produced, then re-gated


class TestSerialization:
    def test_describe_is_ordered_and_carries_policy_fields(self):
        rows = r.describe_action_space()
        assert rows[0]["label"] == "approve"
        assert rows[-1]["label"] == "abstain→escalate_human"
        for row in rows:
            assert {"label", "primary", "handoff", "cost_tier",
                    "ships_answer", "is_safe_stop", "typical_trigger"} <= set(row)
        # plain primaries carry a null handoff
        assert next(r_ for r_ in rows if r_["label"] == "approve")["handoff"] is None
