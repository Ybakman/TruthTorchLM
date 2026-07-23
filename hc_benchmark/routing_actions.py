"""Intervention & routing action space (protocol Q6).

The Intervention & Routing Engine is the consumer of everything the benchmark measures: it
maps the combined signals — UQ score (M), the OOD/competence gate, risk flags, and the
grounding result — to *what to do next* with a candidate response. This module defines the
**action space** it selects over; the policy that does the selecting (a rule-based MVP
first, a learned MoE later) is built on top.

The space is **two-level**: three primary decisions, and — because "don't ship this
answer" can be resolved several ways — an `abstain` primary fans out into a **handoff**.

Primary actions:

    APPROVE   deliver the response to the user as-is
    CLARIFY   respond directly asking the user to resolve an ambiguity, then answer (a normal
              conversational turn, not a safety stop)
    ABSTAIN   do not ship this answer; then pick a handoff ↓

Abstain handoffs (how the abstention is resolved):

    RAG                  retrieve grounding context and regenerate, then re-gate
    ESCALATE_LARGER_LLM  hand off to a larger / reasoning model
    ESCALATE_HUMAN       human-in-the-loop review (highest cost)
    (none)               plain safe refusal — decline with no downstream

Every decision carries a **cost tier**, because the protocol frames the router as an
accuracy–safety–cost trade, not a plain classifier: escalating to a larger model or a
human is expensive, so the objective is the *cheapest adequate* decision — approve when
safe, reach for an expensive handoff only when the signals demand it. `is_safe_stop` marks
the decisions that never ship an unreviewed answer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "PrimaryAction", "AbstainHandoff", "RoutingDecision",
    "ABSTAIN_HANDOFFS", "decision_space", "decisions_by_cost",
    "describe_action_space", "print_action_space",
]


class PrimaryAction(str, Enum):
    APPROVE = "approve"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class AbstainHandoff(str, Enum):
    """How an ABSTAIN is resolved. Only valid when the primary action is ABSTAIN."""

    RAG = "rag"                              # retrieve grounding context and regenerate
    ESCALATE_LARGER_LLM = "escalate_larger_llm"
    ESCALATE_HUMAN = "escalate_human"


#: The handoff options offered under ABSTAIN, cheapest → most expensive.
ABSTAIN_HANDOFFS = [
    AbstainHandoff.RAG,
    AbstainHandoff.ESCALATE_LARGER_LLM,
    AbstainHandoff.ESCALATE_HUMAN,
]


@dataclass(frozen=True)
class RoutingDecision:
    """A full routing decision: a primary action, plus a handoff when abstaining."""

    primary: PrimaryAction
    handoff: Optional[AbstainHandoff]   # None unless primary is ABSTAIN (None = plain refusal)
    cost_tier: int                      # 0 = cheapest (approve) … higher = more expensive
    ships_answer: bool                  # does the user get a model answer as a direct result?
    is_safe_stop: bool                  # guarantees no unreviewed answer reaches the user?
    description: str
    typical_trigger: str

    @property
    def label(self) -> str:
        return self.primary.value if self.handoff is None else f"abstain→{self.handoff.value}"

    def __post_init__(self):
        if self.handoff is not None and self.primary is not PrimaryAction.ABSTAIN:
            raise ValueError("A handoff is only valid under the ABSTAIN primary action.")


# The full decision space. cost_tier orders it for the "cheapest adequate decision"
# objective: a rule policy walks from APPROVE outward and stops at the first decision whose
# safety / quality bar is met.
_DECISIONS = [
    RoutingDecision(
        PrimaryAction.APPROVE, None, cost_tier=0, ships_answer=True, is_safe_stop=False,
        description="Deliver the response to the user unchanged.",
        typical_trigger="high UQ confidence + no risk flag + grounded + in-domain (OOD gate pass).",
    ),
    RoutingDecision(
        PrimaryAction.CLARIFY, None, cost_tier=1, ships_answer=True, is_safe_stop=False,
        description="Respond directly to the user asking for clarification, then answer once "
                    "the ambiguity is resolved. A normal conversational turn, not a safety stop.",
        typical_trigger="the model needs clarification before it can answer — ambiguous / "
                        "underspecified query (uncertainty is about intent, not knowledge).",
    ),
    RoutingDecision(
        PrimaryAction.ABSTAIN, None, cost_tier=1, ships_answer=False, is_safe_stop=True,
        description="Plain safe refusal — decline with no downstream handoff.",
        typical_trigger="out-of-domain (OOD gate fail) or a safety boundary with no useful handoff.",
    ),
    RoutingDecision(
        PrimaryAction.ABSTAIN, AbstainHandoff.RAG, cost_tier=2, ships_answer=True,
        is_safe_stop=False,
        description="Retrieve grounding context (RAG) and regenerate, then re-gate.",
        typical_trigger="unsupported / possibly-hallucinated factual claim while a relevant KB exists.",
    ),
    RoutingDecision(
        PrimaryAction.ABSTAIN, AbstainHandoff.ESCALATE_LARGER_LLM, cost_tier=3,
        ships_answer=True, is_safe_stop=False,
        description="Route to a larger / reasoning model, then re-gate its answer.",
        typical_trigger="hard-reasoning failure the base model can't self-correct.",
    ),
    RoutingDecision(
        PrimaryAction.ABSTAIN, AbstainHandoff.ESCALATE_HUMAN, cost_tier=4,
        ships_answer=False, is_safe_stop=True,
        description="Route to a human-in-the-loop reviewer.",
        typical_trigger="safety-critical (crisis, prescriptive medical advice) — highest-cost, safest stop.",
    ),
]


def decision_space() -> list:
    """Every routing decision (primary + handoff combinations)."""
    return list(_DECISIONS)


def decisions_by_cost() -> list:
    """Decisions ordered cheapest → most expensive (the order a policy should prefer)."""
    return sorted(_DECISIONS, key=lambda d: d.cost_tier)


def describe_action_space() -> list:
    """A serializable view of the decision space, for docs / policy config."""
    return [
        {
            "label": d.label,
            "primary": d.primary.value,
            "handoff": d.handoff.value if d.handoff else None,
            "cost_tier": d.cost_tier,
            "ships_answer": d.ships_answer,
            "is_safe_stop": d.is_safe_stop,
            "description": d.description,
            "typical_trigger": d.typical_trigger,
        }
        for d in decisions_by_cost()
    ]


def print_action_space() -> None:
    print("\nIntervention & routing decisions (protocol Q6) — cheapest first\n" + "=" * 70)
    print("Primary actions: approve · clarify · abstain (abstain fans out into a handoff)\n")
    for d in decisions_by_cost():
        tags = []
        if d.is_safe_stop:
            tags.append("safe-stop")
        if not d.ships_answer:
            tags.append("no answer shipped")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  [cost {d.cost_tier}] {d.label:24s}{tag_str}")
        print(f"            {d.description}")
        print(f"            when: {d.typical_trigger}")
    print()


if __name__ == "__main__":
    print_action_space()
