"""Method readiness — tell users, up front, which methods need a prep step and whether it's done.

Most UQ methods in this benchmark are inference-only: construct and run. Two are not, and
the failure mode without this module is discovering that *at score time*, via an exception:

* **DisAAD** needs a distilled **proxy** (an offline teacher->student training step).
* **The OOD-PCA gate** needs a one-time **KB fit** (`.fit(kb_documents)`).

This module gives a single answer to "what can I run right now, and what do I need to
prepare first?". A component is *ready* when its artifact exists — for DisAAD, a training
manifest next to the proxy (written automatically when training completes); for the gate,
a completed fit. Point the report at a proxy path and it will flip from
"needs training" to "ready" the moment the cluster job finishes — no guessing.

    from hc_benchmark.readiness import readiness_report, print_readiness
    print_readiness(proxy_path="hc_benchmark/disaad/proxy")

or from the shell:

    python -m hc_benchmark.readiness --proxy-path hc_benchmark/disaad/proxy
"""

from dataclasses import dataclass

__all__ = ["MethodReadiness", "readiness_report", "print_readiness"]


@dataclass
class MethodReadiness:
    name: str
    kind: str            # "inference" | "needs-training" | "needs-fit"
    ready: bool
    detail: str          # what state it's in
    how_to_prepare: str  # empty if nothing to do


# The inference-only methods: ready as soon as they're constructed (a model download on
# first use is not a prep step the user has to manage).
_INFERENCE_ONLY = [
    "VerbalizedConfidence",
    "DiscreteSemanticEntropy",
    "LexicalSimilarity",
    "EigV",
    "NumSemanticSetUncertainty",
    "SPUQ",
    "IUQ",
    "NeighborhoodConsistencyBelief",
]


def readiness_report(proxy_path: str = None, ood_gate=None):
    """Structured readiness for every benchmark component.

    ``proxy_path`` — a DisAAD proxy directory to check (its manifest flips the status to
    ready). ``ood_gate`` — an optional live :class:`PCAGate` to report fitted/unfitted.
    """
    rows = [
        MethodReadiness(name, "inference", True, "inference-only — ready to run", "")
        for name in _INFERENCE_ONLY
    ]

    # DisAAD -- needs a trained proxy.
    from hc_benchmark.disaad_train import is_proxy_trained, read_training_manifest

    if proxy_path and is_proxy_trained(proxy_path):
        m = read_training_manifest(proxy_path)
        rows.append(MethodReadiness(
            "DisAAD", "needs-training", True,
            f"proxy ready at {proxy_path} (teacher={m.get('teacher_model')}, "
            f"student={m.get('student_model')}, trained_at={m.get('trained_at')})",
            "",
        ))
    else:
        where = f" at '{proxy_path}'" if proxy_path else ""
        rows.append(MethodReadiness(
            "DisAAD", "needs-training", False,
            f"no trained proxy{where} — cannot score yet",
            "Train a proxy on the GPU server via hc_benchmark/disaad_train.train_proxy(...), "
            "then DisAAD.from_pretrained(proxy_path). See hc_benchmark/configs/disaad.yaml.",
        ))

    # OOD-PCA gate -- needs a per-KB fit.
    if ood_gate is not None and getattr(ood_gate, "is_ready", lambda: False)():
        rows.append(MethodReadiness("OOD-PCA gate", "needs-fit", True,
                                    "fitted on a KB — ready to gate queries", ""))
    else:
        rows.append(MethodReadiness(
            "OOD-PCA gate", "needs-fit", False,
            "not fitted — cannot gate queries yet",
            "Call PCAGate(embed_fn=...).fit(kb_documents) once on your knowledge base.",
        ))
    return rows


def print_readiness(proxy_path: str = None, ood_gate=None) -> None:
    """Human-readable readiness table."""
    rows = readiness_report(proxy_path=proxy_path, ood_gate=ood_gate)
    ready = [r for r in rows if r.ready]
    not_ready = [r for r in rows if not r.ready]

    print(f"\nMethod readiness — {len(ready)}/{len(rows)} ready to run\n" + "=" * 60)
    print("\nREADY NOW:")
    for r in ready:
        print(f"  ✓ {r.name:32s} {r.detail}")
    if not_ready:
        print("\nNEEDS A PREP STEP FIRST:")
        for r in not_ready:
            print(f"  ✗ {r.name:32s} {r.detail}")
            print(f"      → {r.how_to_prepare}")
    else:
        print("\nAll components are ready.")
    print()


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Report which UQ methods are ready to run.")
    parser.add_argument("--proxy-path", default=None,
                        help="DisAAD proxy directory to check for a training manifest.")
    args = parser.parse_args()
    print_readiness(proxy_path=args.proxy_path)


if __name__ == "__main__":
    _main()
