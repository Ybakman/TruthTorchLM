"""EigV — the graph-Laplacian eigenvalue estimate of the number of semantic meanings.

Official source: **Lin, Trivedi & Sun, "Generating with Confidence", TMLR
([arXiv:2305.19187](https://arxiv.org/abs/2305.19187))**, `zlin7/UQ-NLG` (pinned at
`third_party/UQ-NLG`).

**This is not a re-implementation — it is a verified alias.** Upstream TTLM already ships
this method as :class:`SumEigenUncertainty` (`calculate_U_eigv`). We checked it against the
source line by line and it is identical:

* **Laplacian.** TTLM `get_L_mat` computes ``inv(sqrt(D)) @ (D - W) @ inv(sqrt(D))`` —
  byte-for-byte the same symmetric normalized Laplacian as UQ-NLG's `get_L_mat`.
* **Aggregation.** TTLM: ``sum(max(0, 1 - eig) for eig in eigvalsh(L))``. UQ-NLG's
  ``SpectralClusteringFromLogits.__call__`` with the default ``adjust=True``:
  ``(1 - get_eig(L)[0]).clip(0).sum()``. These are the same function.

The only free choice is how the affinity matrix ``W`` is built from the pairwise NLI
signal; both use the entailment model's agree/disagree judgement, symmetrized. So rather
than duplicate the code (and risk drift), `EigV` is a thin, explicitly-named subclass so
the benchmark's method table can say "EigV" while pointing at the audited upstream
implementation. ``tests/test_eigv_parity.py`` pins the aggregation equality so a future
upstream edit that breaks parity fails loudly.

Higher EigV = more distinct semantic meanings among the samples = more uncertain; the
truth value is its negation, inherited from :class:`SumEigenUncertainty`.
"""

from .sum_eigen_uncertainty import SumEigenUncertainty

__all__ = ["EigV"]


class EigV(SumEigenUncertainty):
    """Graph-eigenvalue consistency (Lin et al. 2023). Verified identical to
    :class:`SumEigenUncertainty`; named for the benchmark's method table. Pure black-box."""

    # Inherits REQUIRES_SAMPLED_TEXT, __init__, forward_* unchanged. Existing as a distinct
    # class is intentional: it documents the provenance and lets the shortlist name it
    # "EigV" without pretending it is separate code.
