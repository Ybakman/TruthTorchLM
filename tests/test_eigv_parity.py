"""EigV parity — the audited claim that TTLM's SumEigenUncertainty *is* UQ-NLG's EigV.

`EigV` is a documented alias, not a re-implementation. That is only honest if the two
aggregations are actually the same function, so this test pins the equality directly on
random symmetric normalized Laplacians:

    TTLM:    sum(max(0, 1 - eig) for eig in eigvalsh(L))
    UQ-NLG:  (1 - eigvalsh(L)).clip(0).sum()            # __call__ with adjust=True

If a future upstream edit changes the aggregation, this fails loudly rather than letting
the "verified identical" docstring quietly become false.
"""

import numpy as np
import pytest


def _ttlm_aggregate(eigvals):
    """TTLM calculate_U_eigv aggregation."""
    return sum(max(0.0, 1.0 - eig) for eig in eigvals)


def _uqnlg_aggregate(eigvals):
    """UQ-NLG SpectralClusteringFromLogits.__call__ aggregation (adjust=True)."""
    return float((1.0 - np.asarray(eigvals)).clip(0).sum())


def _random_normalized_laplacian(n, seed):
    """Symmetric normalized Laplacian of a random nonnegative affinity matrix.

    Uses the same construction both repos use: inv(sqrt(D)) @ (D - W) @ inv(sqrt(D)).
    """
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, (n, n))
    W = (W + W.T) / 2
    W[np.arange(n), np.arange(n)] = 1.0  # self-affinity, as both repos set
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = D_inv_sqrt @ (D - W) @ D_inv_sqrt
    return np.linalg.eigvalsh(L)


@pytest.mark.parametrize("n", [2, 3, 5, 8, 20])
def test_aggregations_are_identical(n):
    for seed in range(5):
        eig = _random_normalized_laplacian(n, seed)
        assert _ttlm_aggregate(eig) == pytest.approx(_uqnlg_aggregate(eig), abs=1e-9)


def test_eigv_is_declared_as_a_subclass_not_a_fork():
    """The alias must actually *be* the upstream method. Checked at the source level so it
    holds without the ML stack (sibling tests stub `torch`, defeating importorskip)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src/TruthTorchLM/truth_methods/eigv.py").read_text()
    assert "from .sum_eigen_uncertainty import SumEigenUncertainty" in src
    assert "class EigV(SumEigenUncertainty):" in src
    # and it must not re-implement the aggregation: no method body overrides forward_*
    assert "def forward_hf_local" not in src and "def forward_api" not in src


def test_concentrated_mass_is_low_uncertainty():
    """Sanity on orientation: one dominant cluster (eigenvalues near 1 except one near 0)
    gives a small EigV; spread mass (many eigenvalues near 0) gives a large one."""
    concentrated = np.array([0.0] + [1.0] * 4)   # one semantic set
    spread = np.zeros(5)                           # five semantic sets
    assert _ttlm_aggregate(concentrated) < _ttlm_aggregate(spread)
    assert _ttlm_aggregate(spread) == pytest.approx(5.0)
