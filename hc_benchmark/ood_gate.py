"""OOD / density gate — the pre-generation input gate (protocol §1, "passes: 0").

Official source: **Triantafyllopoulos, Qu, Giorgi, Curtis, Ungar & Sedoc, "Knowing When
Not to Answer: Lightweight KB-Aligned OOD Detection for Safe RAG", ACL 2026
([arXiv:2508.02296](https://arxiv.org/abs/2508.02296))**, `toastedqu/rag_safety_pca` (pinned
at `third_party/rag_safety_pca`, no license — used as an unmodified submodule; this is a
faithful re-port of its PCA + neighbor-classifier core).

Unlike every other method in this benchmark, this is **not** a `TruthMethod`: it scores the
*input query* before generation, not a generation. It answers "is this query inside the
coaching knowledge base's domain?", so an out-of-domain query can be refused or escalated
*before* the model is ever called. In the protocol it is measured separately (a
pre-generation gate, one-time per-KB fit) and is the natural input to the Q6 routing
engine's OOD/competence gate.

How it works, faithful to the source:
1. **Fit** PCA on the in-domain KB document embeddings (StandardScaler -> PCA), keeping the
   top ``n_components`` KB-aligned directions.
2. **Project** a query embedding into that subspace.
3. **Classify** in-/out-of-domain by a neighbor test against the projected KB points:
   * **E-ball** — a KB point within Euclidean ``radius`` (sklearn ``RadiusNeighborsClassifier``
     with ``outlier_label=0``);
   * **E-cube** — a KB point inside the axis-aligned box of half-width ``radius`` per
     dimension (the repo's own ``EpsilonCubeNeighborsClassifier``, ported below).
   No KB neighbor in range -> label 0 = out-of-domain.

The embedder is **injected** (``embed_fn``) so this is testable without sentence-transformers
and works with whatever encoder the deployment uses; the paper's main encoder is MPNet.
Pure input-side; no target model is involved.
"""

from typing import Callable, Optional

import numpy as np

__all__ = ["EpsilonCubeNeighborsClassifier", "PCAGate"]


class EpsilonCubeNeighborsClassifier:
    """Axis-aligned box neighbor classifier — faithful port of `model/ecn_classifer.py`.

    A test point is assigned the majority label of the training points whose every
    coordinate lies within ``sides`` of it (an L-infinity / box neighborhood); if the box
    is empty it gets ``outlier_label``.
    """

    def __init__(self, sides, outlier_label) -> None:
        self.sides = np.abs(np.array(sides, dtype=float).reshape(1, -1))  # (1, n_dims)
        self.n_dims = self.sides.shape[1]
        self.outlier_label = int(outlier_label)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        assert len(X) == len(y)
        assert X.shape[1] == self.n_dims
        self.X = X
        self.y = y
        return self

    def _majority_or_outlier(self, inds):
        if inds.size == 0:
            return self.outlier_label
        vals, counts = np.unique(self.y[inds], return_counts=True)
        return vals[int(np.argmax(counts))]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.shape[1] == self.n_dims
        lows = X - self.sides       # (n_test, n_dims)
        highs = X + self.sides
        preds = []
        for i in range(len(X)):
            inside = np.all((self.X >= lows[i]) & (self.X <= highs[i]), axis=1)
            preds.append(self._majority_or_outlier(np.where(inside)[0]))
        return preds


class PCAGate:
    """KB-aligned PCA out-of-domain gate (Triantafyllopoulos et al., ACL 2026).

    ``embed_fn(list_of_texts) -> np.ndarray`` supplies embeddings. Fit once on the KB with
    :meth:`fit`; then :meth:`is_in_domain` / :meth:`predict` gate incoming queries.
    """

    IN_DOMAIN = 1
    OUT_OF_DOMAIN = 0

    # Not a trained model, but it does need a one-time per-KB fit before use. Surfaced so
    # the readiness report treats an unfitted gate the same way it treats an untrained
    # proxy: "prepare this before you run".
    REQUIRES_FIT = True

    def __init__(
        self,
        embed_fn: Callable,
        method: str = "eball",
        radius: float = 0.01,
        n_components: Optional[int] = None,
        scale: bool = True,
    ):
        if method not in ("eball", "ecube"):
            raise ValueError(f"method must be 'eball' or 'ecube', got '{method}'.")
        self.embed_fn = embed_fn
        self.method = method
        self.radius = radius
        self.n_components = n_components
        self.scale = scale
        self._scaler = None
        self._pca = None
        self._clf = None

    # -- fit on the knowledge base -------------------------------------------------------

    def fit(self, kb_documents):
        """Fit the KB-aligned PCA subspace and the in-domain neighbor classifier."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        embeddings = np.asarray(self.embed_fn(list(kb_documents)), dtype=float)
        if self.scale:
            self._scaler = StandardScaler().fit(embeddings)
            embeddings = self._scaler.transform(embeddings)
        self._pca = PCA(n_components=self.n_components).fit(embeddings)

        kb_proj = self._pca.transform(embeddings)          # KB points in the subspace
        labels = np.full(len(kb_proj), self.IN_DOMAIN)     # every KB point is in-domain

        if self.method == "eball":
            from sklearn.neighbors import RadiusNeighborsClassifier

            self._clf = RadiusNeighborsClassifier(
                radius=self.radius, outlier_label=self.OUT_OF_DOMAIN
            ).fit(kb_proj, labels)
        else:  # ecube
            n_dims = kb_proj.shape[1]
            self._clf = EpsilonCubeNeighborsClassifier(
                sides=[self.radius] * n_dims, outlier_label=self.OUT_OF_DOMAIN
            ).fit(kb_proj, labels)
        return self

    # -- gate incoming queries -----------------------------------------------------------

    def _project(self, queries):
        embeddings = np.asarray(self.embed_fn(list(queries)), dtype=float)
        if self._scaler is not None:
            embeddings = self._scaler.transform(embeddings)
        return self._pca.transform(embeddings)

    def predict(self, queries):
        """Return IN_DOMAIN (1) / OUT_OF_DOMAIN (0) for each query."""
        if self._clf is None:
            raise RuntimeError("PCAGate is not fitted. Call fit(kb_documents) first.")
        proj = self._project(queries)
        return list(self._clf.predict(proj))

    def is_in_domain(self, query: str) -> bool:
        """True if a single query falls inside the KB's domain."""
        return bool(self.predict([query])[0] == self.IN_DOMAIN)

    def is_ready(self) -> bool:
        """True once :meth:`fit` has been run on a KB — checked before gating queries."""
        return self._clf is not None
