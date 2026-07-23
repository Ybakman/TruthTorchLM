# Third-Party Notices

`TruthTorchLM-HC` is a research fork of [`Ybakman/TruthTorchLM`](https://github.com/Ybakman/TruthTorchLM)
(MIT). Upstream copyright and the `LICENSE` file are retained unmodified at the repository root.

## Policy

Every uncertainty-quantification method reproduced in this repository must come from the
**official implementation released by the paper's authors**, or from an authoritative source.
Where no official source exists, we either do not build the method, or we build it and label it
explicitly as **"novel — this work"**. We never present our own construction as a reproduction.

Reference implementations are pinned as **git submodules under `third_party/`**. They are:

- **never copied** into `src/` or otherwise redistributed;
- **never modified** in place;
- accessed only through thin adapter classes in `src/TruthTorchLM/` that cite the source repo and
  commit SHA in their docstring.

This matters legally as well as scientifically: two of the pinned repositories declare no license
at all, which under default copyright means *all rights reserved*. Vendoring their code into an
MIT-licensed package would be a license violation. Using them as unmodified, separately-cloned
submodules is not redistribution.

Fetch them with:

```bash
git submodule update --init --recursive --depth 1
```

## Pinned reference implementations

| Directory | Upstream | Commit | License | Paper | Used for |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `third_party/semantic_uncertainty` | [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) | `a8d9aa8` | BSD-3-Clause-Clear | Farquhar, Kossen, Kuhn & Gal, *Detecting hallucinations in large language models using semantic entropy*, **Nature** 2024 | **Discrete Semantic Entropy (DSE)** — the text-only `cluster_assignment_entropy` variant. **Ported** → `src/TruthTorchLM/truth_methods/discrete_semantic_entropy.py` (entropy aggregation is a faithful port of `cluster_assignment_entropy`; clustering reuses TTLM's equivalent bidirectional-entailment rule). |
| `third_party/UQ-NLG` | [zlin7/UQ-NLG](https://github.com/zlin7/UQ-NLG) | `ecaea917` | MIT | Lin, Trivedi & Sun, *Generating with Confidence: Uncertainty Quantification for Black-box LLMs*, **TMLR** ([arXiv:2305.19187](https://arxiv.org/abs/2305.19187)) | **Lexical Similarity (LeS)** — **ported** → `truth_methods/lexical_similarity.py` (faithful port of `_compute_lexical_sim`, mean pairwise ROUGE-L). **EigV** — **verified alias** → `truth_methods/eigv.py`: upstream TTLM's `SumEigenUncertainty`/`calculate_U_eigv` is identical to UQ-NLG's `(1 - eig(L)).clip(0).sum()` (same symmetric normalized Laplacian, same aggregation); `tests/test_eigv_parity.py` pins the equality. |
| `third_party/DisAAD` | [huizi-Cui/DisAAD](https://github.com/huizi-Cui/DisAAD) | `baf5adb6` | **NONE DECLARED** — see note below | Cui, Ma, Wang, Gao & Zhang, *Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation*, **ACL 2026** ([2026.acl-long.1979](https://aclanthology.org/2026.acl-long.1979/), [arXiv:2605.05777](https://arxiv.org/abs/2605.05777)) | **DisAAD** — **setup only, no code copied.** Stage-2 inference adapter → `truth_methods/disaad.py` (proxy-based evidential scoring; evidential AU/EU re-implemented from the standard EDL formulation). Stage-1 teacher→student proxy training → `hc_benchmark/disaad_train.py`, which **shells out to the official scripts** (`data_builder.py`, `train_disaad.py`) in this unmodified submodule — runnable on the GPU server, no proxy trained locally. |
| `third_party/IUQ` | [louisfanhz/IUQ](https://github.com/louisfanhz/IUQ) | `95c8928a` | MIT | Fan, Duan & Xu, *IUQ: Interrogative Uncertainty Quantification for Long-Form LLM Generation*, **ACL 2026** ([2026.acl-long.606](https://aclanthology.org/2026.acl-long.606/)) | **IUQ** — **ported** → `long_form_generation/iuq.py` (prompts vendored verbatim; `supportness × impact` with exp-decay error propagation ported from `schemas.py`/`main.py`). A *long-form, generation-level* method — supportness needs the N generations and impact needs the ordered claim sequence, so it does not fit the per-claim `ClaimCheckMethod`; it lives as its own pipeline. |
| `third_party/belief-neighborhood-consistency` | [zjunlp/belief](https://github.com/zjunlp/belief) | `ccabadef` | **No `LICENSE` file** (README carries an MIT badge) | Xu, Zhao, Yao, et al., *Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency*, **ACL 2026** ([2026.acl-long.203](https://aclanthology.org/2026.acl-long.203/)) | **NeighborhoodConsistency (NCB)** — **ported** → `truth_methods/neighbor_consistency_belief.py`. Faithful port of `calc_belief_score.py::calculate_belief_metrics`: `NCB = P(y) × aggregate(neighbor p_i)`, three aggregations, validity gate. The UQ **measure is inference-only** (the SAT fine-tuning in the repo is a separate intervention, not the signal). Requires a neighbor-question dataset. |
| `third_party/SPUQ` | [intuit-ai-research/SPUQ](https://github.com/intuit-ai-research/SPUQ) | `55d27161` | Apache-2.0 | Gao, Zhang, Mouatadid & Das, *SPUQ: Perturbation-Based Uncertainty Quantification for LLMs*, **EACL 2024** ([arXiv:2403.02509](https://arxiv.org/abs/2403.02509)) | **SPUQ** — **ported** → `truth_methods/spuq.py` (perturbation classes + input-weighted inter-sample aggregation; perturbs the prompt and makes its own target calls, so it is not served from the shared cache). Two upstream bugs fixed and flagged inline. |
| `third_party/rag_safety_pca` | [toastedqu/rag_safety_pca](https://github.com/toastedqu/rag_safety_pca) | `9d5a393c` | **NONE DECLARED** | Triantafyllopoulos, Qu, Giorgi, Curtis, Ungar & Sedoc, *Knowing When Not to Answer: Lightweight KB-Aligned OOD Detection for Safe RAG*, **ACL 2026** ([2026.acl-long.740](https://aclanthology.org/2026.acl-long.740/)) | **OOD / density gate** — **ported** → `hc_benchmark/ood_gate.py`. KB-aligned PCA + E-ball/E-cube neighbor test (the E-cube classifier re-ported from `model/ecn_classifer.py`). A pre-generation *input* gate (not a `TruthMethod`); one-time per-KB fit; feeds the Q6 routing engine. The unlicensed source is used only as an unmodified submodule; this is a clean re-port. |

### Note on unlicensed repositories

`third_party/DisAAD` and `third_party/rag_safety_pca` (and, strictly, `third_party/belief-neighborhood-consistency`,
whose MIT claim appears only as a README badge with no `LICENSE` file) declare no license.

**TODO — outreach:** email the corresponding authors requesting an explicit open-source license,
and record the outcome here. Until then these remain unmodified submodules only. If permission is
declined, the affected methods will either be re-implemented clean-room from the published paper
— and re-labelled as such, no longer "the official implementation" — or dropped.

## Excluded for lack of an official implementation

| Item | Status | Decision |
| :--- | :--- | :--- |
| **AGSC** — Luo et al., *Adaptive Granularity and Semantic Clustering for UQ in Long-text Generation*, ACL 2026 ([2026.acl-long.434](https://aclanthology.org/2026.acl-long.434/)) | Paper exists; **no code repository exists** (no link in the anthology page, arXiv abstract, or PDF; GitHub search returns nothing) | **Dropped** under the policy above. Revisit if the authors release code. |

## Novel components (this work — not reproductions)

These have no prior paper or reference implementation and are original contributions of this
project. They are labelled `novel — this work` in code and documentation, and must never be cited
or presented as reproductions:

- **Training-free reference proxy** (research question Q3b) — the zero-per-target-cost control arm
  against which DisAAD's per-target distillation is measured.
- **Chain-of-interaction semantic entropy** (Q4) — a multi-turn dialogue UQ construct. Verified as
  of July 2026 to have no prior publication or implementation under this or any equivalent name.
- **Med-Stress-style belief-collapse protocol** — re-implemented from the procedure described in
  *When Correct Beliefs Collapse: Epistemic Resilience of LLMs under Clinical Pressure*
  ([arXiv:2605.23932](https://arxiv.org/abs/2605.23932)), which released **no code or data**. Our
  implementation is an independent re-creation of the described protocol, not their artifact.

## Models and datasets

Model weights and datasets are downloaded at runtime from their canonical hosts and are **not**
redistributed here. Their individual licenses and terms of use apply — notably **BioASQ**, which
requires registration at [bioasq.org](http://bioasq.org) and is not distributable through this
repository. See `hc_benchmark/configs/` and `src/TruthTorchLM/utils/dataset_utils.py` for the
exact identifiers used.
