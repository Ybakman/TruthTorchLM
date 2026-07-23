# Pure Black-Box UQ Benchmark — Health-Coaching Uncertainty Agent

**Purpose.** Answer, empirically and for our setting, the one question the literature can't:
under a *text-output-only* constraint, how do candidate UQ methods trade off **accuracy**
(can the score separate reliable from unreliable responses?) against **latency**
(what does it cost, in user-facing wait time, to produce that score?).

**Scope constraint.** Methods must run on the target's **generated text only** — no weights,
no hidden states, no token log-probabilities. This excludes the log-prob/entropy family,
internal-state probes, and the logprob variant of Semantic Entropy. It is the regime an
inline guardrail wrapping an external coaching model actually lives in.

> **Status:** this is the repository's spec of record. It supersedes the original working draft.
> All citations were verified live against ACL Anthology, arXiv, GitHub, and the HuggingFace Hub
> in **July 2026**; see [§0 Corrections log](#0-corrections-log) for what changed and why, and
> [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for pinned sources and licenses.

---

## 0. Corrections log

The working draft contained five citation errors that would have sent us to build the wrong
thing. They are corrected throughout this document; recorded here so the change is auditable.

| Draft claim | Verified reality | Resolution |
| :--- | :--- | :--- |
| `2026.acl-long.1979` is **both** DisAAD and Discrete Semantic Entropy | That ID is **DisAAD only** — Cui, Ma, Wang, Gao & Zhang, *Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation*, ACL 2026 ([arXiv:2605.05777](https://arxiv.org/abs/2605.05777)). | DSE recited to **Farquhar, Kossen, Kuhn & Gal, *Nature* 2024**; official code [`jlko/semantic_uncertainty`](https://github.com/jlko/semantic_uncertainty) (BSD-3-Clause-Clear), the `cluster_assignment_entropy` variant. |
| `2026.acl-long.203` = "NCB" | *Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency* (Xu et al.). The acronym "NCB" appears nowhere. | Renamed **NeighborhoodConsistency**; official code [`zjunlp/belief`](https://github.com/zjunlp/belief). |
| `2026.acl-long.740` = generic "OOD / density gate" | *Knowing When Not to Answer: Lightweight KB-Aligned OOD Detection for Safe RAG* (Triantafyllopoulos et al.); PCA-based, **KB-aligned** — it needs a knowledge base to fit against, which is a real precondition for us. | Kept, scope clarified; code [`toastedqu/rag_safety_pca`](https://github.com/toastedqu/rag_safety_pca) (**no license declared**). |
| AGSC (`2026.acl-long.434`) is a candidate refinement | Paper exists; **no code repository exists anywhere** — no link in the anthology page, arXiv abstract, or PDF; GitHub search returns nothing. | **Dropped** under the provenance policy. Revisit if the authors release code. |
| Chain-of-interaction SE and MED-STRESS are existing constructs to adopt | **No paper and no repository** exist for chain-of-interaction semantic entropy. "Med-Stress" is a multi-turn sycophancy **stress-test framework** from *When Correct Beliefs Collapse* ([arXiv:2605.23932](https://arxiv.org/abs/2605.23932)), with **no code or data release**. | Chain-of-interaction SE is reclassified as a **novel contribution of this work** (Q4) and labelled as such everywhere. The belief-collapse protocol is an independent re-implementation of the described procedure, not their artifact. |

One further correction, in our favour: the draft's dataset table understated upstream. TTLM
already ships `trivia_qa`, `gsm8k`, `natural_qa`, `pop_qa`, `simple_qa`, `wikipedia_factual`,
`narrative_qa`, and `web_questions` (`src/TruthTorchLM/availability.py`). Only **health**, **MCQ**,
**extractive**, and **dialogue** are genuinely empty.

---

## Research questions

**Q1 — The accuracy–latency frontier.** Under the pure black-box constraint, what is the
accuracy–latency Pareto frontier across the candidate methods, measured on one common protocol
rather than pooled from incommensurable papers? Latency is reported as measured marginal wall-clock
in milliseconds (p50/p95/p99), decomposed by stage and reported for both serial and concurrent
execution, with a pass/fail against a fixed real-time budget — not merely a single-pass-vs-multi
ordering. Accuracy is reported as discrimination, calibration, and safety-weighted metrics,
separately for single-turn QA and multi-turn dialogue, because the answer differs by regime.

**Q2 — Robustness under distribution shift (proxy vs non-proxy).** Does black-box UQ reliability
hold OOD, and do proxy methods degrade more sharply than non-proxy consistency methods? Fit each
method's components in-domain, evaluate on held-out domains (a different medical subdomain, a
rare-condition slice, an adversarial-paraphrase slice), and test the prediction that the proxy shows
a sharper AUROC drop and a silent calibration collapse once extrapolating, while consistency methods
degrade only in step with the target itself.

**Q3 — Can we build better methods (single-instance and proxy)?** (a) Can a single-instance (one
target call) method reach multi-sample accuracy, or is the single-pass–multi-sample gap fundamental
in text-only settings? (b) Can a proxy close that gap by relocating the iteration off the target
(external-single / internal-multi-turn), and does a **training-free reference proxy** *(novel — this
work)* capture most of DisAAD's benefit at zero per-target cost, making per-target distillation
unnecessary? The training-free proxy is the control arm any fancier design must beat.

**Q4 — Multi-turn dialogue and chain-of-interaction.** These methods were validated on non-dialogue
text, so their behavior on multi-turn dialogue is itself the open challenge. Is **chain-of-interaction
semantic entropy** *(novel — this work; no prior art exists, verified July 2026)* the natively correct
construct for dialogue, where single-turn and parallel-sample methods measure the wrong object —
dispersion over a fixed prompt rather than belief stability under conversational pressure? Tested two
ways: (validity) do the incumbents miss the belief-collapse cases chain-of-interaction catches;
(efficiency) can the interaction chain run internally against a proxy at a single target call without
losing the signal? Treated as a hypothesis, with an explicit guard against probe-induced artifacts
(measuring sycophancy-to-the-probe, not genuine fragility).

**Q5 — How does UQ correlate with downstream labels?** For which downstream outcomes is UQ a valid
signal, and where does it structurally fail? UQ should track factuality-type errors (hallucination,
unsupported claims), but a confident, fluent, factually-correct-but-unsafe response (prescriptive
medical advice, a body-image trigger) carries low uncertainty — so UQ almost certainly does not
predict harm or safety-boundary violations. Measure UQ-to-label correlation **per outcome type**
(`factual_error`, `safety_violation`, and in dialogue, `belief_collapse`) rather than lumped, to
establish which failures UQ can gate and which require the separate Risk-Analysis and Grounding
components. This question is what justifies a multi-component architecture over a single UQ gate.

**Q6 — A learned routing engine (MoE) over intervention actions.** Can we learn a policy mapping the
combined signals — UQ score, risk flags, grounding result, and the OOD/competence gate — to the next
action? The **action space** (defined in `hc_benchmark/routing_actions.py`) is two-level: **three
primary actions** — `approve`, `clarify`, `abstain` — where `abstain` fans out into a **handoff**
(how the abstention is resolved). Cheapest → most expensive:

| Decision | Ships answer? | Safe stop? | When |
| :--- | :---: | :---: | :--- |
| **approve** | ✓ | — | confident + safe + grounded + in-domain → deliver as-is |
| **clarify** | ✓ | — | model needs clarification before it can answer → respond directly asking the user, then answer |
| **abstain** (plain) | — | ✓ | out-of-domain or a safety boundary, no useful handoff → safe refusal |
| abstain → **rag** | ✓ | — | unsupported / hallucinated claim while a KB exists → retrieve grounding + regenerate |
| abstain → **escalate_larger_llm** | ✓ | — | hard-reasoning failure the base model can't self-correct |
| abstain → **escalate_human** | — | ✓ | safety-critical (crisis, prescriptive advice) → human-in-the-loop (highest cost) |

Each decision carries a **cost tier**, because the router solves an accuracy–safety–cost trade, not a
plain classification: escalating to a larger model or a human is expensive, so the objective is the
*cheapest adequate* decision — approve when safe, reach for an expensive handoff only when the signals
demand it. The MoE/learned-gating question is whether that formulation is justified over the rule-based
MVP policy, given the router must jointly optimize safety-recall, over-refusal, and escalation cost.
Open sub-question: the training signal — offline logs, simulated outcomes, or RL on a composite reward.
This is the Intervention & Routing Engine, and its escalation actions make it a consumer of the UQ
signal validated in Q5.

*This protocol operationalizes Q1 and Q2 directly; Q3–Q6 depend on components tracked in the research
plan's answerability map (chain-of-interaction construct, multi-outcome labels, routing engine).*

---

## 1. Method shortlist (survivors of the pure-black-box filter)

The filter is mechanical, not editorial. In code it is exactly:

```python
m.REQUIRES_SAMPLED_LOGPROBS is False and m.REQUIRES_SAMPLED_LOGITS is False \
    and m.REQUIRES_SAMPLED_ATTENTIONS is False and m.REQUIRES_SAMPLED_ACTIVATIONS is False \
    and m.REQUIRES_LOGPROBS is False
```

> **Consequence worth stating loudly:** upstream `SemanticEntropy`
> (`src/TruthTorchLM/truth_methods/semantic_entropy.py`) sets `REQUIRES_SAMPLED_LOGPROBS = True`
> and therefore **fails this filter**. The SC workhorse for our regime must be the *discrete /
> cluster-assignment* variant, which needs sampled text only.

| Method | Passes | Setup | Official source | Role in benchmark |
| :---- | :---: | :---- | :---- | :---- |
| Verbalized confidence (raw) | 1 | None | upstream TTLM | Cheap single-pass floor |
| **DisAAD** (proxy) | 1 (inference) | Per-target training | [`huizi-Cui/DisAAD`](https://github.com/huizi-Cui/DisAAD) · [ACL 2026](https://aclanthology.org/2026.acl-long.1979/) | Single-pass contender w/ offline cost |
| **Discrete Semantic Entropy (DSE)** | N | None | [`jlko/semantic_uncertainty`](https://github.com/jlko/semantic_uncertainty) · Farquhar et al., *Nature* 2024 | Multi-sample consistency workhorse |
| **Lexical Similarity (LeS)** | N | None | [`zlin7/UQ-NLG`](https://github.com/zlin7/UQ-NLG) | Multi-sample, similarity-only |
| **EigV** | N | None | [`zlin7/UQ-NLG`](https://github.com/zlin7/UQ-NLG) · [arXiv:2305.19187](https://arxiv.org/abs/2305.19187) | Multi-sample, graph eigenvalue |
| **IUQ** | N | None | [`louisfanhz/IUQ`](https://github.com/louisfanhz/IUQ) · [ACL 2026](https://aclanthology.org/2026.acl-long.606/) | 2026 refinement |
| **NeighborhoodConsistency** | N | None | [`zjunlp/belief`](https://github.com/zjunlp/belief) · [ACL 2026](https://aclanthology.org/2026.acl-long.203/) | 2026 refinement |
| **SPUQ** | N | None | [`intuit-ai-research/SPUQ`](https://github.com/intuit-ai-research/SPUQ) · EACL 2024 | Perturbation-based sampling |
| **OOD / density gate** | 0 | One-time fit (per-KB) | [`toastedqu/rag_safety_pca`](https://github.com/toastedqu/rag_safety_pca) · [ACL 2026](https://aclanthology.org/2026.acl-long.740/) | Pre-generation input gate; measured separately |
| ~~AGSC~~ | — | — | **none — no code released** | **Dropped** (§0) |

**Novel arms (this work — not reproductions):** the **training-free reference proxy** (Q3b control)
and **chain-of-interaction semantic entropy** (Q4).

**Reference-only (not deployable, for the upper-bound line):** on an *open* target we can also
run a white-box scorer (entropy / LogTokU on the target's own logits). It is not usable in
deployment; it exists only to quantify the black-box gap on the plot.

---

## Implementation substrate — TruthTorchLM (this fork)

The benchmark is built by **expanding** [TruthTorchLM (TTLM)](https://arxiv.org/pdf/2507.08203)
(USC; EMNLP 2025 demo), via the fork [`jsl5710/TruthTorchLM-HC`](https://github.com/jsl5710/TruthTorchLM-HC)
(forked from [`Ybakman/TruthTorchLM`](https://github.com/Ybakman/TruthTorchLM), MIT; package under
`src/TruthTorchLM`).

TTLM's central abstraction is the **`TruthMethod`**: a post-hoc scorer that assigns a scalar *truth
value* to a generation without interfering with decoding, via a standardized `forward`. Long-form is
handled by **`DecompositionMethod`** + **`ClaimCheckMethod`** (claim-level scoring). Entry points:
`generate_with_truth_value`, `evaluate_truth_method`, `calibrate_truth_method`, and
`long_form_generation_with_truth_value`. Backends are **HuggingFace** (open models) and **LiteLLM**
(closed APIs), which maps directly onto our generator set. TTLM already categorizes methods by
document-grounding, supervised/self-supervised, **access level (black/gray/white)**, and
**sampling-required** — the same axes as our taxonomy.

**Three upstream facts that shape the whole build** (verified by reading the code, not the paper):

1. `run_truth_methods` (`src/TruthTorchLM/generation.py`) **already implements §6's shared-generation
   control**: `get_sampling_properties` takes `max(number_of_generations)` across the method list and
   hands one `sampled_generations_dict` to every method. We reuse this rather than rebuild it — but it
   means a naive wall clock around a method call *excludes* sampling cost, which is why §5's latency
   layer splits into three separately-timed stages.
2. `run_over_dataset` (`src/TruthTorchLM/utils/eval_utils.py`) is **already split** into
   `eval_model_over_dataset` (generate + label) and `run_truth_methods_over_dataset` (score) — that
   split maps 1:1 onto §7's Stage A/B and Stage C. We add disk caching between them.
3. `sample_generations_api` (`src/TruthTorchLM/generation.py`) samples **serially**
   (`for i in range(number_of_generations)`). The concurrent arm of §5 does not exist upstream and
   must be built, or the serial-vs-concurrent question cannot be asked at all.

**Provided out-of-the-box (reuse directly):**

| Benchmark need | TTLM provides |
| :---- | :---- |
| SC family methods | SemanticEntropy *(not black-box — logprobs)*, KernelLanguageEntropy, NumSemanticSetUncertainty, Eccentricity/MatrixDegree/SumEigen (Lin et al. spectral → EigV), SelfDetection, CrossExamination, MultiLLMCollab, DirectionalEntailmentGraph |
| VB family | VerbalizedConfidence, PTrue |
| Pob (white/gray-box, for the reference line) | Confidence, Entropy, MARS, LARS, sentSAR/tokenSAR, Inside, AttentionScore, SAPLMA |
| Generators (open + closed) | HuggingFace + LiteLLM (OpenAI directly; Anthropic/Gemini via LiteLLM) |
| Discrimination metrics | AUROC, AUPR (AUPRC), AUARC, PRR (+ F1/acc/precision/recall) |
| Score→probability mapping | Normalizers: IsotonicRegression, min-max, sigmoid, standard |
| Correctness labels | ModelJudge (LLM-as-judge), ROUGE / EM / BLEU |
| Long-form (claim-level) | Decomposition + ClaimCheck + SAFE (Google Search via Serper) |
| Built-in datasets | trivia_qa, gsm8k, natural_qa, pop_qa, simple_qa, wikipedia_factual, narrative_qa, web_questions, longfact (+ custom-dataset support) |

**To build (the expansion), mapped to the research questions:**

| Gap | RQ | Round | Notes |
| :---- | :---: | :---: | :---- |
| Latency instrumentation (ms, marginal, generator×family, serial/concurrent, p50/95/99) | Q1 | 1 | TTLM scores truth values but does **not** time anything — the whole §5 harness is net-new |
| Calibration + safety metrics: ECE, ACE, MCE, Brier; per-stratum + harm-recall | Q1/Q5 | 1 | TTLM has normalizers but not calibration-*error* metrics; safety-weighted metrics are new |
| Stage A–D cached harness with N-sweep | Q1 | 1 | Disk cache between the existing generate/score split |
| Datasets: BioASQ, MedQA, MMLU-med, HotpotQA, TruthfulQA, SQuAD 2.0, K-QA/MedLFQA, health dialogue | Q1/Q2 | 1 | Custom-dataset loaders; MCQ needs its own correctness evaluator |
| OOD splits (fit-on-A / test-on-B, graded shift, degradation curves) | Q2 | 1 | `calibrate`→`evaluate` can fit a normalizer on A and score B; the orchestration is new |
| Method ports: DSE, LeS, EigV, DisAAD, IUQ, NeighborhoodConsistency, SPUQ, OOD gate | Q1/Q3 | 2 | Each via the authors' official implementation, submodule + adapter |
| Multi-outcome labels (safety violation, belief collapse) | Q5 | 2 | TTLM does factual correctness only; new label types + per-outcome correlation |
| Training-free reference proxy *(novel)* | Q3 | 2 | Loads a *separate* proxy model and scores the target's response through it |
| Chain-of-interaction / multi-turn dialogue construct + dialogue eval *(novel)* | Q4 | 2 | TTLM is single-turn; `chat`/`previous_context` carry multi-turn *input* but there is no multi-turn UQ construct or dialogue evaluation |
| Routing engine (MoE / rules) over intervention actions | Q6 | 3 | Out of TTLM scope — consumes the truth value; built on top |

---

## Toolkit update inventory — G · D · M · V

The benchmark is parameterized on four axes, read as a pipeline **G × D → M → V**: a *generator* (G)
on a *dataset* (D) produces outputs, a *method* (M) scores them, and an *evaluation metric* (V) scores
the method. TTLM-name mapping so spec and code don't drift: **G** = HuggingFace/LiteLLM model,
**D** = dataset, **M** = *truth method*, **V** = *eval_metric*. Legend: ✅ present · ➕ to add.

### G — Generators / target models, by sub-category

| Axis · value | ✅ Have (infra + tested) | ➕ Add |
| :---- | :---- | :---- |
| Access · closed API (black-box) | LiteLLM backend; GPT-4o(-mini) tested; **Anthropic (Claude) + Gemini now in the registry** (`hc_benchmark/generators.py`), text-only BB check via `scripts/verify_provider.py` | — |
| Access · open (WB reference + proxy) | HuggingFace backend; Llama-3.1-8B, Llama-3.3-70B, Qwen3-8B, Mistral-7B in the registry | Llama-4 when hosted |
| Type · LLM | ✅ (both backends) | — |
| Type · **LRM (reasoning)** | **registry flags Claude/Gemini thinking + open DeepSeek-R1 / Qwen3; `reasoning_trace` policy per model** | o-series; wire the HF generation path for open LRMs |
| Role · target | ✅ full closed (OpenAI/Anthropic/Gemini) + open target set in the registry | — |
| Role · **proxy (small open)** | **Llama-3.2-1B / 3B, Mistral-7B, Qwen3-8B tagged `PROXY`** (P-family substrate; DisAAD's own proxy scale) | — |

*G summary: both backends already exist, so G is mostly configuration, not code — except two real
builds: **reasoning-trace handling** for LRMs (does the method see the trace or only the answer?) and
the **proxy-model slot** the P family needs.*

### D — Datasets, by sub-category

All HuggingFace identifiers below were verified to resolve (July 2026).

| Axis · value | ✅ Have (built-in) | ➕ Add (verified HF id) |
| :---- | :---- | :---- |
| Area · general | trivia_qa, natural_qa, pop_qa, simple_qa, web_questions, wikipedia_factual, narrative_qa; longfact | `hotpotqa/hotpot_qa`, `truthfulqa/truthful_qa`, `rajpurkar/squad_v2` |
| Area · math | gsm8k | (MATH optional) |
| Area · **health** | ❌ | **BioASQ (`bigbio/bioasq_task_b`, manual download), `GBaker/MedQA-USMLE-4-options`, `cais/mmlu` med configs, `Itaykhealth/K-QA`, `dmis-lab/MedLFQA`, health dialogue** |
| Format · free-form QA | trivia_qa, gsm8k, natural_qa, … | BioASQ, HotpotQA |
| Format · MCQ | ❌ | MMLU-med, MedQA — **needs its own correctness evaluator** |
| Format · extractive/contextual | narrative_qa, wikipedia_factual | SQuAD 2.0 (unanswerable subset = a natural abstention slice) |
| Format · long-form (claim-level) | longfact | K-QA / MedLFQA |
| Hop · multi | gsm8k | HotpotQA (+ MATH) |
| Scope · **dialogue (multi-turn)** | ❌ | **belief-collapse protocol (re-implemented, §0), health dialogue (Q4)** |
| Scope · trajectory / agentic | ❌ | τ-bench (optional) |

*D summary: general is better covered than the draft assumed; **health is empty across every
format**, and MCQ, extractive, and dialogue are missing. TTLM supports custom datasets, so most of
this is loader work — except MCQ, whose constrained-output correctness and UQ behaviour differ from
free-form, and BioASQ, which requires registration at bioasq.org and cannot be redistributed.*

### M — Method (UQ / truth method), by family

Four families: **Pob** (token-probability / logit), **SC** (self-consistency, O(N) sampling),
**VB** (verbalized — self or external judge), and **P** (Proxy / Surrogate — *hybrid*: the
truthfulness judgment is routed through a model's internal signal rather than raw consistency or a
stated number). **P(True)** is the hybrid case that motivates the family — it is verbalized in form
("is this true?") but read as a token probability, a *self-surrogate*; **DisAAD** is the *external*
surrogate (a distilled proxy). Grounded fact-checkers are listed separately — they feed the Grounding
component, not the UQ signal.

| Family | Sub-category | ✅ Have (TTLM) | ➕ Add |
| :---- | :---- | :---- | :---- |
| **Pob** | unsupervised | Confidence, Entropy, sent/tokenSAR | — (gray/white-box; enters BB only via **P**) |
| | supervised scorer | MARS, LARS | — |
| | white-box internal (reference-only) | Inside, AttentionScore, SAPLMA | — (out of BB scope) |
| **SC** | cluster / entropy | SemanticEntropy *(gray — logprobs)*, NumSemanticSetUncertainty, KLE | **Discrete SE** (`jlko/semantic_uncertainty`), **Lexical Similarity** (`zlin7/UQ-NLG`) |
| | graph-spectral (EigV family) | Eccentricity, MatrixDegree, SumEigen, DirectionalEntailment | verify parity against `zlin7/UQ-NLG` |
| | sampling / self-check | SelfDetection | SPUQ, IUQ, NeighborhoodConsistency |
| | **multi-turn** | ❌ | **chain-of-interaction SE — *novel, this work* (Q4)** |
| **VB** | self-verbalized | VerbalizedConfidence | health-tuned prompt (optional) |
| | external judge / multi-LLM | CrossExamination, MultiLLMCollab | — |
| **P (Proxy/Surrogate)** | self-surrogate (prob-read) | **PTrue** | — |
| | distilled proxy (supervised) | ❌ | **DisAAD** (`huizi-Cui/DisAAD`) |
| | training-free reference proxy (unsup) | ❌ | ***novel, this work* — Q3 control** |
| | proxy chain-of-interaction | ❌ | ***novel, this work* (Q4)** |
| *(Grounding — not a UQ family)* | doc-grounded fact-check | GoogleSearchCheck, MiniCheck | → feeds Grounding component |

*M summary: SC and VB are largely covered; Pob is gray/white-box (reference line only); **P is the
biggest build** — only PTrue exists, and DisAAD, the reference proxy, and the proxy
chain-of-interaction all need adding.*

### V — eValuation metric, by function

Full definitions live in §4 (accuracy metric suite) and §5 (latency).

| Function | ✅ Have (TTLM eval_metrics) | ➕ Add |
| :---- | :---- | :---- |
| Discrimination | AUROC, AUPR (AUPRC), AUARC, PRR | — |
| Threshold-based | Accuracy, F1, Precision, Recall | — |
| Calibration | (normalizers only: isotonic, min-max, sigmoid, standard) | **ECE, ACE, MCE, Brier, KDE-ECE, class-wise ECE** |
| Safety-weighted | ❌ | **per-stratum MCE/ECE, harm-recall @ operating point, risk@coverage, coverage@risk** |
| Latency | ❌ | **marginal ms (p50/p95/p99), overhead ratio, SLA pass/fail** |

*V summary: discrimination and threshold metrics are covered; the additions are the calibration-error
family (TTLM ships the normalizers but not the error metrics), the safety-weighted metrics, and the
entire latency layer.*

---

## 2. Factors (cross these; hold everything else fixed)

1. **UQ method** — the shortlist above.
2. **Sample budget N** for multi-sample methods — sweep N ∈ {1, 3, 5, 10, 20}. This is the
   central accuracy–latency dial; single-pass methods are N=1 by construction. Sweeping N is
   what produces the frontier.

---

## 3. Targets, data, and labels *(decision points — team to fix)*

**Target model(s).** Fix a primary and a secondary.
- *Primary:* whatever the conversational agent actually uses.
- *Recommended pairing:* one **open** model we can host (enables the white-box reference line)
  plus one **API** model (GPT-4-class) to confirm the black-box story holds across a true API.

**Datasets.** Aim for three, to separate factuality, ecological validity, and domain shift:
- **BioASQ (factoid)** — biomedical short-answer factuality; closest to clinical.
- **A slice of our real coaching data** — ecologically valid; requires correctness labeling.
- **One general-domain set** (e.g., TriviaQA) — reference point to expose domain sensitivity
  (recall DisAAD's edge largely evaporated off-TriviaQA/TruthfulQA).
- *Optional safety slice:* a medical-abstention / belief-stability set to probe boundary behavior.

**Correctness labels** (define the binary "reliable/unreliable" ground truth the UQ score is
graded against). Use **BLEURT>0.5** *and* **LLM-as-judge** — this matches DisAAD's own
`--thres_gt 0.5` criterion (`third_party/DisAAD/scripts/generate_black.py`), so our numbers are
directly comparable to theirs. For health, give the judge a clinical rubric and spot-validate
against a human. **The correctness criterion materially moves AUROC** (it defines the labels), so
the harness persists *both* labels per item and reports sensitivity across the two criteria rather
than picking one silently.

---

## 4. Accuracy metric suite

Three functional groups — they answer different questions; report all three. If a single headline
row is needed per method, use **PRR** (selective-prediction quality), **Brier** (joint
accuracy+calibration), and **ECE** (pure calibration) — they span discrimination and calibration
without collapsing into one number that hides either.

**A. Discrimination — the primary UQ quality (rank-based, needs no probability mapping).**
- **AUROC** — threshold-free separation of correct vs incorrect.
- **AUPR** — preferred under class imbalance (errors may be the minority in health); report both
  the positive- and negative-class variants.
- **AUARC / risk–coverage** — selective prediction: as we abstain on the most-uncertain, how fast
  does accuracy on the retained set rise? **This is the metric that most directly maps to the
  guardrail's job** (block/escalate the uncertain). Report **risk at fixed coverage** (e.g. 80%)
  and **coverage at fixed risk**.
- **PRR (Prediction–Rejection Ratio)** — the same rejection curve, normalized against oracle and
  random baselines: PRR = 1 means the uncertainty ranks responses by correctness as well as a
  perfect scorer, PRR = 0 means no better than random rejection. It makes methods comparable across
  datasets with different base error rates (which raw AUARC does not). Report alongside AUARC.

**B. Calibration — do the confidence values match empirical accuracy?**
- **ECE** (baseline; global-average, hides local risk).
- **ACE** (adaptive/equal-mass bins) — fixes clustering instability at the 0.9–1.0 pile-up.
- **MCE** (worst bin) — the safety-critical calibration metric for medical boundaries.
- **Brier score** — joint accuracy+calibration; penalizes confidently-wrong *and* uselessly-safe.
- **KDE-ECE, class-wise ECE** — binning-free and per-class refinements.
- *Wrinkle:* verbalized and consistency scores aren't probabilities in [0,1]. Calibration metrics
  require a monotone score→probability map (isotonic/Platt) fit on a held-out split; **fit and
  report that mapping**, and keep the discrimination metrics (which don't need it) as the
  primary comparison. **Enforced in code:** the metric dispatcher refuses to compute a calibration
  metric for a method still carrying the default uncalibrated normalizer, because silently scoring
  a raw entropy as if it were a probability is exactly the failure mode this warning describes.

**C. Safety-weighted / boundary metrics — health-specific (ECE globally hides the "danger zone").**
- **Per-stratum MCE / ECE** on high-risk items (medical-advice, crisis, body-image strata).
- **Harm recall at the operating point** — at the threshold where we block/escalate, what fraction
  of genuinely unsafe responses is caught (target our Stanford-derived **Recall ≥ 0.96**), and at
  what **benign-rejection / over-refusal cost** (precision at that point).
- **Selective risk at target coverage** on the safety slice specifically.

---

## 5. Latency protocol (measure in wall-clock milliseconds)

The query-count multiplier is the coarse story and stays as a hardware-independent reference
(recorded per method as `NUM_TARGET_CALLS`). The **deliverable is measured wall-clock in ms**,
because "single-pass < multi-sample" hides two things only measurement reveals: whether a method
fits an absolute real-time budget, and the parallel-vs-serial effect that can either collapse or
amplify the N× penalty.

**Stage decomposition (report each stage in ms).**
Total per-estimate latency = target-generation time + extra-generation time + auxiliary compute.
- *Target generation* — the one answer the user gets anyway; measure it, but **exclude from marginal latency**.
- *Extra target generations* — multi-sample draws and chain-of-interaction turns; the dominant marginal term.
- *Auxiliary compute* — NLI clustering, embedding, proxy forward pass, tensor decomposition; the query-independent floor.

**Marginal vs total.** Report **marginal latency** = time the guardrail adds beyond producing the
user's answer. This is the number the SLA is written against; total is secondary.

**Isolating the UQ overhead (the core measurement).** Measure it as a **generator × UQ-family matrix**:
- Baseline `g` = generator-only time (produce the answer, no UQ), measured per model.
- Method total `t` = end-to-end time with the UQ method.
- **Marginal UQ = t − g** (absolute ms), and **overhead ratio = (t − g) / g** (dimensionless).

Report both: absolute ms answers "does it fit this deployment's budget," the ratio answers "what is the
method's structural cost, independent of how fast the generator is." Rows are generators (open and
closed, each a fixed checkpoint/endpoint); columns are the families (Pob / SC / VB / P).

**Two attribution modes, both reported** (this is a consequence of upstream fact #1 above — the
shared sample cache makes the two questions genuinely different, and reporting only one is
misleading):
- `shared_cache` — §6's fairness control: one sample set drawn at `max(N)` and reused by every
  method. Isolates auxiliary compute; answers "which scorer is cheapest given the samples exist."
- `isolated` — each method runs its own pass. Answers "what does adding *this method alone* to the
  deployment cost," which is the SLA question.

**The finding the matrix exposes — coupling to the generator.** UQ families differ in *whether their
overhead scales with the generator*, and that, not "single vs multi," is the real axis:
- **P via proxy (DisAAD, reference proxy)** — overhead is one small proxy forward pass,
  **decoupled** from the generator: roughly constant ms regardless of target, so its overhead *ratio*
  shrinks as the generator slows. State it as an absolute constant (tens of ms).
- **VB** — bimodal: appended to the answer → ≈0 marginal; issued as a separate call → +1 generation (ratio ≈ 1).
- **SC** — overhead is (N−1) extra generations, **coupled/multiplicative** in the generator: ratio ≈ N−1
  serial (≈ generator-tail concurrent), so its *absolute* ms explodes on slow generators even though the
  ratio is constant. State it as a ratio.

**Open vs closed changes the generator term, not the method term.** Closed APIs have large, variable `g`
(network + queue + large model), so coupled families (SC) pay far more in absolute ms there, and API
concurrency caps may force SC serial (removing the parallel rescue). Open models give small, controlled
`g` on your own hardware. **LRMs** inflate `g` most (long reasoning traces), so the worst cell in the
whole matrix is **SC × closed LRM** (re-running the full trace N times), while **P-proxy × LRM** stays
cheap (one pass over the trace) — the quantitative reason a single-pass/proxy method matters most exactly
for expensive reasoning generators.

**Parallel vs serial — measure both; they give different answers.**
- Multi-sample draws are *independent* — issue concurrently (batched locally or concurrent API calls).
  Under concurrency, N samples cost ≈ the **tail of N parallel generations**, not N × one generation.
  Report the **serial** cost (N sequential) *and* the **concurrent** cost (max of N parallel + overhead).
- Chain-of-interaction turns are *dependent* — each conditions on the previous, so they **cannot** be
  parallelized; cost is unavoidably **depth × per-turn generation**. This is why serial-multi-turn is
  the worst-case latency, and why moving the chain onto a cheap proxy is the only available lever.

**Measurement hygiene (for trustworthy ms).**
- Fix and report: target model, hardware / API endpoint, output token count (latency scales with it),
  batch size, concurrency level, sampling params. *Recorded by construction* — every result artifact
  embeds the config hash, rather than relying on discipline.
- Warm up (discard first runs); ≥100 trials per method; report the **full distribution — p50, p95, p99**
  (tail is what a real-time guardrail feels), never just the mean.
- For API targets, separate *our compute* from *network + queue*: log request-sent → first-token → last-token,
  and report generation time and round-trip separately so numbers transfer across deployments.
- Time auxiliary models (NLI / embedding / proxy) in isolation so the query-independent floor is visible.

**DisAAD:** report **offline per-target distillation cost** and **online inference cost** as two
separate budget lines — they are not the same commitment.

**Budget/SLA column.** Beyond the plot, report each method's **marginal p95 against a fixed guardrail
budget** (e.g. 500 ms / 1 s / 2 s) as a pass/fail "fits real-time?" verdict — for an inline guardrail
the question is absolute admissibility, not only relative ordering.

**Deliverable plot:** best accuracy metric vs **marginal wall-clock ms** (p50, p95 whiskers), with N and
chain-depth swept, and **serial vs concurrent as separate series**. This is the answer to Q1.

---

## 6. Controls (so the numbers are trustworthy)

- **Same generations for every method:** cache the target's primary answer + N samples once; all
  scorers consume the identical cache. This isolates generation latency and removes it as a
  confound between methods. *(Upstream `run_truth_methods` already does this in-memory; Stage A
  persists it to disk so the N-sweep and every re-scoring pass reuse the identical draws.)*
- **Fixed decoding:** one decoding config for the primary answer; one held-constant sampling
  config (temp / top-p) for the multi-sample draws, identical across methods.
- **Variance is large** — DisAAD's own tables show ±3–5 AUROC. Run **multiple seeds**, report
  **mean ± std**, and put **bootstrap CIs** on AUROC/AUPR. Single-run numbers will mislead.
- **Report per-dataset, never only the average** — averaging hid DisAAD's biomedical weakness;
  don't let it hide ours.
- **Include the white-box reference line**, clearly labeled *not available in deployment*, to size
  the black-box gap honestly.

---

## 7. Harness

Implemented in `hc_benchmark/`. Mirrors the DisAAD repo skeleton so the two are comparable.

- **Stage A — Generate.** For each (item, target): primary answer + N_max samples → Parquet cache
  keyed by (dataset, generator, decoding config, seed, item id).
- **Stage B — Label.** Correctness annotation on the primary answer (BLEURT **and** LLM-judge, both persisted).
- **Stage C — Score.** Each UQ method reads the cache, emits one uncertainty score per response,
  and logs its own timing record, swept over N ∈ {1,3,5,10,20}.
- **Stage D — Evaluate.** Compute the §4 metric suite + §5 frontier plots, per-dataset and pooled,
  with multi-seed mean ± std and bootstrap CIs.

Stages A/B/C/D map directly onto DisAAD's `generate_black.py` → BLEURT/LLM annotation →
scoring → `one_eval.py` (`third_party/DisAAD/scripts/`), so that repo is a usable cross-check for A, B, and D.

> Note on the DisAAD shell scripts: `SH_set/train.sh` actually invokes `data_builder.py` (data
> collection) and `SH_set/one_eval.sh` actually invokes the training script — the two filenames are
> swapped relative to the README's stage ordering. Follow the Python entry points, not the filenames.

---

## 8. Open decisions for the team

1. Primary + secondary **target model**.
2. Final **dataset** set, and whether the real coaching slice is ready/labelable for a first pass.
3. **Correctness criterion** of record (the harness runs both BLEURT and LLM-judge by default).
4. Whether to pursue an explicit license from the DisAAD authors, re-implement clean-room, or drop
   the method — see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
