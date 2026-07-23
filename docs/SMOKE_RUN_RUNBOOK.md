# Live smoke-run runbook

The goal of the smoke run is to **prove the whole G → D → M → V pipeline works end to end**
on a real API target — nothing in this repo has run live yet; it's all been unit-tested
offline. This runbook takes you from a fresh clone to a green run and tells you exactly what
"green" looks like.

It is intentionally tiny (a handful of items, ~a few dollars of API calls, minutes of
wall-clock). It validates the machinery, **not** any research claim — AUROC ≈ 0.5 on ten
items is expected and fine.

---

## 0. What you need

- A machine with **Python ≥ 3.10** and either a GPU or CPU (CPU works; the NLI clustering
  model just runs slower).
- **~8 GB disk** for the ML stack + the `microsoft/deberta-large-mnli` NLI model (~1.5 GB,
  downloaded on first use by the consistency methods).
- An **OpenAI API key** with a few dollars of budget (used for both the target *and* the
  LLM judge). Any LiteLLM-supported provider works — see the variations at the end.

You do **not** need the GPU cluster for the smoke run (it uses an API target). GPUs matter
only for open-model targets and DisAAD proxy training.

---

## 1. Clone and check out the branch

```bash
git clone https://github.com/jsl5710/TruthTorchLM-HC.git
cd TruthTorchLM-HC
git checkout feat/hc-benchmark-infra
```

## 2. Fetch the method submodules

The seven official method repos are pinned as submodules. IUQ, SPUQ, DSE, EigV/LeS, NCB,
and the OOD gate were ported into `src/`/`hc_benchmark/`, so the smoke run itself doesn't
import them — but fetch them anyway so the tree is complete and DisAAD training is possible
later.

```bash
git submodule update --init --recursive --depth 1
```

## 3. Create an environment and install

```bash
conda create -y --name ttlm-hc python=3.10 && conda activate ttlm-hc   # or python -m venv
pip install -r requirements.txt
pip install -e .
```

> On a GPU box, install the CUDA build of PyTorch **first** (per your CUDA version, from
> pytorch.org) before `pip install -e .`, so torch isn't pulled as CPU-only.

## 4. Verify the install offline (no API calls)

```bash
# the offline test suite must be green
python -m pytest -q            # expect ~249 passed, a few skipped

# see what's supported and what's ready to run
python -m hc_benchmark.capabilities
python -m hc_benchmark.readiness      # 8/10 ready; DisAAD + OOD gate need prep (expected)
```

If the suite is green, the pure logic is sound and only the live wiring remains.

## 5. Set the API key

```bash
export OPENAI_API_KEY=sk-...
```

Optional: confirm the target behaves as a pure text-only black box before spending on a
full run (this makes one cheap call):

```bash
python scripts/verify_provider.py openai        # expect: PASS ... TEXT-ONLY (black-box OK)
```

## 6. Run the smoke test

```bash
python scripts/smoke_run.py
```

Knobs (all optional, via env):

| Var | Default | Meaning |
| :-- | :-- | :-- |
| `SMOKE_GENERATOR` | `gpt-4o-mini` | target + judge model (LiteLLM id) |
| `SMOKE_DEVICE` | `cuda` if available else `cpu` | where the NLI model loads |
| `SMOKE_DATASETS` | `trivia_qa,medqa` | comma-separated dataset keys |
| `SMOKE_SIZE` | `0.004` | fraction of each dataset (keep tiny) |

First run downloads `deberta-large-mnli` (~1.5 GB) — subsequent runs are faster.

---

## 7. What a PASS looks like

Expected shape of the output (numbers will vary):

```
Smoke run: generator=gpt-4o-mini device=cpu datasets=['trivia_qa', 'medqa'] size=0.004
================================================================
DATASET: trivia_qa
================================================================
[A] cached 8 items -> stageA_trivia_qa_gpt-4o-mini_<hash>_seed0.parquet
[B] labelled 8 items (accuracy 6/8)
[D] VerbalizedConfidence     N=1: AUROC=0.583 PRR=0.210 ECE=0.31 MCE=0.55 Brier=0.24 | aux p50=0.0ms fits500ms=True
[D] DiscreteSemanticEntropy  N=5: AUROC=0.667 PRR=0.333 ECE=0.28 MCE=0.60 Brier=0.22 | aux p50=48.1ms fits500ms=True
[D] NumSemanticSetUncertainty N=5: AUROC=0.625 ...
================================================================
DATASET: medqa
================================================================
...
================================================================
§5 CONCURRENCY CHECK (N=5 draws)
================================================================
  serial     :    3200 ms
  concurrent :     900 ms
  -> concurrency RESCUES multi-sample (concurrent < serial)

Smoke run complete. Results -> hc_benchmark/results/smoke_run.json
```

**The run is green when all of these hold:**

1. **Stage A** printed a cached item count and a Parquet filename for each dataset, and the
   file exists under `hc_benchmark/cache/`.
2. **Stage B** printed a label count and accuracy for each dataset.
3. **Stage D** printed a full metric row (AUROC / PRR / ECE / MCE / Brier) for **every
   method × N**, per dataset — numbers, not `None`/exceptions.
4. **`VerbalizedConfidence` aux p50 ≈ 0 ms** — it's a single-pass VB method with no
   auxiliary compute (protocol §5's bimodal-VB prediction).
5. **The consistency methods have non-trivial aux ms** (NLI clustering) and an SLA verdict.
6. **The §5 concurrency check ran** and (usually) reports `concurrent < serial`.
7. `hc_benchmark/results/smoke_run.json` was written.

Sanity, not accuracy: AUROC near 0.5 and ECE anywhere in [0,1] are fine on ~8 items — the
point is that every stage produced a real number.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| :-- | :-- |
| `preflight failed: ML stack not importable` | `pip install -e .` didn't finish, or wrong env active. |
| `OPENAI_API_KEY is not set` | `export OPENAI_API_KEY=...` in the same shell. |
| Hangs / slow on first run | Downloading `deberta-large-mnli`; wait it out (one-time). |
| CUDA OOM on the NLI model | `export SMOKE_DEVICE=cpu` (the smoke run is tiny; CPU is fine). |
| `RateLimitError` from the API | Lower `SMOKE_SIZE`, or wait; the SDK retries automatically. |
| Concurrency check: `did NOT beat serial` | The API account's concurrency cap serialized the calls — expected on constrained keys; not a failure of the code. |
| `AUROC couldn't be calculated ... Returning 0.5` | All items got the same correctness label (tiny sample). Raise `SMOKE_SIZE` slightly. |
| MCQ (`medqa`) labels look wrong | The smoke run uses the LLM judge for all datasets; MCQ correctness is best judged with `evaluators.MCQMatch` in a real run. Fine for the smoke. |

---

## 9. Variations

**Different provider (Anthropic / Gemini):** set the key and the generator id.
```bash
export ANTHROPIC_API_KEY=...   # or GEMINI_API_KEY=...
SMOKE_GENERATOR=claude-haiku-4-5 python scripts/smoke_run.py     # or gemini-2.5-pro
```
(The registry resolves `claude-haiku-4-5` → `anthropic/claude-haiku-4-5` for LiteLLM.)

**Health biomedical slice:**
```bash
SMOKE_DATASETS=bioasq python scripts/smoke_run.py
```

**Once the smoke run is green**, scale up with a real config:
```bash
python -m hc_benchmark.run --config hc_benchmark/configs/health_primary.yaml
```

---

## 10. What this does *not* cover (next steps)

- **Open-model targets + the white-box reference line** — needs a GPU host and
  `generate_stage_a_local(...)` with a loaded model.
- **DisAAD** — needs a trained proxy first (`hc_benchmark/disaad_train.py` on the cluster,
  then `DisAAD.from_pretrained(...)`); `python -m hc_benchmark.readiness --proxy-path ...`
  tells you when it's ready.
- **Serial-vs-concurrent as a swept frontier series**, multi-seed CIs, and the deliverable
  frontier plot — those are the full `hc_benchmark.run` grid, not the smoke run.
