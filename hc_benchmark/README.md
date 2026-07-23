# hc_benchmark — Stage A→D harness

The orchestration layer for the pure black-box UQ benchmark
([`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md)). It sits **on top of** the
`TruthTorchLM` library and wires the fork's additions — latency instrumentation,
calibration/safety metrics, health datasets — into four cached stages that mirror the
DisAAD repo skeleton.

```
Stage A  generate   primary answer + n_max samples  →  Parquet cache   (stage_a_generate.py)
Stage B  label      correctness via BLEURT and LLM-judge, both kept    (stage_b_label.py)
Stage C  score      each method reads the cache, sweeps N∈{1,3,5,10,20} (stage_c_score.py)
Stage D  evaluate   §4 metrics + §5 frontier, per-dataset, mean±std     (stage_d_evaluate.py)
```

## The one idea that matters

**The same generations feed every method and every N.** Stage A draws them once and
caches them; Stage C serves each N in the sweep by *truncating* that cached list, and
hands it to each `TruthMethod` as a pre-built `sampled_generations_dict` so the target is
never called again at score time. This is protocol §6's central control: it removes
generation cost and generation variance as confounds between methods, and turns the
N-sweep into a cache read rather than N separate runs.

Every artifact is keyed by `BenchmarkConfig.content_hash()`, so a change to decoding
params yields a new cache rather than silently reusing an incompatible one (protocol §5
measurement hygiene).

## Run it

```bash
git submodule update --init --recursive --depth 1
pip install -r requirements.txt && pip install -e .

export OPENAI_API_KEY=...          # for an API target + LLM judge
python -m hc_benchmark.run --config hc_benchmark/configs/smoke.yaml
```

`smoke.yaml` is the round-one acceptance test (≈50 items, one API generator, two
black-box methods, N∈{1,5}). `health_primary.yaml` is a representative research config.

### Targets (the G axis)

Generators live in one registry, [`generators.py`](generators.py), so a config names a
target by a **friendly key** and the harness fills in the provider-qualified model string,
backend, access level, and reasoning-trace policy. Adding a target is a config edit, not
code.

| Provider | Key example | Resolves to | Key env var |
|---|---|---|---|
| OpenAI | `gpt-4o-mini` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-opus-4-8` | `anthropic/claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini-2.5-pro` | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Open (HF) | `llama-3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | — (hosted) |

Ready-to-run configs: `configs/anthropic_claude.yaml`, `configs/gemini.yaml`,
`configs/open_llama.yaml`. Closed APIs are black-box targets; open models are white-box
(the reference line + the small-proxy substrate the P family needs). The registry also
flags reasoning models (Claude/Gemini thinking, DeepSeek-R1, Qwen3) and records whether a
UQ method sees the trace or only the answer.

**Before a real Claude/Gemini run**, confirm the API is genuinely text-only black-box:

```bash
export ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...
python scripts/verify_provider.py anthropic gemini
```

## Which methods need a prep step first?

Most methods are inference-only — construct and run. **Two** need a preparation step, and
the readiness report tells you which, and whether the prep is done:

| Component | Needs | Ready when |
| :--- | :--- | :--- |
| **DisAAD** | a distilled proxy (offline teacher→student training) | a training manifest (`disaad_ready.json`) sits next to the proxy — written automatically when `train_proxy()` finishes |
| **OOD-PCA gate** | a one-time `PCAGate(...).fit(kb_documents)` | the gate has been fitted (`gate.is_ready()`) |

Check status at a glance — before a run, and to confirm a cluster training job finished:

```bash
python -m hc_benchmark.readiness --proxy-path hc_benchmark/disaad/proxy
```

```
Method readiness — 8/10 ready to run
READY NOW:
  ✓ DiscreteSemanticEntropy   inference-only — ready to run
  ✓ SPUQ / IUQ / NCB / ...     inference-only — ready to run
NEEDS A PREP STEP FIRST:
  ✗ DisAAD                     no trained proxy — cannot score yet
      → Train a proxy on the GPU server via hc_benchmark/disaad_train.train_proxy(...) ...
```

The DisAAD row flips to `✓ proxy ready … (teacher=…, student=…, trained_at=…)` the moment
the training job writes its manifest — so a user who didn't run the training still knows,
instantly, that a proxy is present and usable. `DisAAD.from_pretrained(path)` also refuses a
directory without that manifest, so you never silently score against a half-trained proxy.

## What is deliberately left to the launch script

- **Method construction** (`run.build_methods`) is code, not config: a `TruthMethod` may
  need a loaded entailment model or a proxy checkpoint — an object, not a string. Round
  one ships the black-box pair `VerbalizedConfidence` + `NumSemanticSetUncertainty`
  (the text-only cluster-count consistency method; `SemanticEntropy` is grey-box here and
  excluded by the §1 filter).
- **Local (HuggingFace) targets** need a model + tokenizer handed in; `stage_a_generate`
  wires the API path and points you at the local one.
- **DisAAD and the other round-two methods** are not in `build_methods` yet — see the
  round-two section of the plan and `THIRD_PARTY_NOTICES.md`.
