"""DisAAD proxy training — the teacher -> student distillation entry point (setup only).

This is the **offline Stage 1** of DisAAD (Cui et al., ACL 2026): collect the black-box
**target/teacher** outputs across diverse prompts, then adversarially distil a small open
**proxy/student** that reproduces the teacher's high-probability output regions. The
trained proxy is then used by `TruthTorchLM.truth_methods.DisAAD` for Stage-2 scoring.

**Design decision (per the research plan): setup, not a local run.** DisAAD's source
declares no license, so we never copy its code — training is driven by shelling out to the
official scripts in `third_party/DisAAD/scripts`, which assume a multi-GPU box and load
models locally. This module builds and (on request) runs those commands; it will refuse to
run unless the submodule is present, and is intended to be launched on the GPU server. The
command construction itself is pure and unit-tested, so the teacher/student wiring is
verifiable without any GPUs.

Two stages, two official scripts:
  1. ``data_builder.py``  — collect teacher outputs -> SFT distillation dataset.
  2. ``train_disaad.py``  — distribution-aligned adversarial distillation of the proxy.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["DisAADTrainingConfig", "build_data_builder_command", "build_train_command",
           "collect_distillation_data", "train_proxy",
           "TRAINING_MANIFEST_NAME", "write_training_manifest", "read_training_manifest",
           "is_proxy_trained"]

# The official scripts live in the pinned submodule; we invoke, never import/copy them.
_DISAAD_SCRIPTS = Path("third_party/DisAAD/scripts")

# A completion marker written next to a trained proxy. Its presence is the signal that
# "training finished, this proxy is ready to use" -- so DisAAD scoring and the readiness
# report can tell a ready proxy from an empty/half-written output directory.
TRAINING_MANIFEST_NAME = "disaad_ready.json"


def write_training_manifest(config: "DisAADTrainingConfig", proxy_path: str = None) -> Path:
    """Stamp a proxy directory as trained-and-ready. Called on successful training."""
    proxy_path = Path(proxy_path or config.proxy_output_path)
    proxy_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ready": True,
        "method": "DisAAD",
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "teacher_is_api": config.teacher_is_api,
        "dataset": config.dataset,
        "train_num_samples": config.train_num_samples,
        "epochs": config.epochs,
        "seed": config.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    path = proxy_path / TRAINING_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_training_manifest(proxy_path: str) -> dict:
    """Return the training manifest for a proxy dir, or None if it isn't trained/ready."""
    path = Path(proxy_path) / TRAINING_MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if data.get("ready") else None


def is_proxy_trained(proxy_path: str) -> bool:
    """True if ``proxy_path`` holds a completed DisAAD proxy (has a ready manifest)."""
    return read_training_manifest(proxy_path) is not None


@dataclass
class DisAADTrainingConfig:
    """Teacher -> student distillation configuration.

    ``teacher_model`` is the target whose black-box behaviour we distil; ``student_model``
    is the small open proxy we train. These map onto the official scripts'
    ``--target_model_name`` and ``--scoring_model_name`` (and, for API teachers,
    ``--api_model_name``).
    """

    teacher_model: str                 # the target (e.g. "gpt-4-0613", "llama2-70b")
    student_model: str = "llama3.2-3b"  # the proxy to distil (small open model)
    teacher_is_api: bool = False        # True -> data_builder uses the API teacher path

    dataset: str = "tqa"
    ori_dataset_path: str = "third_party/DisAAD/datasets/mixed_questions/wild_tqa"
    sft_dataset_path: str = "hc_benchmark/disaad/sft_data"
    proxy_output_path: str = "hc_benchmark/disaad/proxy"
    cache_dir: str = "hc_benchmark/disaad/cache"

    num_samples: int = 140              # teacher generations to collect for distillation
    train_num_samples: int = 100
    val_num_samples: int = 50
    epochs: int = 1
    seed: int = 42
    gpu_ids: str = "0,1,2,3"

    # sampling for the teacher-output collection
    temperature: float = 0.7
    top_p: float = 0.9

    api_key: str = ""                   # only when teacher_is_api
    api_base: str = "https://api.openai.com/v1"

    extra_data_args: list = field(default_factory=list)
    extra_train_args: list = field(default_factory=list)


def build_data_builder_command(config: DisAADTrainingConfig) -> list:
    """Stage 1-1: the command that collects teacher outputs into an SFT distillation set."""
    cmd = [
        "python", str(_DISAAD_SCRIPTS / "data_builder.py"),
        "--what_to_do", "api" if config.teacher_is_api else "local",
        "--dataset", config.dataset,
        "--ori_dataset_path", config.ori_dataset_path,
        "--sft_dataset_path", config.sft_dataset_path,
        "--num_samples", str(config.num_samples),
        "--cache_dir", config.cache_dir,
        "--seed", str(config.seed),
        "--do_temperature", "--temperature", str(config.temperature),
        "--do_top_p", "--top_p", str(config.top_p),
        "--gpuids", config.gpu_ids,
    ]
    if config.teacher_is_api:
        cmd += ["--api_model_name", config.teacher_model,
                "--api_base", config.api_base, "--api_key", config.api_key]
    else:
        cmd += ["--base_model_name", config.teacher_model]
    return cmd + list(config.extra_data_args)


def build_train_command(config: DisAADTrainingConfig) -> list:
    """Stage 1-2: the command that adversarially distils the proxy (student) from the teacher."""
    train_data = f"{config.sft_dataset_path}.raw_data.json"
    return [
        "python", str(_DISAAD_SCRIPTS / "train_disaad.py"),
        "--target_model_name", config.teacher_model,       # teacher
        "--scoring_model_name", config.student_model,      # student / proxy
        "--train_dataset_path", train_data,
        "--output_path", config.proxy_output_path,
        "--train_num_samples", str(config.train_num_samples),
        "--val_num_samples", str(config.val_num_samples),
        "--epochs", str(config.epochs),
        "--seed", str(config.seed),
        "--cache_dir", config.cache_dir,
        "--gpu_ids", config.gpu_ids,
    ] + list(config.extra_train_args)


def _require_submodule():
    if not _DISAAD_SCRIPTS.exists():
        raise FileNotFoundError(
            f"{_DISAAD_SCRIPTS} not found. Initialise the submodule first:\n"
            "  git submodule update --init --recursive --depth 1\n"
            "DisAAD's code is used as an unmodified submodule (it declares no license)."
        )


def collect_distillation_data(config: DisAADTrainingConfig, dry_run: bool = False):
    """Run Stage 1-1 (teacher-output collection). GPU server only."""
    _require_submodule()
    cmd = build_data_builder_command(config)
    print("[DisAAD] data collection:", " ".join(cmd))
    if dry_run:
        return cmd
    return subprocess.run(cmd, check=True)


def train_proxy(config: DisAADTrainingConfig, dry_run: bool = False):
    """Run Stage 1 end to end: collect teacher outputs, then distil the proxy. GPU server only.

    Returns the proxy output path on success; with ``dry_run`` returns the two commands
    without executing, so the teacher/student wiring can be inspected anywhere.
    """
    _require_submodule()
    data_cmd = build_data_builder_command(config)
    train_cmd = build_train_command(config)
    if dry_run:
        return {"data_builder": data_cmd, "train_disaad": train_cmd}

    print("[DisAAD] Stage 1-1: collecting teacher outputs ...")
    subprocess.run(data_cmd, check=True)
    print("[DisAAD] Stage 1-2: adversarially distilling the proxy ...")
    subprocess.run(train_cmd, check=True)
    # Mark the proxy ready so DisAAD scoring / the readiness report know it can be used.
    write_training_manifest(config)
    print(f"[DisAAD] proxy written to {config.proxy_output_path} "
          f"(marked ready: {TRAINING_MANIFEST_NAME})")
    return config.proxy_output_path
