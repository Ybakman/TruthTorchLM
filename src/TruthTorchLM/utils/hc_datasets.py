"""Dataset loaders added by the HC fork (benchmark protocol, D axis).

Upstream covers general-domain free-form QA well (trivia_qa, natural_qa, pop_qa,
simple_qa, web_questions, wikipedia_factual, narrative_qa, gsm8k). What the benchmark
needs and upstream has none of is **health across every format**, plus MCQ, extractive,
and adversarial general-domain sets.

Every loader returns upstream's item contract so the rest of the library is untouched::

    {"context": str, "question": str, "ground_truths": list[str]}

with two optional HC additions, defaulted so existing loaders keep working:

* ``stratum`` -- risk stratum for per-stratum calibration (protocol §4C).
* ``outcome_type`` -- ``factual_error`` | ``safety_violation`` | ``belief_collapse``,
  the axis protocol Q5 says must not be lumped together.

All HuggingFace identifiers and column names below were verified against the live
datasets-server, not assumed. Several are counter-intuitive and are commented where they
bite: MMLU's answer is an integer index while MedQA's is answer *text*; MedQA's options
are a dict keyed "A".."D" rather than a list; SQuAD 2.0 marks unanswerable questions with
*empty lists* rather than a null; K-QA and MedLFQA capitalize their column names; and
TruthfulQA ships only a validation split.
"""

from tqdm import tqdm


def load_dataset(*args, **kwargs):
    """Lazy proxy for ``datasets.load_dataset``.

    Imported on call rather than at module scope so the row-shaping helpers below
    (``_format_mcq``, ``_parse_bioasq_text``) stay importable -- and unit-testable --
    in an environment without the ``datasets`` package. Those helpers encode the
    dataset-specific gotchas, so they are the part most worth testing cheaply and often.
    """
    from datasets import load_dataset as _load_dataset

    return _load_dataset(*args, **kwargs)


__all__ = [
    "get_hotpot_qa",
    "get_truthful_qa",
    "get_squad_v2",
    "get_medqa",
    "get_mmlu_med",
    "get_kqa",
    "get_medlfqa",
    "get_bioasq",
    "MMLU_MEDICAL_SUBJECTS",
    "OPTION_LETTERS",
]

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

#: The medical/clinical slice of MMLU. MMLU has no "medical" config; MMLU-med in the
#: literature is the union of these subject configs.
MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "virology",
    "nutrition",
    "human_aging",
]


def _subsample(raw_dataset, size_of_data, seed):
    """Match upstream's subsampling convention exactly (see dataset_utils.get_trivia_qa)."""
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)["train"]
    return raw_dataset


def _item(question, ground_truths, context="", stratum=None, outcome_type="factual_error"):
    return {
        "context": context,
        "question": question,
        "ground_truths": ground_truths,
        "stratum": stratum,
        "outcome_type": outcome_type,
    }


def _format_mcq(question: str, options: list) -> str:
    """Render an MCQ as a single prompt string.

    The options have to live in the question text because upstream's user prompt
    template only interpolates ``{question}``; putting them anywhere else would mean
    asking the model to choose between options it was never shown.
    """
    lines = [f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return (
        f"{question.strip()}\n\n"
        + "\n".join(lines)
        + "\n\nAnswer with the letter of the correct option."
    )


# ---------------------------------------------------------------------------
# General domain -- the reference points that expose domain sensitivity
# ---------------------------------------------------------------------------


def get_hotpot_qa(size_of_data: float = 1.0, seed: int = 0, split="test", config="distractor"):
    """HotpotQA -- multi-hop general-domain QA.

    The ``distractor`` config supplies the gold paragraphs plus distractors, so the
    multi-hop difficulty is retrieval-independent. ``fullwiki``'s test split is unlabeled
    (``answer`` is null), so we never map ``split="test"`` onto it.
    """
    hf_split = "validation" if split == "test" else "train"
    raw_dataset = load_dataset("hotpotqa/hotpot_qa", config, split=hf_split, trust_remote_code=True)
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing HotpotQA"):
        # `context` is {"title": [...], "sentences": [[...], ...]} -- one sentence list
        # per title, so it needs flattening rather than a plain join.
        paragraphs = [
            f"{title}: {' '.join(sentences)}"
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        dataset.append(
            _item(
                question=row["question"],
                ground_truths=[row["answer"]],
                context="\n".join(paragraphs),
            )
        )
    return dataset


def get_truthful_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    """TruthfulQA (generation config) -- adversarial questions that invite plausible falsehoods.

    Ships **only a validation split**; both ``split`` values map to it, which is stated
    here rather than silently, because a "train" split that is secretly the test split is
    exactly how a calibration set leaks into evaluation.
    """
    raw_dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing TruthfulQA"):
        # best_answer first: several correctness evaluators use ground_truths[0] as the
        # canonical reference, and the remaining correct_answers are valid aliases.
        ground_truths = [row["best_answer"]] + [
            a for a in row["correct_answers"] if a and a != row["best_answer"]
        ]
        dataset.append(
            _item(
                question=row["question"],
                ground_truths=ground_truths,
                stratum=row.get("category"),
            )
        )
    return dataset


def get_squad_v2(size_of_data: float = 1.0, seed: int = 0, split="test",
                 unanswerable_only: bool = False):
    """SQuAD 2.0 -- extractive QA with an unanswerable subset.

    The unanswerable questions are the interesting part for us: they are a natural
    abstention slice, where the correct behaviour is to decline rather than to answer.

    Unanswerable rows have ``answers = {"text": [], "answer_start": []}`` -- empty lists,
    not null and not a sentinel string -- so they are detected by emptiness.
    """
    hf_split = "validation" if split == "test" else "train"
    raw_dataset = load_dataset("rajpurkar/squad_v2", split=hf_split)
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing SQuAD 2.0"):
        answers = row["answers"]["text"]
        is_unanswerable = len(answers) == 0
        if unanswerable_only and not is_unanswerable:
            continue
        dataset.append(
            _item(
                question=row["question"],
                # For an unanswerable question the correct response is an abstention, so
                # the reference is the refusal string rather than an empty list -- an
                # empty ground_truths would make every response trivially wrong.
                ground_truths=["unanswerable"] if is_unanswerable else list(dict.fromkeys(answers)),
                context=row["context"],
                stratum="unanswerable" if is_unanswerable else "answerable",
            )
        )
    return dataset


# ---------------------------------------------------------------------------
# Health -- empty upstream across every format
# ---------------------------------------------------------------------------


def get_medqa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    """MedQA (USMLE, 4 options) -- clinical MCQ.

    Column gotchas, verified live: ``options`` is a **dict** keyed "A".."D", not a list;
    ``answer`` is the option *text*; ``answer_idx`` is the letter. ``ground_truths`` is
    emitted as ``[letter, option_text]`` for ``evaluators.MCQMatch``.
    """
    hf_split = "test" if split == "test" else "train"
    raw_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=hf_split)
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing MedQA"):
        options = [row["options"][letter] for letter in OPTION_LETTERS if letter in row["options"]]
        dataset.append(
            _item(
                question=_format_mcq(row["question"], options),
                ground_truths=[row["answer_idx"], row["answer"]],
                stratum="medical_mcq",
            )
        )
    return dataset


def get_mmlu_med(size_of_data: float = 1.0, seed: int = 0, split="test", subjects=None):
    """MMLU medical subjects -- clinical MCQ across nine subject configs.

    ``answer`` here is an **integer index 0-3**, unlike MedQA's answer text. The stratum
    is the MMLU subject, which is what makes graded domain shift measurable: fit on one
    subject, test on another (protocol Q2).
    """
    subjects = subjects or MMLU_MEDICAL_SUBJECTS
    hf_split = "test" if split == "test" else "validation"

    dataset = []
    for subject in tqdm(subjects, desc="Processing MMLU-med"):
        raw_dataset = load_dataset("cais/mmlu", subject, split=hf_split)
        raw_dataset = _subsample(raw_dataset, size_of_data, seed)
        for row in raw_dataset:
            choices = list(row["choices"])
            answer_index = int(row["answer"])
            dataset.append(
                _item(
                    question=_format_mcq(row["question"], choices),
                    ground_truths=[OPTION_LETTERS[answer_index], choices[answer_index]],
                    stratum=subject,
                )
            )
    return dataset


def get_kqa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    """K-QA -- real patient questions with clinician-written long-form answers.

    Two unusual properties, both verified: the splits are named
    ``questions_with_answers`` and ``questions`` (not train/test) and have **incompatible
    schemas** -- only the former carries answers, so only it is usable here. Columns are
    capitalized.

    ``Must_have`` statements are the claim-level references the long-form pipeline scores
    against, so they are carried after the free-form answer rather than discarded.
    """
    raw_dataset = load_dataset("Itaykhealth/K-QA", split="questions_with_answers")
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing K-QA"):
        ground_truths = [row["Free_form_answer"]] + list(row.get("Must_have") or [])
        dataset.append(
            _item(
                question=row["Question"],
                ground_truths=ground_truths,
                stratum=row.get("ICD_10_diag") or "health_long_form",
            )
        )
    return dataset


def get_medlfqa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    """MedLFQA -- long-form medical QA with must-have / nice-to-have statements.

    Test split only. Note the dataset carries no column identifying which source subset
    (LiveQA / MedicationQA / HealthSearchQA / K-QA) a row came from, so a source-based
    OOD split is not available from this artifact alone.
    """
    raw_dataset = load_dataset("dmis-lab/MedLFQA", split="test")
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    for row in tqdm(raw_dataset, desc="Processing MedLFQA"):
        ground_truths = [row["Free_form_answer"]] + list(row.get("Must_have") or [])
        dataset.append(
            _item(question=row["Question"], ground_truths=ground_truths,
                  stratum="health_long_form")
        )
    return dataset


def get_bioasq(size_of_data: float = 1.0, seed: int = 0, split="test",
               hf_dataset: str = "kroshan/BioASQ", data_dir: str = None):
    """BioASQ -- biomedical short-answer factuality; the closest set we have to clinical.

    **BioASQ has no clean canonical presence on the Hub.** Two routes, both awkward, and
    the awkwardness is documented rather than hidden because it determines what the
    biomedical results can claim:

    * ``kroshan/BioASQ`` (default) loads without credentials but has only ``question`` and
      ``text``, where ``text`` packs the answer and its supporting abstract into one
      string as ``"<answer> ... <context> ..."``. We parse those tags out.
    * ``bigbio/bioasq_task_b`` is the official-format loader, but it is a script dataset
      that requires ``trust_remote_code=True`` **and** a manual download after registering
      at bioasq.org. Pass ``hf_dataset="bigbio/bioasq_task_b"`` and ``data_dir=...`` once
      you have the archive.
    """
    hf_split = "validation" if split == "test" else "train"

    if hf_dataset == "bigbio/bioasq_task_b":
        if not data_dir:
            raise ValueError(
                "bigbio/bioasq_task_b is a manual-download dataset: register at "
                "http://bioasq.org, download the Task B archive, and pass data_dir=<path>. "
                "It cannot be fetched automatically and is not redistributable through "
                "this repository. For a credential-free approximation use the default "
                "hf_dataset='kroshan/BioASQ'."
            )
        raw_dataset = load_dataset(
            hf_dataset, split=hf_split, data_dir=data_dir, trust_remote_code=True
        )
        raw_dataset = _subsample(raw_dataset, size_of_data, seed)
        return [
            _item(question=row["question"], ground_truths=list(row["answer"]),
                  stratum="biomedical_factoid")
            for row in tqdm(raw_dataset, desc="Processing BioASQ (official)")
        ]

    raw_dataset = load_dataset(hf_dataset, split=hf_split)
    raw_dataset = _subsample(raw_dataset, size_of_data, seed)

    dataset = []
    skipped = 0
    for row in tqdm(raw_dataset, desc="Processing BioASQ"):
        answer, context = _parse_bioasq_text(row["text"])
        if answer is None:
            skipped += 1
            continue
        dataset.append(
            _item(question=row["question"], ground_truths=[answer], context=context,
                  stratum="biomedical_factoid")
        )
    if skipped:
        # Loud rather than silent: a malformed-row count is the difference between "this
        # slice is small" and "this slice is quietly missing a third of its items".
        print(f"[BioASQ] skipped {skipped} rows with no parseable <answer> tag.")
    return dataset


def _parse_bioasq_text(text: str):
    """Split ``"<answer> A <context> C"`` into (answer, context)."""
    if not text or "<answer>" not in text:
        return None, ""
    _, _, remainder = text.partition("<answer>")
    answer, _, context = remainder.partition("<context>")
    answer = answer.strip()
    return (answer or None), context.strip()
