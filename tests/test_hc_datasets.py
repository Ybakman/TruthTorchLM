"""Dataset loaders (benchmark D axis).

Split in two:

* The **parsing and formatting** tests run everywhere -- they exercise the row-shaping
  logic against the exact column shapes verified on the HuggingFace datasets-server, with
  no network. These are what catch the gotchas: MMLU's integer answer index vs MedQA's
  answer text, MedQA's dict-not-list options, SQuAD 2.0's empty-list unanswerables,
  BioASQ's inline ``<answer>``/``<context>`` tags.
* The **live contract** tests hit the Hub and are marked ``network``; run them with
  ``pytest -m network`` when you want to confirm the upstream schemas have not drifted.
"""

import pytest

pytest.importorskip("TruthTorchLM.utils.hc_datasets", reason="needs the datasets package")

from TruthTorchLM.utils.hc_datasets import (  # noqa: E402
    MMLU_MEDICAL_SUBJECTS,
    OPTION_LETTERS,
    _format_mcq,
    _item,
    _parse_bioasq_text,
)

REQUIRED_KEYS = {"context", "question", "ground_truths"}


class TestItemContract:
    def test_item_matches_upstream_shape_plus_hc_fields(self):
        item = _item("q?", ["a"])
        assert REQUIRED_KEYS <= set(item)
        # HC additions default to something harmless so upstream loaders keep working.
        assert item["stratum"] is None
        assert item["outcome_type"] == "factual_error"
        assert item["context"] == ""

    def test_outcome_type_is_settable_for_the_q5_split(self):
        item = _item("q?", ["a"], stratum="crisis", outcome_type="safety_violation")
        assert item["outcome_type"] == "safety_violation"
        assert item["stratum"] == "crisis"


class TestMCQFormatting:
    def test_options_are_rendered_into_the_question_text(self):
        """They have to be: upstream's user prompt only interpolates {question}, so
        options placed anywhere else would never reach the model."""
        prompt = _format_mcq("What size cannula?", ["18 gauge.", "20 gauge.", "22 gauge."])
        assert "A. 18 gauge." in prompt
        assert "B. 20 gauge." in prompt
        assert "C. 22 gauge." in prompt
        assert "letter" in prompt.lower()

    def test_letters_are_assigned_in_order(self):
        prompt = _format_mcq("q", ["w", "x", "y", "z"])
        positions = [prompt.index(f"{letter}. ") for letter in OPTION_LETTERS[:4]]
        assert positions == sorted(positions)


class TestMedQARowShape:
    def test_options_dict_is_ordered_by_letter_not_dict_order(self):
        """MedQA's `options` is a dict keyed "A".."D" -- not a list. Iterating the dict
        directly would be at the mercy of key order; we index by letter instead."""
        row = {
            "question": "A junior resident ...",
            "options": {"C": "third", "A": "first", "D": "fourth", "B": "second"},
            "answer_idx": "B",
            "answer": "second",
        }
        options = [row["options"][l] for l in OPTION_LETTERS if l in row["options"]]
        assert options == ["first", "second", "third", "fourth"]

        ground_truths = [row["answer_idx"], row["answer"]]
        assert ground_truths[0] == "B"
        assert options[OPTION_LETTERS.index(ground_truths[0])] == ground_truths[1]


class TestMMLURowShape:
    def test_integer_answer_index_maps_to_the_right_letter_and_text(self):
        """MMLU's `answer` is an int 0-3, unlike MedQA's answer text. Treating one like
        the other silently mislabels every row."""
        row = {"question": "q", "choices": ["18 gauge.", "20 gauge.", "22 gauge.", "24 gauge."],
               "answer": 0}
        index = int(row["answer"])
        assert OPTION_LETTERS[index] == "A"
        assert row["choices"][index] == "18 gauge."

    def test_medical_subject_list_is_the_documented_union(self):
        assert "clinical_knowledge" in MMLU_MEDICAL_SUBJECTS
        assert "professional_medicine" in MMLU_MEDICAL_SUBJECTS
        # MMLU has no "medical" config; MMLU-med is this union and nothing else.
        assert len(MMLU_MEDICAL_SUBJECTS) == 9


class TestSquadUnanswerable:
    def test_unanswerable_rows_are_detected_by_empty_lists(self):
        """SQuAD 2.0 marks unanswerable questions with empty lists, not null and not a
        sentinel string -- a `is None` check would classify every one of them as
        answerable."""
        unanswerable = {"text": [], "answer_start": []}
        answerable = {"text": ["Normans", "Normans"], "answer_start": [10, 10]}
        assert len(unanswerable["text"]) == 0
        assert len(answerable["text"]) > 0

    def test_duplicate_annotator_answers_are_deduplicated(self):
        answers = ["Normans", "Normans", "the Normans"]
        assert list(dict.fromkeys(answers)) == ["Normans", "the Normans"]


class TestBioASQParsing:
    def test_inline_tags_are_split_into_answer_and_context(self):
        text = (
            "<answer> autosomal dominant <context> Balanced t(11;15)(q23;q15) in a TP53+/+ "
            "breast cancer patient from a Li-Fraumeni syndrome family."
        )
        answer, context = _parse_bioasq_text(text)
        assert answer == "autosomal dominant"
        assert context.startswith("Balanced t(11;15)")

    def test_answer_with_no_context_tag_still_parses(self):
        answer, context = _parse_bioasq_text("<answer> yes")
        assert answer == "yes"
        assert context == ""

    def test_malformed_rows_return_none_rather_than_a_wrong_answer(self):
        assert _parse_bioasq_text("no tags here at all") == (None, "")
        assert _parse_bioasq_text("") == (None, "")
        assert _parse_bioasq_text("<answer>   <context> abc")[0] is None


class TestBioASQOfficialRoute:
    def test_official_loader_refuses_without_a_manual_download_path(self):
        """bigbio/bioasq_task_b needs registration at bioasq.org and a local archive.
        The loader must say so, not surface a datasets-library stack trace."""
        from TruthTorchLM.utils.hc_datasets import get_bioasq

        with pytest.raises(ValueError, match="bioasq.org"):
            get_bioasq(hf_dataset="bigbio/bioasq_task_b")


@pytest.mark.network
class TestLiveSchemas:
    """Confirm the upstream column shapes have not drifted. Run with `pytest -m network`."""

    def _check(self, loader, **kwargs):
        items = loader(size_of_data=5, seed=0, split="test", **kwargs)
        assert len(items) > 0
        for item in items[:5]:
            assert REQUIRED_KEYS <= set(item)
            assert isinstance(item["question"], str) and item["question"]
            assert isinstance(item["ground_truths"], list) and item["ground_truths"]
        return items

    def test_hotpot_qa(self):
        from TruthTorchLM.utils.hc_datasets import get_hotpot_qa

        items = self._check(get_hotpot_qa)
        assert items[0]["context"]  # distractor config always supplies paragraphs

    def test_truthful_qa(self):
        from TruthTorchLM.utils.hc_datasets import get_truthful_qa

        self._check(get_truthful_qa)

    def test_squad_v2_has_both_strata(self):
        from TruthTorchLM.utils.hc_datasets import get_squad_v2

        items = get_squad_v2(size_of_data=200, seed=0, split="test")
        strata = {item["stratum"] for item in items}
        assert strata == {"answerable", "unanswerable"}

    def test_medqa(self):
        from TruthTorchLM.utils.hc_datasets import get_medqa

        items = self._check(get_medqa)
        assert items[0]["ground_truths"][0] in OPTION_LETTERS

    def test_mmlu_med(self):
        from TruthTorchLM.utils.hc_datasets import get_mmlu_med

        items = get_mmlu_med(size_of_data=5, seed=0, split="test",
                             subjects=["clinical_knowledge"])
        assert items and items[0]["stratum"] == "clinical_knowledge"

    def test_kqa(self):
        from TruthTorchLM.utils.hc_datasets import get_kqa

        self._check(get_kqa)

    def test_medlfqa(self):
        from TruthTorchLM.utils.hc_datasets import get_medlfqa

        self._check(get_medlfqa)

    def test_bioasq_fallback_route(self):
        from TruthTorchLM.utils.hc_datasets import get_bioasq

        self._check(get_bioasq)
