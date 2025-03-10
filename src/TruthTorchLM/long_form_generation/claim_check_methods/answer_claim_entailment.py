from .claim_check_method import ClaimCheckMethod
from TruthTorchLM.utils import check_entailment
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from TruthTorchLM.normalizers import Normalizer, SigmoidNormalizer
from ..templates import QUESTION_GENERATION_INSTRUCTION, ANSWER_GENERATION_INSTRUCTION

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers import DebertaForSequenceClassification, DebertaTokenizer

import torch
from typing import Union
from copy import deepcopy
from litellm import completion


class AnswerClaimEntailment(ClaimCheckMethod):
    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        num_questions: int,
        num_answers_per_question: int = 3,
        instruction: list = QUESTION_GENERATION_INSTRUCTION,
        first_claim_instruction: list = QUESTION_GENERATION_INSTRUCTION,
        generate_answer_instruction: list = ANSWER_GENERATION_INSTRUCTION,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        entailment_model: PreTrainedModel = None,
        entailment_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ] = None,
        entailment_model_device="cuda",
        normalizer: Normalizer = SigmoidNormalizer(threshold=0, std=1.0),
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.num_questions = num_questions
        self.instruction = instruction
        self.first_claim_instruction = first_claim_instruction
        self.generate_answer_instruction = generate_answer_instruction
        self.num_answers_per_question = num_answers_per_question
        self.entailment_model = entailment_model
        self.entailment_tokenizer = entailment_tokenizer
        self.kwargs = {
            # "max_length": 50,
            "num_return_sequences": 1,
            "seed": 42,
            "do_sample": True,
        }
        self.kwargs.update(kwargs)

        self.normalizer = normalizer

        if type(model) != str:
            self.kwargs.pop("seed", None)
            eos_token_id = self.kwargs.pop("eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = model.config.eos_token_id
            self.kwargs["eos_token_id"] = eos_token_id

            pad_token_id = self.kwargs.pop("pad_token_id", None)
            if pad_token_id is None:
                if type(eos_token_id) == list:
                    pad_token_id = eos_token_id[0]
                else:
                    pad_token_id = eos_token_id
            self.kwargs["pad_token_id"] = pad_token_id
        else:
            self.kwargs.pop("do_sample", None)
            self.kwargs.pop("num_return_sequences", None)
            self.kwargs.pop("max_length", None)

        if self.entailment_model is None or self.entailment_tokenizer is None:
            self.entailment_model = DebertaForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).to(entailment_model_device)
            self.entailment_tokenizer = DebertaTokenizer.from_pretrained(
                "microsoft/deberta-large-mnli"
            )

    def _generate_question(self, claim: str, text_so_far: str, question: str):

        messages = (
            deepcopy(self.first_claim_instruction)
            if text_so_far is None
            else deepcopy(self.instruction)
        )
        messages[-1]["content"] = messages[-1]["content"].format(
            claim=claim, text_so_far=text_so_far, question=question
        )

        if type(self.model) == str:
            response = completion(
                model=self.model, messages=messages, **self.kwargs)
            question = response.choices[0].message["content"]
        else:
            self.tokenizer, messages = fix_tokenizer_chat(
                self.tokenizer, messages)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            generated_output = generate(
                text, self.model, self.tokenizer, **self.kwargs)
            question = generated_output["generated_text_skip_specials"]

        return question.strip()

    def _get_questions(self, question: str, claim: str, text_so_far: str):
        # Generate questions
        questions = []
        question_check = []
        org_seed = self.kwargs.get("seed", None)
        for _ in range(self.num_questions):
            question = self._generate_question(
                claim=claim, text_so_far=text_so_far, question=question
            )
            if question.lower() not in question_check:
                question_check.append(question.lower())
                questions.append(question)
            if type(self.model) == str:
                seed = self.kwargs.pop("seed", None)
                self.kwargs["seed"] = (
                    seed + 1
                )  # Increment seed to get different questions
        if org_seed is not None:
            self.kwargs["seed"] = org_seed
        return questions

    def _does_entail(self, claim: str, question: str, answer: str) -> bool:
        # Check if the question entails the answer
        implication_1 = check_entailment(
            self.entailment_model, self.entailment_tokenizer, question, answer, claim
        )
        implication_2 = check_entailment(
            self.entailment_model, self.entailment_tokenizer, question, claim, answer
        )

        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        implications = [implication_1, implication_2]
        semantically_equivalent = (0 not in implications) and ([
            1, 1] != implications)
        # semantically_equivalent = (implications[0] == 2) and (implications[1] == 2) #strict check
        return semantically_equivalent

    def check_claim_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        claim: str,
        text_so_far: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        messages: list = [],
        context:str="",
        **kwargs,
    ):

        questions = self._get_questions(
            question=question, claim=claim, text_so_far=text_so_far
        )
        # Get model answers for each question (generate answers until it entails the claim)
        answers = []
        entailment = []
        for i, question in enumerate(questions):
            messages = deepcopy(self.generate_answer_instruction)
            messages[-1]["content"] = messages[-1]["content"].format(
                question=question)
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            # check if the answer aligns with the claim
            for _ in range(self.num_answers_per_question):
                generated_output = generate(text, model, tokenizer, **kwargs)
                answer = generated_output["generated_text_skip_specials"]
                answers.append(answer)
                del generated_output

                if self._does_entail(claim=claim, question=question, answer=answer):
                    entailment.append(1)
                else:
                    entailment.append(0)

        return {
            "claim": claim,
            "normalized_truth_values": self.normalizer(
                sum(entailment) / len(entailment)
            ),
            "truth_values": sum(entailment) / len(entailment),
            "questions": questions,
            "answers": answers,
            "entailment": entailment,
        }

    def check_claim_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        claim: str,
        text_so_far: str,
        generation_seed=None,
        context:str="",
        **kwargs,
    ):

        questions = self._get_questions(
            question=question, claim=claim, text_so_far=text_so_far
        )
        answers = []
        entailment = []
        for i, question in enumerate(questions):
            q_messages = deepcopy(self.generate_answer_instruction)
            q_messages[-1]["content"] = q_messages[-1]["content"].format(
                question=question
            )
            # check if the answer aligns with the claim
            for _ in range(self.num_answers_per_question):
                response = completion(
                    model=model, messages=q_messages, **kwargs)
                answer = response.choices[0].message["content"]
                answers.append(answer)
                if self._does_entail(claim=claim, question=question, answer=answer):
                    entailment.append(1)
                else:
                    entailment.append(0)

        return {
            "claim": claim,
            "normalized_truth_values": self.normalizer(
                sum(entailment) / len(entailment)
            ),
            "truth_values": sum(entailment) / len(entailment),
            "questions": questions,
            "answers": answers,
            "entailment": entailment,
        }

    def __str__(self):

        model_name = self.model.__class__ if type(
            self.model) != str else self.model
        ent_model_name = (
            self.entailment_model.__class__
            if type(self.entailment_model) != str
            else self.entailment_model
        )

        return f"Claim Check Method by Entailment Check with Answers and claim. Answers generated to the generated questions.\n\
Question generation model: {model_name}\n\
Number of questions to be generated for a stament: {self.num_questions}\n\
Number of answers per question: {self.num_answers_per_question}\n\
Entailment check model: {ent_model_name}\n\n\
Question generation instruction for the first claim:\n  {self.first_claim_instruction}\n\n\
Question generation instruction for claims with preceeding text:\n  {self.instruction}\n\n\
Answer generation instruction:\n    {self.generate_answer_instruction}"
