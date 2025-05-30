from .claim_check_method import ClaimCheckMethod
from TruthTorchLM.utils import check_entailment
from TruthTorchLM.truth_methods import TruthMethod
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from TruthTorchLM.generation import (
    get_sampling_properties,
    sample_generations_hf_local,
    sample_generations_api,
)
from TruthTorchLM.error_handler import handle_logprobs_error
from ..templates import QUESTION_GENERATION_INSTRUCTION, ANSWER_GENERATION_INSTRUCTION

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers import DebertaForSequenceClassification, DebertaTokenizer

import torch
from typing import Union
from copy import deepcopy
from litellm import completion
import numpy as np


class QuestionAnswerGeneration(ClaimCheckMethod):
    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        num_questions: int,
        max_answer_trials: int = 3,
        aggregation_strategy: str = "max",  # can be avg, min, or max
        instruction: list = QUESTION_GENERATION_INSTRUCTION,
        first_claim_instruction: list = QUESTION_GENERATION_INSTRUCTION,
        generate_answer_instruction: list = ANSWER_GENERATION_INSTRUCTION,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        truth_methods: list[TruthMethod] = None,
        entailment_model: PreTrainedModel = None,
        entailment_tokenizer: Union[
            PreTrainedTokenizer, PreTrainedTokenizerFast
        ] = None,
        batch_generation: bool = True,
        entailment_model_device="cuda",
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.num_questions = num_questions
        self.instruction = instruction
        self.first_claim_instruction = first_claim_instruction
        self.generate_answer_instruction = generate_answer_instruction
        self.max_answer_trials = max_answer_trials
        self.truth_methods = truth_methods
        self.entailment_model = entailment_model
        self.entailment_tokenizer = entailment_tokenizer
        self.batch_generation = batch_generation
        self.kwargs = {
            # "max_length": 50,
            "num_return_sequences": 1,
            "do_sample": True,
        }
        self.kwargs.update(kwargs)

        if aggregation_strategy.lower() == "min":
            self.aggregation_strategy = np.min
        elif aggregation_strategy.lower() == "max":
            self.aggregation_strategy = np.max
        elif aggregation_strategy.lower() == "avg":
            self.aggregation_strategy = np.mean
        else:
            raise ValueError(
                f"aggregation strategy {aggregation_strategy} is not supported. Choose from ['min', 'max', 'avg']"
            )

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
            gen_question = response.choices[0].message["content"]
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
            gen_question = generated_output["generated_text_skip_specials"]

        return gen_question.strip()

    def _get_questions(self, question: str, claim: str, text_so_far: str):
        # Generate questions
        questions = []
        question_check = []
        org_seed = self.kwargs.get("seed", None)
        for _ in range(self.num_questions):
            gen_question = self._generate_question(
                claim=claim, text_so_far=text_so_far, question=question
            )
            if gen_question.lower() not in question_check:
                question_check.append(gen_question.lower())
                questions.append(gen_question)
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

    def _get_truth_value_local(
        self,
        truth_methods,
        model,
        tokenizer,
        question,
        text,
        answer,
        model_output,
        generation_seed,
        messages,
        context,
        **kwargs,
    ):

        (
            number_of_generations,
            return_text,
            return_logits,
            return_logprobs,
            return_attentions,
            return_activations,
        ) = get_sampling_properties(truth_methods)

        sampled_gen_dict = sample_generations_hf_local(
            model,
            text,
            tokenizer,
            generation_seed,
            number_of_generations=number_of_generations,
            return_text=return_text,
            return_logits=return_logits,
            return_logprobs=return_logprobs,
            return_attentions=return_attentions,
            return_activations=return_activations,
            batch_generation=self.batch_generation,
            **kwargs,
        )

        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for truth_method in truth_methods:
            truth_values = truth_method(
                model=model,
                input_text=text,
                generated_text=answer,
                question=question,
                all_ids=model_output,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                sampled_generations_dict=sampled_gen_dict,
                messages=messages,
                context=context,
                **kwargs,
            )
            normalized_truth_values.append(
                truth_values["normalized_truth_value"])
            unnormalized_truth_values.append(truth_values["truth_value"])
            method_spec_outputs.append(truth_values)

        return normalized_truth_values, unnormalized_truth_values, method_spec_outputs

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
        context:str = "",
        **kwargs,
    ):
        main_question = question

        all_questions = self._get_questions(
            question=main_question, claim=claim, text_so_far=text_so_far
        )

        # Get model answers for each question (generate answers until it entails the claim)
        questions = []
        answers = []
        texts = []
        model_outputs = []
        failed_tuples = []
        for i, question in enumerate(all_questions):
            q_messages = deepcopy(self.generate_answer_instruction)
            q_messages[-1]["content"] = q_messages[-1]["content"].format(
                question=question
            )

            tokenizer, q_messages = fix_tokenizer_chat(tokenizer, q_messages)
            text = tokenizer.apply_chat_template(
                q_messages,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            # check if the answer aligns with the claim
            for _ in range(self.max_answer_trials):
                generated_output = generate(text, model, tokenizer, **kwargs)
                answer = generated_output["generated_text_skip_specials"]
                model_output = generated_output["all_ids"]
                del generated_output

                if self._does_entail(claim=claim, question=question, answer=answer):
                    questions.append(question)
                    answers.append(answer)
                    texts.append(text)
                    model_outputs.append(model_output)
                    break
                failed_tuples.append((question, answer))

        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, text, answer, model_output in zip(
            questions, texts, answers, model_outputs
        ):
            t_messages = deepcopy(self.generate_answer_instruction)
            t_messages[-1]["content"] = t_messages[-1]["content"].format(
                question=question
            )
            normalized_truth_value, unnormalized_truth_value, method_spec_output = (
                self._get_truth_value_local(
                    self.truth_methods,
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    text=text,
                    answer=answer,
                    model_output=model_output,
                    generation_seed=generation_seed,
                    messages=t_messages,
                    context=context,
                    **kwargs,
                )
            )
            normalized_truth_values.append(normalized_truth_value)
            unnormalized_truth_values.append(unnormalized_truth_value)
            method_spec_outputs.append(method_spec_output)

        if len(normalized_truth_values) == 0:
            normalized_truth_values = [[0] * len(self.truth_methods)]
            unnormalized_truth_values = [[-1e10] * len(self.truth_methods)]
            method_spec_outputs = [[{}] * len(self.truth_methods)]

        final_normalized_truth_values = []
        final_unnormalized_truth_values = []
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {
                "Truth method name": self.truth_methods[i].__class__.__name__,
                "detailed_outputs": [],
            }
            total = []
            for truth_values in normalized_truth_values:
                total.append(truth_values[i])
            final_normalized_truth_values.append(
                self.aggregation_strategy(total))
            total = []
            for truth_values in unnormalized_truth_values:
                total.append(truth_values[i])
            final_unnormalized_truth_values.append(
                self.aggregation_strategy(total))
            for outputs in method_spec_outputs:
                outputs[i].pop("generated_text", None)
                output_dict["detailed_outputs"].append(outputs[i])
            final_method_specific_outputs.append(output_dict)

        return {
            "claim": claim,
            "normalized_truth_values": final_normalized_truth_values,
            "truth_values": final_unnormalized_truth_values,
            "questions": questions,
            "answers": answers,
            "non_entailment_tuples": failed_tuples,
            "truth_method_spec_outputs": final_method_specific_outputs,
        }

    def _get_truth_value_api(
        self,
        truth_methods,
        model,
        q_messages,
        question,
        answer,
        generation_seed,
        logprobs,
        generated_tokens,
        context,
        **kwargs,
    ):

        # Get sampled generations to be used in truth methods
        (
            number_of_generations,
            return_text,
            return_logits,
            return_logprobs,
            return_attentions,
            return_activations,
        ) = get_sampling_properties(truth_methods)
        sampled_gen_dict = sample_generations_api(
            model,
            q_messages,
            generation_seed,
            number_of_generations=number_of_generations,
            return_text=return_text,
            return_logits=return_logits,
            return_logprobs=return_logprobs,
            return_attentions=return_attentions,
            return_activations=return_activations,
            **kwargs,
        )

        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for truth_method in truth_methods:
            truth_values = truth_method(
                model=model,
                messages=q_messages,
                generated_text=answer,
                question=question,
                generation_seed=generation_seed,
                sampled_generations_dict=sampled_gen_dict,
                logprobs=logprobs,
                generated_tokens=generated_tokens,
                context=context,
                **kwargs,
            )
            normalized_truth_values.append(
                truth_values["normalized_truth_value"])
            unnormalized_truth_values.append(truth_values["truth_value"])
            method_spec_outputs.append(truth_values)

        return normalized_truth_values, unnormalized_truth_values, method_spec_outputs

    @handle_logprobs_error
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
        main_question = question
        all_questions = self._get_questions(
            question=main_question, claim=claim, text_so_far=text_so_far
        )

        requires_logprobs = False
        for truth_method in self.truth_methods:
            if truth_method.REQUIRES_LOGPROBS:
                requires_logprobs = True

        # Get model answers for each question (generate answers until it entails the claim)
        questions = []
        answers = []
        logprobs = []
        generated_tokens = []
        failed_tuples = []

        for i, question in enumerate(all_questions):
            q_messages = deepcopy(self.generate_answer_instruction)
            q_messages[-1]["content"] = q_messages[-1]["content"].format(
                question=question
            )

            # check if the answer aligns with the claim
            for _ in range(self.max_answer_trials):
                response = completion(
                    model=model,
                    messages=q_messages,
                    logprobs=requires_logprobs,
                    **kwargs,
                )
                answer = response.choices[0].message["content"]
                if self._does_entail(claim=claim, question=question, answer=answer):
                    questions.append(question)
                    answers.append(answer)
                    logprobs.append(
                        [
                            token["logprob"]
                            for token in response.choices[0].logprobs["content"]
                        ]
                        if requires_logprobs
                        else None
                    )
                    generated_tokens.append(
                        [
                            token["token"]
                            for token in response.choices[0].logprobs["content"]
                        ]
                        if requires_logprobs
                        else None
                    )
                    break
                failed_tuples.append((question, answer))

        # Get truth value for truth method
        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, answer, logprob, tokens in zip(
            questions, answers, logprobs, generated_tokens
        ):
            q_messages = deepcopy(self.generate_answer_instruction)
            q_messages[-1]["content"] = q_messages[-1]["content"].format(
                question=question
            )
            normalized_truth_value, unnormalized_truth_value, method_spec_output = (
                self._get_truth_value_api(
                    self.truth_methods,
                    model=model,
                    q_messages=q_messages,
                    question=question,
                    answer=answer,
                    generation_seed=generation_seed,
                    logprobs=logprob,
                    generated_tokens=tokens,
                    context=context,
                    **kwargs,
                )
            )
            normalized_truth_values.append(normalized_truth_value)
            unnormalized_truth_values.append(unnormalized_truth_value)
            method_spec_outputs.append(method_spec_output)

        if len(normalized_truth_values) == 0:
            normalized_truth_values = [[0] * len(self.truth_methods)]
            unnormalized_truth_values = [[-1e10] * len(self.truth_methods)]
            method_spec_outputs = [[{}] * len(self.truth_methods)]

        final_normalized_truth_values = []
        final_unnormalized_truth_values = []
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {
                "Truth method name": self.truth_methods[i].__class__.__name__,
                "detailed_outputs": [],
            }
            total = []
            for truth_values in normalized_truth_values:
                total.append(truth_values[i])
            final_normalized_truth_values.append(
                self.aggregation_strategy(total))
            total = []
            for truth_values in unnormalized_truth_values:
                total.append(truth_values[i])
            final_unnormalized_truth_values.append(
                self.aggregation_strategy(total))
            for outputs in method_spec_outputs:
                outputs[i].pop("generated_text", None)
                output_dict["detailed_outputs"].append(outputs[i])
            final_method_specific_outputs.append(output_dict)

        return {
            "claim": claim,
            "normalized_truth_values": final_normalized_truth_values,
            "truth_values": final_unnormalized_truth_values,
            "questions": questions,
            "answers": answers,
            "non_entailment_tuples": failed_tuples,
            "truth_method_spec_outputs": final_method_specific_outputs,
        }

    def __str__(self):

        model_name = self.model.__class__ if type(
            self.model) != str else self.model
        ent_model_name = (
            self.entailment_model.__class__
            if type(self.entailment_model) != str
            else self.entailment_model
        )

        return f"Claim Check Method by Generating Questions and Answers.\n\
Question generation model: {model_name}\n\
Number of questions to be generated for a stament: {self.num_questions}\n\
Number of trials to generate an answer that entails with the claim: {self.max_answer_trials}\n\
Entailment check model: {ent_model_name}\n\n\
Question generation instruction for the first claim:\n  {self.first_claim_instruction}\n\n\
Question generation instruction for claims with preceeding text:\n  {self.instruction}\n\n\
Answer generation instruction:\n    {self.generate_answer_instruction}\n\n\
Truth methods to assign a score the question(s):\n   {self.truth_methods}"
