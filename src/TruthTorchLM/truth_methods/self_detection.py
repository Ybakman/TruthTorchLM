import copy
import torch
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLM.utils import *
from .truth_method import TruthMethod
from TruthTorchLM.templates import (
    SELF_DETECTION_QUESTION_PROMPT,
    SELF_DETECTION_SYSTEM_PROMPT,
    ENTAILMENT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    SELF_DETECTION_USER_PROMPT,
)
from TruthTorchLM.generation import sample_generations_hf_local, sample_generations_api


# https://arxiv.org/pdf/2310.17918


class SelfDetection(TruthMethod):
    def __init__(
        self,
        output_type: str = "entropy",
        method_for_similarity: str = "semantic",
        number_of_questions=5,
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        prompt_for_generating_question=SELF_DETECTION_QUESTION_PROMPT,
        system_prompt=SELF_DETECTION_SYSTEM_PROMPT,
        prompt_for_entailment: str = ENTAILMENT_PROMPT,
        system_prompt_for_entailment: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt=SELF_DETECTION_USER_PROMPT,
        entailment_model_device="cuda",
        batch_generation=True,
        question_max_new_tokens=128,
        question_temperature=1.0,
        **generation_kwargs
    ):
        super().__init__()

        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment

        if (
            model_for_entailment is None or tokenizer_for_entailment is None
        ) and method_for_similarity == "semantic":
            self.model_for_entailment = (
                DebertaForSequenceClassification.from_pretrained(
                    "microsoft/deberta-large-mnli"
                ).to(entailment_model_device)
            )
            self.tokenizer_for_entailment = DebertaTokenizer.from_pretrained(
                "microsoft/deberta-large-mnli"
            )

        self.number_of_questions = number_of_questions
        self.prompt_for_generating_question = prompt_for_generating_question
        self.system_prompt = system_prompt
        self.prompt_for_entailment = prompt_for_entailment
        self.system_prompt_for_entailment = system_prompt_for_entailment
        self.batch_generation = batch_generation
        self.question_max_new_tokens = question_max_new_tokens
        self.question_temperature = question_temperature
        self.generation_kwargs = generation_kwargs
        self.user_prompt = user_prompt
        if output_type not in ["entropy", "consistency"]:
            raise ValueError(
                "output_type should be either 'entropy' or 'consistency'")
        self.output_type = output_type

        if method_for_similarity not in ["generation", "semantic", "jaccard"]:
            raise ValueError(
                "method_for_similarity should be either 'generation' or 'semantic' or 'jaccard'"
            )
        self.method_for_similarity = method_for_similarity

    def generate_similar_questions(
        self,
        question: str,
        prompt_for_generating_question: str = None,
        system_prompt: str = None,
        model=None,
        tokenizer=None,
        generation_seed=0,
    ):
        generated_questions = [question]
        previous_questions = [question]
        for i in range(self.number_of_questions - 1):
            if self.system_prompt is not None:
                chat = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self.prompt_for_generating_question.format(
                            question=question,
                            previous_questions=previous_questions,
                        ),
                    },
                ]
            else:
                chat = [
                    {
                        "role": "user",
                        "content": self.prompt_for_generating_question.format(
                            question=question,
                            previous_questions=previous_questions,
                        ),
                    }
                ]
            if type(model) != str:
                tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
                input_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    continue_final_message=False,
                )
                sampled_generations_dict = sample_generations_hf_local(
                    model,
                    input_text,
                    tokenizer,
                    number_of_generations=1,
                    return_text=True,
                    generation_seed=generation_seed,
                    max_new_tokens=self.question_max_new_tokens,
                    temperature=self.question_temperature,
                    batch_generation=self.batch_generation,
                    **self.generation_kwargs
                )
            if type(model) == str:
                sampled_generations_dict = sample_generations_api(
                    model,
                    chat,
                    number_of_generations=1,
                    return_text=True,
                    generation_seed=generation_seed,
                    temperature=self.question_temperature,
                )

            previous_questions.append(
                sampled_generations_dict["generated_texts"][0])
            generated_questions.append(
                sampled_generations_dict["generated_texts"][0])
        return generated_questions

    def _self_detection_output(
        self,
        model,
        tokenizer,
        generated_texts: list,
        question: str,
        generated_questions,
    ):
        if self.method_for_similarity == "semantic":
            clusters = bidirectional_entailment_clustering(
                self.model_for_entailment,
                self.tokenizer_for_entailment,
                question,
                generated_texts,
                self.method_for_similarity,
                entailment_prompt=self.prompt_for_entailment,
                system_prompt=self.system_prompt_for_entailment,
            )
        else:
            clusters = bidirectional_entailment_clustering(
                model,
                tokenizer,
                question,
                generated_texts,
                self.method_for_similarity,
                entailment_prompt=self.prompt_for_entailment,
                system_prompt=self.system_prompt_for_entailment,
            )

        entropy = 0
        for cluster in clusters:
            entropy -= (
                len(cluster)
                / self.number_of_questions
                * np.log(len(cluster) / self.number_of_questions)
            )
        consistency = (len(clusters[0]) - 1) / (self.number_of_questions - 1)

        if self.output_type == "entropy":
            truth_value = -entropy
        elif self.output_type == "consistency":
            truth_value = consistency

        return {
            "truth_value": truth_value,
            "entropy": entropy,
            "consistency": consistency,
            "generated_questions": generated_questions,
            "generated_texts": generated_texts,
            "clusters": clusters,
        }

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        messages: list = [],
        context: str = "",
        **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        kwargs.pop("do_sample", None)
        kwargs.pop("num_return_sequences", None)
        generated_questions = self.generate_similar_questions(
            question=question,
            prompt_for_generating_question=self.prompt_for_generating_question,
            model=model,
            tokenizer=tokenizer,
            generation_seed=generation_seed,
        )

        for generated_question in generated_questions:
            chat = messages.copy()
            chat[-1] = {
                "role": "user",
                "content": self.user_prompt.format(question=generated_question),
            }
            tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
            prompt = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt").to(model.device)
            model_output = model.generate(
                input_ids, num_return_sequences=1, do_sample=True, **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return self._self_detection_output(
            model, tokenizer, generated_texts, question, generated_questions
        )

    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs
    ):

        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        generated_questions = self.generate_similar_questions(
            question=question,
            prompt_for_generating_question=self.prompt_for_generating_question,
            system_prompt=self.system_prompt,
            model=model,
            generation_seed=generation_seed,
        )

        for generated_question in generated_questions:
            chat = messages.copy()
            chat[-1] = {
                "role": "user",
                "content": self.user_prompt.format(question=generated_question),
            }
            response = completion(model=model, messages=chat, **kwargs)
            generated_texts.append(response.choices[0].message["content"])

        return self._self_detection_output(
            model, None, generated_texts, question, generated_questions
        )
