from .claim_check_method import ClaimCheckMethod
from TruthTorchLM.truth_methods import TruthMethod
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat
from TruthTorchLM.generation import (
    get_sampling_properties,
    sample_generations_hf_local,
    sample_generations_api,
)
from ..templates import ANSWER_GENERATION_INSTRUCTION

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel

import torch
from typing import Union
from copy import deepcopy


class NaiveApplication(ClaimCheckMethod):
    def __init__(
        self,
        generate_answer_instruction: list = ANSWER_GENERATION_INSTRUCTION,
        truth_methods: list[TruthMethod] = None,
        batch_generation: bool = True,
        use_question: bool = True,
    ):
        super().__init__()

        self.generate_answer_instruction = generate_answer_instruction
        self.truth_methods = truth_methods
        self.batch_generation = batch_generation
        self.use_question = use_question

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
        context:str="",
        **kwargs,
    ):

        question = question if self.use_question else ""
        q_messages = deepcopy(self.generate_answer_instruction)
        q_messages[-1]["content"] = q_messages[-1]["content"].format(
            question=question)
        tokenizer, q_messages = fix_tokenizer_chat(tokenizer, q_messages)
        text = tokenizer.apply_chat_template(
            q_messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        q_messages.append({"role": "assistant", "content": claim})
        tokenizer, q_messages = fix_tokenizer_chat(tokenizer, q_messages)
        text_messsages = tokenizer.apply_chat_template(
            q_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        model_outputs = tokenizer.encode(text_messsages, return_tensors="pt").to(
            model.device
        )

        t_messages = deepcopy(self.generate_answer_instruction)
        t_messages[-1]["content"] = t_messages[-1]["content"].format(
            question=question)
        normalized_truth_values, unnormalized_truth_values, method_spec_outputs = (
            self._get_truth_value_local(
                self.truth_methods,
                model=model,
                tokenizer=tokenizer,
                question=question,
                text=text,
                answer=claim,
                model_output=model_outputs,
                generation_seed=generation_seed,
                messages=t_messages,
                context=context,
                **kwargs,
            )
        )
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {
                "Truth method name": self.truth_methods[i].__class__.__name__
            }
            method_spec_outputs[i].pop("generated_text", None)
            output_dict["detailed_outputs"] = method_spec_outputs[i]
            final_method_specific_outputs.append(output_dict)

        return {
            "claim": claim,
            "normalized_truth_values": normalized_truth_values,
            "truth_values": unnormalized_truth_values,
            "question": question,
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
                context=context,
                **kwargs,
            )
            normalized_truth_values.append(
                truth_values["normalized_truth_value"])
            unnormalized_truth_values.append(truth_values["truth_value"])
            method_spec_outputs.append(truth_values)

        return normalized_truth_values, unnormalized_truth_values, method_spec_outputs

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

        requires_logprobs = False
        for truth_method in self.truth_methods:
            if truth_method.REQUIRES_LOGPROBS:
                requires_logprobs = True
                print(
                    f"Truth method '{truth_method.__class__.__name__}' requires logprobs."
                )

        if requires_logprobs:
            raise ValueError(
                "Truth methods requiring logprobs cannot be used with NaiveApplication claim check method."
            )

        q_messages = deepcopy(self.generate_answer_instruction)
        question = question if self.use_question else ""
        # Get truth value for truth method
        q_messages[-1]["content"] = q_messages[-1]["content"].format(
            question=question)
        normalized_truth_values, unnormalized_truth_values, method_spec_outputs = (
            self._get_truth_value_api(
                self.truth_methods,
                model=model,
                q_messages=q_messages,
                question=question,
                answer=claim,
                generation_seed=generation_seed,
                context=context,
                **kwargs,
            )
        )
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {
                "Truth method name": self.truth_methods[i].__class__.__name__
            }
            method_spec_outputs[i].pop("generated_text", None)
            output_dict["detailed_outputs"] = method_spec_outputs[i]
            final_method_specific_outputs.append(output_dict)

        return {
            "claim": claim,
            "normalized_truth_values": normalized_truth_values,
            "truth_values": unnormalized_truth_values,
            "question": question,
            "truth_method_spec_outputs": final_method_specific_outputs,
        }

    def __str__(self):

        return f"Claim Check Method by using the orginal question and the claim.\n\
Use original question (if false, empty string is used as question): {self.use_question}\n\
Answer generation instruction (used as the template for original question - claim pair):\n    {self.generate_answer_instruction}\n\n\
Truth methods to assign a score the question(s):\n   {self.truth_methods}"
