import torch
import random
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from TruthTorchLM.long_form_generation.decomposition_methods.decomposition_method import (
    DecompositionMethod,
)
from TruthTorchLM.long_form_generation.claim_check_methods.claim_check_method import (
    ClaimCheckMethod,
)
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from TruthTorchLM.error_handler import handle_logprobs_error


def long_form_generation_with_truth_value(
    model: PreTrainedModel,
    messages: list,
    question: str = None,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    decomp_method: DecompositionMethod = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    generation_seed=None,
    add_generation_prompt=True,
    continue_final_message=False,
    context:str="",
    **kwargs
) -> dict:
    if type(model) == str:
        return long_form_generation_with_truth_value_api(
            model=model,
            messages=messages,
            question=question,
            decomp_method=decomp_method,
            claim_check_methods=claim_check_methods,
            generation_seed=generation_seed,
            context=context,
            **kwargs
        )
    else:
        return long_form_generation_with_truth_value_hf_local(
            model=model,
            messages=messages,
            question=question,
            decomp_method=decomp_method,
            claim_check_methods=claim_check_methods,
            tokenizer=tokenizer,
            generation_seed=generation_seed,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            context=context,
            **kwargs
        )


# add cleaning function for the generated text
def long_form_generation_with_truth_value_hf_local(
    model: PreTrainedModel,
    messages: list,
    question: str = None,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    decomp_method: DecompositionMethod = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    generation_seed=None,
    add_generation_prompt=True,
    continue_final_message=False,
    context:str="",
    **kwargs
) -> dict:

    if question == None:
        question = ""
        # search over last user message if exists
        for message in messages[::-1]:
            if message["role"] == "user":
                question = message["content"]
                break

    eos_token_id = kwargs.pop("eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    kwargs["eos_token_id"] = eos_token_id

    pad_token_id = kwargs.pop("pad_token_id", None)
    if pad_token_id is None:
        if type(eos_token_id) == list:
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id
    kwargs["pad_token_id"] = pad_token_id

    # adjust seeds
    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        random.seed(generation_seed)

    # Generate the main output
    tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )
    generated_output = generate(text, model, tokenizer, **kwargs)
    generated_text = generated_output["generated_text_skip_specials"]
    model_output = generated_output["all_ids"]
    del generated_output

    # Factual Decomposition
    print("Decomposing the generated text...")
    claims = decomp_method(generated_text)

    # Get truth score for each claim.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for claim_check_method in claim_check_methods:
        print("Applying claim check method ",
              claim_check_method.__class__.__name__)
        stmt_normalized_truth_values = []
        stmt_unnormalized_truth_values = []
        stmt_method_spec_outputs = []
        for sidx, claim in enumerate(claims):
            text_so_far = " ".join(claims[:sidx]) if sidx > 0 else None
            truth_values = claim_check_method(
                model=model,
                input_text=text,
                generated_text=generated_text,
                question=question,
                claim=claim,
                text_so_far=text_so_far,
                all_ids=model_output,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                messages=messages,
                context=context,                
                **kwargs
            )

            stmt_normalized_truth_values.append(
                truth_values["normalized_truth_values"])
            stmt_unnormalized_truth_values.append(truth_values["truth_values"])
            stmt_method_spec_outputs.append(truth_values)
        normalized_truth_values.append(stmt_normalized_truth_values)
        unnormalized_truth_values.append(stmt_unnormalized_truth_values)
        method_spec_outputs.append(
            {
                "claim_check_method_name": claim_check_method.__class__.__name__,
                "details": stmt_method_spec_outputs,
            }
        )

    # Create TruthObject
    truth_dict = {
        "generated_text": generated_text,
        "claims": claims,
        "normalized_truth_values": normalized_truth_values,
        "unnormalized_truth_values": unnormalized_truth_values,
        "claim_check_method_details": method_spec_outputs,
    }

    # Return TruthObject
    return truth_dict


# for api-based models, we should write a wrapper function to handle exceptions during the api call
@handle_logprobs_error
def long_form_generation_with_truth_value_api(
    model: str,
    messages: list,
    question: str = None,
    decomp_method: DecompositionMethod = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    generation_seed=None,
    context:str="",
    **kwargs
) -> dict:

    if question == None:
        question = ""
        # search over last user message if exists
        for message in messages[::-1]:
            if message["role"] == "user":
                question = message["content"]
                break

    # adjust seeds
    if generation_seed is not None:
        random.seed(generation_seed)

    seed = kwargs.pop("seed", None)
    if seed == None:
        seed = random.randint(0, 1000000)
    kwargs["seed"] = seed  # a random seed is generated if seed is not specified

    # Generate the main output
    response = completion(model=model, messages=messages, **kwargs)
    generated_text = response.choices[0].message["content"]

    # Factual Decomposition
    print("Decomposing the generated text...")
    claims = decomp_method(generated_text)

    # Get truth score for each claim.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for claim_check_method in claim_check_methods:
        print("Applying claim check method ",
              claim_check_method.__class__.__name__)
        stmt_normalized_truth_values = []
        stmt_unnormalized_truth_values = []
        stmt_method_spec_outputs = []
        for sidx, claim in enumerate(claims):
            text_so_far = " ".join(claims[:sidx]) if sidx > 0 else None
            truth_values = claim_check_method(
                model=model,
                messages=messages,
                generated_text=generated_text,
                question=question,
                claim=claim,
                text_so_far=text_so_far,
                generation_seed=generation_seed,
                context=context,                
                **kwargs
            )
            stmt_normalized_truth_values.append(
                truth_values["normalized_truth_values"])
            stmt_unnormalized_truth_values.append(truth_values["truth_values"])
            stmt_method_spec_outputs.append(truth_values)
        normalized_truth_values.append(stmt_normalized_truth_values)
        unnormalized_truth_values.append(stmt_unnormalized_truth_values)
        method_spec_outputs.append(
            {
                "claim_check_method_name": claim_check_method.__class__.__name__,
                "details": stmt_method_spec_outputs,
            }
        )

    # Create TruthObject
    truth_dict = {
        "generated_text": generated_text,
        "claims": claims,
        "normalized_truth_values": normalized_truth_values,
        "unnormalized_truth_values": unnormalized_truth_values,
        "claim_check_method_details": method_spec_outputs,
    }

    # Return TruthObject
    return truth_dict
