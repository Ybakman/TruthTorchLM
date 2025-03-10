from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.long_form_generation.generation import (
    long_form_generation_with_truth_value,
)
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.long_form_generation.decomposition_methods.decomposition_method import (
    DecompositionMethod,
)
from TruthTorchLM.long_form_generation.claim_check_methods.claim_check_method import (
    ClaimCheckMethod,
)

import time
import warnings
from tqdm import tqdm
from typing import Union


def run_over_dataset(
    dataset: list,
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    decomp_method: DecompositionMethod = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    claim_evaluator=None,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_method_details: bool = False,
    return_calim_eval_details: bool = False,
    add_generation_prompt=True,
    continue_final_message=False,
    **kwargs,
):
    
    if dataset[0]["context"] != "" and user_prompt.find("context") == -1:
        user_prompt = "Context: {context}\n" + user_prompt 
        #show warning
        warnings.warn("Context is not in the user prompt but it is provided in the dataset. Adding context to the user prompt. Unexpecting behavior may occur.")
        
    output_dict = {}
    output_dict["previous_context"] = previous_context
    output_dict["user_prompt"] = user_prompt
    output_dict["generation"] = []
    output_dict["claims"] = []
    output_dict["claim_correctness"] = []
    output_dict["question_text"] = []
    output_dict["context"] = []
    output_dict["claim_check_methods"] = []  # save the truth methods
    if return_calim_eval_details:
        output_dict["claim_correctness_details"] = []

    for i in range(len(claim_check_methods)):
        output_dict["claim_check_methods"].append(
            f"{claim_check_methods[i].__class__.__name__}"
        )
        output_dict[f"claim_check_methods_{i}"] = {}
        output_dict[f"claim_check_methods_{i}"]["name"] = str(
            claim_check_methods[i])
        if hasattr(claim_check_methods[i], "truth_methods"):
            output_dict[f"claim_check_methods_{i}"]["truth_methods"] = [
                tm.__class__.__name__ for tm in claim_check_methods[i].truth_methods
            ]
            output_dict[f"claim_check_methods_{i}"]["truth_methods_name"] = [
                str(tm) for tm in claim_check_methods[i].truth_methods
            ]
        output_dict[f"claim_check_methods_{i}"]["truth_values"] = []
        output_dict[f"claim_check_methods_{i}"]["normalized_truth_values"] = []
        if return_method_details:
            output_dict[f"claim_check_methods_{i}"]["method_specific_details"] = [
            ]

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        if dataset[i]["context"] != "":
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(context=dataset[i]["context"], question=dataset[i]["question"]),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(question=dataset[i]["question"]),
                }
            )

        truth_dict = long_form_generation_with_truth_value(
            model=model,
            messages=messages,
            question=dataset[i]["question"],
            tokenizer=tokenizer,
            decomp_method=decomp_method,
            claim_check_methods=claim_check_methods,
            generation_seed=seed,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            context=dataset[i]["context"],
            **kwargs,
        )

        print("Checking for claim support by google search...")
        start_time = time.time()
        results = [claim_evaluator(atomic_fact=claim)
                   for claim in truth_dict["claims"]]
        print(f"Time ellapsed for google search: {time.time()-start_time}")
        output_dict["claim_correctness"].append(
            [
                -1 if res["answer"] == None else 0 if "Not" in res["answer"] else 1
                for res in results
            ]
        )
        if return_calim_eval_details:
            output_dict["claim_correctness_details"].append(results)

        output_dict["generation"].append(truth_dict["generated_text"])
        output_dict["claims"].append(truth_dict["claims"])
        output_dict["question_text"].append(dataset[i]["question"])
        output_dict["context"].append(dataset[i]["context"])

        for j in range(len(claim_check_methods)):
            output_dict[f"claim_check_methods_{j}"]["truth_values"].append(
                truth_dict["unnormalized_truth_values"][j]
            )
            output_dict[f"claim_check_methods_{j}"]["normalized_truth_values"].append(
                truth_dict["normalized_truth_values"][j]
            )
            if return_method_details:
                output_dict[f"claim_check_methods_{j}"][
                    "method_specific_details"
                ].append(truth_dict["claim_check_method_details"][j])

    return output_dict


def decompose_and_label_dataset(
    dataset: list,
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    decomp_method: DecompositionMethod = None,
    claim_evaluator=None,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_calim_eval_details: bool = False,
    add_generation_prompt=True,
    continue_final_message=False,
    **kwargs,
):
    
    if dataset[0]["context"] != "" and user_prompt.find("context") == -1:
        user_prompt = "Context: {context}\n" + user_prompt 
        #show warning
        warnings.warn("Context is not in the user prompt but it is provided in the dataset. Adding context to the user prompt. Unexpecting behavior may occur.")
        

    new_dataset = []
    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        if dataset[i]["context"] != "":
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(context=dataset[i]["context"], question=dataset[i]["question"]),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(question=dataset[i]["question"]),
                }
            )

        if "generated_text" in dataset[i]:
            print("Decomposing the generated text...")
            claims = decomp_method(dataset[i]["generated_text"])
            truth_dict = {
                "claims": claims,
                "generated_text": dataset[i]["generated_text"],
            }
        else:
            truth_dict = long_form_generation_with_truth_value(
                model=model,
                messages=messages,
                question=dataset[i]["question"],
                tokenizer=tokenizer,
                decomp_method=decomp_method,
                claim_check_methods=[],
                generation_seed=seed,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                context=dataset[i]["context"],
                **kwargs,
            )

        print("Checking for claim support by google search...")
        start_time = time.time()
        results = [claim_evaluator(atomic_fact=claim)
                   for claim in truth_dict["claims"]]
        print(f"Time ellapsed for google search: {time.time()-start_time}")

        new_sample = {}
        new_sample["question"] = dataset[i]["question"]
        new_sample["context"] = dataset[i]["context"]
        new_sample["generation"] = truth_dict["generated_text"]
        new_sample["claims"] = truth_dict["claims"]
        new_sample["claim_correctness"] = [
            -1 if res["answer"] == None else 0 if "Not" in res["answer"] else 1
            for res in results
        ]
        if return_calim_eval_details:
            new_sample["claim_correctness_details"] = results
        new_dataset.append(new_sample)

    return new_dataset


def run_over_labelled_dataset(
    dataset: list,
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_method_details: bool = False,
    **kwargs,
):
    output_dict = {}
    output_dict["previous_context"] = previous_context
    output_dict["user_prompt"] = user_prompt
    output_dict["generation"] = []
    output_dict["claims"] = []
    output_dict["claim_correctness"] = []
    output_dict["question_text"] = []
    output_dict["claim_check_methods"] = []  # save the truth methods

    if dataset[0]["context"] != "" and user_prompt.find("context") == -1:
        user_prompt = "Context: {context}\n" + user_prompt 
        #show warning
        warnings.warn("Context is not in the user prompt but it is provided in the dataset. Adding context to the user prompt. Unexpecting behavior may occur.")
        

    for i in range(len(claim_check_methods)):
        output_dict["claim_check_methods"].append(
            f"{claim_check_methods[i].__class__.__name__}"
        )
        output_dict[f"claim_check_methods_{i}"] = {}
        output_dict[f"claim_check_methods_{i}"]["name"] = str(
            claim_check_methods[i])
        if hasattr(claim_check_methods[i], "truth_methods"):
            output_dict[f"claim_check_methods_{i}"]["truth_methods"] = [
                tm.__class__.__name__ for tm in claim_check_methods[i].truth_methods
            ]
            output_dict[f"claim_check_methods_{i}"]["truth_methods_name"] = [
                str(tm) for tm in claim_check_methods[i].truth_methods
            ]
        output_dict[f"claim_check_methods_{i}"]["truth_values"] = []
        output_dict[f"claim_check_methods_{i}"]["normalized_truth_values"] = []
        if return_method_details:
            output_dict[f"claim_check_methods_{i}"]["method_specific_details"] = [
            ]

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        if dataset[i]["context"] != "":
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(context=dataset[i]["context"], question=dataset[i]["question"]),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(question=dataset[i]["question"]),
                }
            )
        
        claims = dataset[i]["claims"]
        claim_correctness = dataset[i]["claim_correctness"]
        generated_text = (
            dataset[i]["generated_text"] if "generated_text" in dataset[i] else None
        )

        truth_dict = process_sample(
            claim_check_methods=claim_check_methods,
            claims=claims,
            question=dataset[i]["question"],
            generated_text=generated_text,
            model=model,
            tokenizer=tokenizer,
            generation_seed=seed,
            context=dataset[i]["context"],
            **kwargs,
        )

        output_dict["claim_correctness"].append(claim_correctness)
        output_dict["claims"].append(claims)
        output_dict["question_text"].append(dataset[i]["question"])
        if "generated_text" in dataset[i]:
            output_dict["generation"].append(dataset[i]["generated_text"])

        for j in range(len(claim_check_methods)):
            output_dict[f"claim_check_methods_{j}"]["truth_values"].append(
                truth_dict["unnormalized_truth_values"][j]
            )
            output_dict[f"claim_check_methods_{j}"]["normalized_truth_values"].append(
                truth_dict["normalized_truth_values"][j]
            )
            if return_method_details:
                output_dict[f"claim_check_methods_{j}"][
                    "method_specific_details"
                ].append(truth_dict["claim_check_method_details"][j])

    return output_dict


def process_sample(
    claim_check_methods: list[ClaimCheckMethod],
    claims: list[str],
    question: str,
    generated_text: str,
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    generation_seed: int = 0,
    context:str="",
    **kwargs,
):
    # Get truth score for each claim.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for claim_check_method in claim_check_methods:
        claim_normalized_truth_values = []
        claim_unnormalized_truth_values = []
        claim_method_spec_outputs = []
        for sidx, claim in enumerate(claims):
            print("Check for claim: ", claim)
            text_so_far = " ".join(claims[:sidx]) if sidx > 0 else None
            if type(model) == str:
                truth_values = claim_check_method(
                    model=model,
                    messages=None,
                    generated_text=generated_text,
                    question=question,
                    claim=claim,
                    text_so_far=text_so_far,
                    generation_seed=generation_seed,
                    context=context,
                    **kwargs,
                )
            else:
                truth_values = claim_check_method(
                    model=model,
                    input_text=None,
                    generated_text=generated_text,
                    question=question,
                    claim=claim,
                    text_so_far=text_so_far,
                    all_ids=None,
                    tokenizer=tokenizer,
                    generation_seed=generation_seed,
                    messages=None,
                    context=context,
                    **kwargs,
                )
            claim_normalized_truth_values.append(
                truth_values["normalized_truth_values"]
            )
            claim_unnormalized_truth_values.append(
                truth_values["truth_values"])
            claim_method_spec_outputs.append(truth_values)
        normalized_truth_values.append(claim_normalized_truth_values)
        unnormalized_truth_values.append(claim_unnormalized_truth_values)
        method_spec_outputs.append(claim_method_spec_outputs)

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
