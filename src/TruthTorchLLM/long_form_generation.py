
import torch
import random
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .decomposition_methods.decomposition_method import FactualDecompositionMethod
from .statement_check_methods.statement_check_method import StatementCheckMethod
from TruthTorchLLM.availability import AVAILABLE_API_MODELS
from TruthTorchLLM.utils.common_utils import generate


def long_form_generation_with_truth_value(model:PreTrainedModel, messages:list, question_context:str = None, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                                          fact_decomp_method:FactualDecompositionMethod=None, stmt_check_method:StatementCheckMethod=None, generation_seed=None,
                                           add_generation_prompt = True, continue_final_message = False,  **kwargs) -> dict:
    if type(model) == str:
        return long_form_generation_with_truth_value_api(model = model, messages = messages, question_context = question_context, fact_decomp_method = fact_decomp_method, stmt_check_method = stmt_check_method, generation_seed=generation_seed, **kwargs)
    else:
        return long_form_generation_with_truth_value_hf_local(model = model, messages = messages, question_context = question_context, fact_decomp_method = fact_decomp_method, stmt_check_method=stmt_check_method, 
        tokenizer = tokenizer, generation_seed = generation_seed, add_generation_prompt = add_generation_prompt, continue_final_message = continue_final_message, **kwargs)


#add cleaning function for the generated text
def long_form_generation_with_truth_value_hf_local(model:PreTrainedModel, messages:list, question_context:str = None, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                                          fact_decomp_method:FactualDecompositionMethod=None, stmt_check_method:StatementCheckMethod=None, generation_seed=None,
                                          add_generation_prompt = True, continue_final_message = False,  **kwargs) -> dict:

    if question_context == None:
        question_context = ''
        #search over last user message if exists
        for message in messages[::-1]:
            if message['role'] == 'user':
                question_context = message['content']
                break

    eos_token_id = kwargs.pop("eos_token_id", None)
    if eos_token_id is None:    
        eos_token_id = model.config.eos_token_id
    kwargs['eos_token_id'] = eos_token_id

    pad_token_id = kwargs.pop("pad_token_id", None)
    if pad_token_id is None:
        if type(eos_token_id) == list:
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id
    kwargs['pad_token_id'] = pad_token_id 
    
    #adjust seeds
    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        random.seed(generation_seed)

    # Generate the main output
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
    generated_output = generate(text, model, tokenizer, **kwargs)
    generated_text = generated_output['generated_text_skip_specials']
    model_output = generated_output['all_ids']
    del generated_output

    #Factual Decomposition
    print("Decomposing the generated text...")
    decomposition_output = fact_decomp_method.decompose_facts(generated_text)
    statements = decomposition_output['statements']
    print(statements)
    print()

    #Get truth score for each statement.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for sidx, statement in enumerate(statements):
        print("Check for statement: ", statement)
        text_so_far = ' '.join(statements[:sidx]) if sidx > 0 else None
        truth_values = stmt_check_method(model=model, input_text=text, generated_text=generated_text, question_context=question_context, statement=statement, 
                                         text_so_far=text_so_far, all_ids=model_output, tokenizer=tokenizer, generation_seed=generation_seed, messages=messages, **kwargs) 
        
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'statements':statements, 'decomposition_specific_outputs':decomposition_output, 
                  'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 
                  'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict


#for api-based models, we should write a wrapper function to handle exceptions during the api call
def long_form_generation_with_truth_value_api(model:str, messages:list, question_context:str = None, fact_decomp_method:FactualDecompositionMethod=None, 
                                          stmt_check_method:StatementCheckMethod=None, generation_seed=None, **kwargs) -> dict:


    # Check if the model is an API model
    if type(model) == str and not model in AVAILABLE_API_MODELS:
        raise ValueError(f"model {model} is not supported.")

    if question_context == None:
        question_context = ''
        #search over last user message if exists
        for message in messages[::-1]:
            if message['role'] == 'user':
                question_context = message['content']
                break

    #adjust seeds
    if generation_seed is not None:
        random.seed(generation_seed)

    seed = kwargs.pop('seed', None)
    if seed == None:
        seed = random.randint(0, 1000000)
    kwargs['seed'] = seed #a random seed is generated if seed is not specified

    # Generate the main output
    response = completion(
        model=model,
        messages=messages,
        **kwargs
    )
    generated_text = response.choices[0].message['content']

    #Factual Decomposition
    print("Decomposing the generated text...")
    decomposition_output = fact_decomp_method.decompose_facts(generated_text)
    statements = decomposition_output['statements']
    print(statements)
    print()
    
    #Get truth score for each statement.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for sidx, statement in enumerate(statements):
        print("Check for statement: ", statement)
        text_so_far = ' '.join(statements[:sidx]) if sidx > 0 else None
        truth_values = stmt_check_method(model=model, messages=messages, generated_text=generated_text, question_context=question_context, 
                                         statement=statement, text_so_far=text_so_far, generation_seed=generation_seed, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'statements':statements, 'decomposition_specific_outputs':decomposition_output, 
                  'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 
                  'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict