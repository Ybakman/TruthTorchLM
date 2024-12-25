from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.long_form_generation.generation import long_form_generation_with_truth_value
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.long_form_generation.decomposition_methods.decomposition_method import FactualDecompositionMethod
from TruthTorchLM.long_form_generation.statement_check_methods.statement_check_method import StatementCheckMethod
import wandb
import pandas as pd
import numpy as np
import time


def run_over_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          fact_decomp_method:FactualDecompositionMethod=None, stmt_check_methods:list[StatementCheckMethod]=None,
                          claim_evaluator = None, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, 
                          return_method_details:bool = False, return_calim_eval_details:bool=False, add_generation_prompt = True, continue_final_message = False, **kwargs):
    output_dict = {}
    output_dict['previous_context'] = previous_context
    output_dict['user_prompt'] = user_prompt
    output_dict['generation'] = []
    output_dict['statements'] = []
    output_dict['statement_correctness'] = []
    output_dict['question_text'] = []
    output_dict['stmt_check_methods'] = []#save the truth methods
    if return_calim_eval_details:
        output_dict['statement_correctness_details'] = []

    
    for i in range(len(stmt_check_methods)):
        output_dict['stmt_check_methods'].append(f'{stmt_check_methods[i].__class__.__name__}')
        output_dict[f'stmt_check_methods_{i}'] = {}
        output_dict[f'stmt_check_methods_{i}']['name'] = str(stmt_check_methods[i])
        if hasattr(stmt_check_methods[i], 'truth_methods'):
            output_dict[f'stmt_check_methods_{i}']['truth_methods'] = [tm.__class__.__name__ for tm in stmt_check_methods[i].truth_methods]
            output_dict[f'stmt_check_methods_{i}']['truth_methods_name'] = [str(tm) for tm in stmt_check_methods[i].truth_methods]
        output_dict[f'stmt_check_methods_{i}']['truth_values'] = []
        output_dict[f'stmt_check_methods_{i}']['normalized_truth_values'] = []  
        if return_method_details:
            output_dict[f'stmt_check_methods_{i}']['method_specific_details'] = []

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        messages.append({'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])})

        truth_dict = long_form_generation_with_truth_value(model=model, messages=messages, question_context=dataset[i]['question'], tokenizer=tokenizer, 
                                          fact_decomp_method=fact_decomp_method, stmt_check_methods=stmt_check_methods, generation_seed=seed,
                                           add_generation_prompt = add_generation_prompt, continue_final_message = continue_final_message,  **kwargs)
  
        print("Checking for claim support by google search...")
        start_time = time.time()
        results = [claim_evaluator(atomic_fact=statement) for statement in truth_dict['statements']]
        print(f"Time ellapsed for google search: {time.time()-start_time}")
        output_dict['statement_correctness'].append([ -1 if res['answer'] == None else 0 if "Not" in res['answer'] else 1 for res in results])
        if return_calim_eval_details:
            output_dict['statement_correctness_details'].append(results)

        output_dict['generation'].append(truth_dict['generated_text'])
        output_dict['statements'].append(truth_dict['statements'])
        output_dict['question_text'].append(dataset[i]['question'])
        
        for j in range(len(stmt_check_methods)):
            output_dict[f'stmt_check_methods_{j}']['truth_values'].append(truth_dict['unnormalized_truth_values'][j])
            output_dict[f'stmt_check_methods_{j}']['normalized_truth_values'].append(truth_dict['normalized_truth_values'][j])
            if return_method_details:
                output_dict[f'stmt_check_methods_{j}']['method_specific_details'].append(truth_dict['method_specific_outputs'][j])

    return output_dict

def decompose_and_label_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          fact_decomp_method:FactualDecompositionMethod=None, claim_evaluator = None, 
                          previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, 
                          return_calim_eval_details:bool=False, add_generation_prompt = True, continue_final_message = False, **kwargs):
    new_dataset = []
    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        messages.append({'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])})

        if "generated_text" in dataset[i]:
            print("Decomposing the generated text...")
            statements = fact_decomp_method(dataset[i]["generated_text"])
            truth_dict = {'statements':statements, 'generated_text':dataset[i]["generated_text"]}
        else:
            truth_dict = long_form_generation_with_truth_value(model=model, messages=messages, question_context=dataset[i]['question'], tokenizer=tokenizer, 
                                            fact_decomp_method=fact_decomp_method, stmt_check_methods=[], generation_seed=seed,
                                            add_generation_prompt = add_generation_prompt, continue_final_message = continue_final_message,  **kwargs)
  
        print("Checking for claim support by google search...")
        start_time = time.time()
        results = [claim_evaluator(atomic_fact=statement) for statement in truth_dict['statements']]
        print(f"Time ellapsed for google search: {time.time()-start_time}")
        
        new_sample = {}
        new_sample['question'] = dataset[i]['question']
        new_sample['generation'] = truth_dict['generated_text']
        new_sample['statements'] = truth_dict['statements']
        new_sample['statement_correctness'] = [ -1 if res['answer'] == None else 0 if "Not" in res['answer'] else 1 for res in results]
        if return_calim_eval_details:
            new_sample['statement_correctness_details'] = results
        new_dataset.append(new_sample)

    return new_dataset
    

def run_over_labelled_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          stmt_check_methods:list[StatementCheckMethod]=None, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, 
                          return_method_details:bool = False, **kwargs):
    output_dict = {}
    output_dict['previous_context'] = previous_context
    output_dict['user_prompt'] = user_prompt
    output_dict['generation'] = []
    output_dict['statements'] = []
    output_dict['statement_correctness'] = []
    output_dict['question_text'] = []
    output_dict['stmt_check_methods'] = []#save the truth methods

    for i in range(len(stmt_check_methods)):
        output_dict['stmt_check_methods'].append(f'{stmt_check_methods[i].__class__.__name__}')
        output_dict[f'stmt_check_methods_{i}'] = {}
        output_dict[f'stmt_check_methods_{i}']['name'] = str(stmt_check_methods[i])
        if hasattr(stmt_check_methods[i], 'truth_methods'):
            output_dict[f'stmt_check_methods_{i}']['truth_methods'] = [tm.__class__.__name__ for tm in stmt_check_methods[i].truth_methods]
            output_dict[f'stmt_check_methods_{i}']['truth_methods_name'] = [str(tm) for tm in stmt_check_methods[i].truth_methods]
        output_dict[f'stmt_check_methods_{i}']['truth_values'] = []
        output_dict[f'stmt_check_methods_{i}']['normalized_truth_values'] = []  
        if return_method_details:
            output_dict[f'stmt_check_methods_{i}']['method_specific_details'] = []

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        messages.append({'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])})
        statements = dataset[i]['statements']
        statement_correctness = dataset[i]['statement_correctness']
        generated_text = dataset[i]['generated_text'] if 'generated_text' in dataset[i] else None

        truth_dict = process_sample(stmt_check_methods=stmt_check_methods, statements=statements,
                                    question_context=dataset[i]['question'], generated_text=generated_text,
                                    model=model, tokenizer=tokenizer, generation_seed=seed, **kwargs)
  
        output_dict['statement_correctness'].append(statement_correctness)
        output_dict['statements'].append(statements)
        output_dict['question_text'].append(dataset[i]['question'])
        if 'generated_text' in dataset[i]:
            output_dict['generation'].append(dataset[i]['generated_text'])
        
        for j in range(len(stmt_check_methods)):
            output_dict[f'stmt_check_methods_{j}']['truth_values'].append(truth_dict['unnormalized_truth_values'][j])
            output_dict[f'stmt_check_methods_{j}']['normalized_truth_values'].append(truth_dict['normalized_truth_values'][j])
            if return_method_details:
                output_dict[f'stmt_check_methods_{j}']['method_specific_details'].append(truth_dict['method_specific_outputs'][j])

    return output_dict
    
def process_sample(stmt_check_methods:list[StatementCheckMethod], statements:list[str], 
                   question_context:str, generated_text:str, 
                   model:Union[str,PreTrainedModel], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                   generation_seed:int=0, **kwargs):
    #Get truth score for each statement.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for stmt_check_method in stmt_check_methods:
        stmt_normalized_truth_values = []
        stmt_unnormalized_truth_values = []
        stmt_method_spec_outputs = []
        for sidx, statement in enumerate(statements):
            print("Check for statement: ", statement)
            text_so_far = ' '.join(statements[:sidx]) if sidx > 0 else None
            if type(model) == str:
                truth_values = stmt_check_method(model=model, messages=None, generated_text=generated_text, question_context=question_context, 
                                                statement=statement, text_so_far=text_so_far, generation_seed=generation_seed, **kwargs)
            else:
                truth_values = stmt_check_method(model=model, input_text=None, generated_text=generated_text, question_context=question_context, statement=statement, 
                                                text_so_far=text_so_far, all_ids=None, tokenizer=tokenizer, generation_seed=generation_seed, messages=None, **kwargs) 
            stmt_normalized_truth_values.append(truth_values['normalized_truth_values'])
            stmt_unnormalized_truth_values.append(truth_values['truth_values'])
            stmt_method_spec_outputs.append(truth_values)
        normalized_truth_values.append(stmt_normalized_truth_values)
        unnormalized_truth_values.append(stmt_unnormalized_truth_values)
        method_spec_outputs.append(stmt_method_spec_outputs)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'statements':statements,
                  'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 
                  'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict
