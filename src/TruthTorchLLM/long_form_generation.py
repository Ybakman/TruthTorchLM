
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
# from .truth_methods.truth_method import TruthMethod
from litellm import completion
import random
from .decomposition_methods.decomposition_method import FactualDecompositionMethod
from .statement_check_methods.statement_check_method import StatementCheckMethod
from TruthTorchLLM.availability import AVAILABLE_API_MODELS


#add cleaning function for the generated text
def long_form_generation_with_truth_value(model:PreTrainedModel, messages:list, question_context:str = None, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                                          fact_decomp_method:FactualDecompositionMethod=None, stmt_check_method:StatementCheckMethod=None, **kwargs) -> dict:
    
    text = tokenizer.apply_chat_template(messages, tokenize = False)
    if question_context == None:
        question_context = ''
        #search over last user message if exists
        for message in messages[::-1]:
            if message['role'] == 'user':
                question_context = message['content']
                break

    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    model_output = model.generate(input_ids)
    tokens = model_output[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(tokens, skip_special_tokens = True)


    #Factual Decomposition
    print("Decomposing the generated text...")
    decomposition_output = fact_decomp_method.decompose_facts(generated_text)
    statements = decomposition_output['statements']
    print(statements)
    print()

    #Get uncertainty score for each statement.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for sidx, statement in enumerate(statements):
        print("Check for statement: ", statement)
        text_so_far = ' '.join(statements[:sidx]) if sidx > 0 else None
        truth_values = stmt_check_method.check_statement_local(model, text, generated_text, question_context, statement, text_so_far, 
                                                        all_ids=model_output, tokenizer=tokenizer, **kwargs) 
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
def long_form_completion_with_truth_value(model:str, messages:list, question_context:str = None, fact_decomp_method:FactualDecompositionMethod=None, 
                                          stmt_check_method:StatementCheckMethod=None, **kwargs) -> dict:
    
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
    # Generate the main output
    seed = kwargs.pop('seed', None)
    if seed == None:
        seed = random.randint(0, 1000000)
    kwargs['seed'] = seed #a random seed is generated if seed is not specified

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
    
    #Get uncertainty score for each statement.
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    for sidx, statement in enumerate(statements):
        print("Check for statement: ", statement)
        text_so_far = ' '.join(statements[:sidx]) if sidx > 0 else None
        truth_values = stmt_check_method.check_statement_api(model, messages, generated_text, question_context, statement, text_so_far, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'statements':statements, 'decomposition_specific_outputs':decomposition_output, 
                  'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 
                  'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict