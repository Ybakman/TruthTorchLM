
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from .truth_methods.truth_method import TruthMethod
from litellm import completion
import random
from TruthTorchLLM.availability import AVAILABLE_API_MODELS


#change the naming of the functions to be more descriptive

#add cleaning function for the generated text
def generate_with_truth_value(model:PreTrainedModel, messages:list, question_context:str = None, truth_methods: list[TruthMethod] = [], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs) -> dict:
    
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
    generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method.generate_forward(model, text, generated_text, question_context, all_ids=model_output, tokenizer=tokenizer, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict


#for api-based models, we should write a wrapper function to handle exceptions during the api call
def completion_with_truth_value(model:str, messages:list, question_context:str = None, truth_methods: list[TruthMethod] = [], **kwargs) -> dict:
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
    
    
      
    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method.completion_forward(model, messages, generated_text, question_context, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict