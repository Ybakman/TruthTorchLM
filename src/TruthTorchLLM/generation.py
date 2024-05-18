
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from .truth_methods.truth_method import TruthMethod
from litellm import completion
import random
from TruthTorchLLM.availability import AVAILABLE_API_MODELS


#add cleaning function for the generated text
def generate_with_truth_value(model:PreTrainedModel, text:str, truth_methods: list[TruthMethod] = [], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs) -> dict:
    
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    model_output = model.generate(input_ids)
    tokens = model_output[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method.generate_forward(model, text, generated_text, all_ids=model_output, tokenizer=tokenizer, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict


#for api-based models, we should write a wrapper function to handle exceptions during the api call
def completion_with_truth_value(model:str, text:str, truth_methods: list[TruthMethod] = [], **kwargs) -> dict:
    # Check if the model is an API model
    if type(model) == str and not model in AVAILABLE_API_MODELS:
        raise ValueError(f"model {model} is not supported.")
    # Generate the main output
    
    seed = kwargs.pop('seed', None)
    if seed == None:
        seed = random.randint(0, 1000000)
    kwargs['seed'] = seed #a random seed is generated if seed is not specified

    response = completion(
        model=model,
        messages=[{"content": text, "role": "user"}],#probably remove this one and assume kwargs will have the necessary information
        **kwargs
    )
    generated_text = response.choices[0].message['content']
    
    
      
    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method.completion_forward(model, text, generated_text, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict