
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from .trust_methods.trust_method import TrustMethod
from litellm import completion
import random
from TruthTorchLLM.availability import AVAILABLE_API_MODELS

#for api-based models, we should write a wrapper function to handle exceptions during the api call
#add cleaning function for the generated text
def generate_with_trust_value(model:Union[str,PreTrainedModel], text:str, trust_methods: list[TrustMethod] = [], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs) -> dict:
    # Check if the model is an API model
    if type(model) == str and not model in AVAILABLE_API_MODELS:
        raise ValueError(f"model {model} is not supported.")
    # Generate the main output
    if isinstance(model, str) and model in AVAILABLE_API_MODELS:
        seed = kwargs.pop('seed', None)
        if seed == None:
            seed = random.randint(0, 1000000)
        kwargs['seed'] = seed #a random seed is generated if seed is not specified

        response = completion(
            model=model,
            messages=[{"content": text, "role": "user"}],
            **kwargs
        )
        generated_text = response.choices[0].message['content']
    
    else:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        model_output = model.generate(input_ids)
        tokens = model_output[0][len(input_ids[0]):]
        generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
      
    # Get scores from all trust methods
    normalized_trust_values = []
    unnormalized_trust_values = []
    method_spec_outputs = []
    
    for trust_method in trust_methods:
        trust_values = trust_method(model, text, generated_text, tokenizer=tokenizer, **kwargs)
        normalized_trust_values.append(trust_values['normalized_trust_value'])
        unnormalized_trust_values.append(trust_values['trust_value'])
        method_spec_outputs.append(trust_values)

    # Create TrustObject
    trust_dict = {'generated_text':generated_text, 'normalized_trust_values':normalized_trust_values, 'unnormalized_trust_values':unnormalized_trust_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TrustObject
    return trust_dict