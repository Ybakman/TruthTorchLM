from .trust_method import TrustMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np
import copy
import random


class Confidence(TrustMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(),threshold:float=0.0, std:float = 1.0):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
        self.threshold = threshold
        self.std = std


    def __call__(self, model:Union[str,PreTrainedModel], input_text:str, generated_text:str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs) -> dict: #any generation argument can be passed.

        if type(model) == str and not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Entropy method is not applicable to given model")

        kwargs = copy.deepcopy(kwargs)
        if model in PROB_AVAILABLE_API_MODELS:
           
            response = completion(
                model=model,
                messages=[{ "content": input_text, "role": "user"}],
                logprobs = True,
                **kwargs
                )
                
            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            score = self.scoring_function(input_text, tokens, logprobs)
            generated_text = response.choices[0].message['content']
        else:
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
            #concatenation of input and generated text
            model_output = torch.concat((input_ids , tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(model.device)), dim=1)
        
            tokens = model_output[0][len(input_ids[0]):]
            tokens_text = [tokenizer.decode(token) for token in tokens]

            with torch.no_grad():
                outputs = model(model_output)
                logits = outputs.logits  # Logits for each token in the input

                # Calculate probabilities from logits
                logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
                logprobs = logprobs[0, len(input_ids[0])-1:-1, :]#logprobs for each token in the generated text
                logprobs = torch.gather(logprobs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
                logprobs = logprobs.view(-1).tolist()#convert to list

                score = self.scoring_function(input_text, tokens_text, logprobs)
                score = score
                generated_text = generated_text
        
        normalized_trust_value = sigmoid_normalization(score, self.threshold, self.std)
        
        return {"trust_value": score, 'normalized_trust_value': normalized_trust_value, "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
    

    def __str__(self):
        return "Confidence Trust Method with " + str(self.scoring_function) + " scoring function." + " Threshold: " + str(self.threshold) + "Std: " + str(self.std)