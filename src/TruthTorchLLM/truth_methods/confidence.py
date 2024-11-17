from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from .truth_method import TruthMethod

from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np
import copy
import random


class Confidence(TruthMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(),threshold:float=0.0, std:float = 1.0):#normalization, 
        super().__init__(threshold = threshold, std = std)
        self.scoring_function = scoring_function


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        model_output = all_ids
        tokens = model_output[0][len(input_ids[0]):]
        tokens_text = [tokenizer.decode([token]) for token in tokens]

        with torch.no_grad():
            outputs = model(model_output)
            logits = outputs.logits  # Logits for each token in the input

            # Calculate probabilities from logits
            logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
            logprobs = logprobs[0, len(input_ids[0])-1:-1, :]#logprobs for each token in the generated text
            logprobs = torch.gather(logprobs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
            logprobs = logprobs.view(-1).tolist()#convert to list

            score = self.scoring_function(logprobs)
            score = score
            generated_text = generated_text
                
        return {"truth_value": score,  "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):

        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Confidence method is not applicable to given model")

        kwargs = copy.deepcopy(kwargs)
        
           
        response = completion(
            model=model,
            messages=messages,
            logprobs = True,
            **kwargs
            )
            
        logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
        tokens = [token['token'] for token in response.choices[0].logprobs['content']]
        generated_text = response.choices[0].message['content']
        score = self.scoring_function(logprobs)

        normalized_truth_value = sigmoid_normalization(score, self.threshold, self.std)

        return {"truth_value": score, "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format





    def __str__(self):
        return "Confidence Truth Method with " + str(self.scoring_function) + " scoring function." + " Threshold: " + str(self.threshold) + " Std: " + str(self.std)