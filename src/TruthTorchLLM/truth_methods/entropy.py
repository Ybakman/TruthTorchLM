from .truth_method import TruthMethod
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


class Entropy(TruthMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations: int = 5, threshold:float = 0.0, std:float = 1.0):#normalization, 
        super().__init__()
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.threshold = threshold
        self.std = std


    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        scores = []
        generated_texts = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        for i in range(self.number_of_generations):
            
            model_output = model.generate(input_ids, num_return_sequences = 1, do_sample = True, **kwargs)
            
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

            score = self.scoring_function(question_context, tokens_text, logprobs)
            scores.append(score)
            generated_texts.append(tokenizer.decode(tokens))

        entropy = -np.sum(scores) / len(scores)#scores are in log scale

        normalized_truth_value = sigmoid_normalization(-entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_truth_value':normalized_truth_value,  'entropy': entropy,  "score_for_each_generation": scores, 'generated_texts_for_entropy': generated_texts}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context, **kwargs):
        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Entropy method is not applicable to given model")

        kwargs = copy.deepcopy(kwargs)
        scores = []
        generated_texts = []
        
            
        for i in range(self.number_of_generations):
            kwargs.pop('logprobs', None)
            seed = kwargs.pop('seed', None) #if user specifies seed, it won't be generated randomly

            seed = random.randint(0, 1000000)
            kwargs['seed'] = seed
            response = completion(
                model = model,
                messages = messages,
                logprobs = True,
                **kwargs
                )
            
            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            score = self.scoring_function(question_context, tokens, logprobs)
            scores.append(score)
            generated_texts.append(response.choices[0].message['content'])
        
        entropy = -np.sum(scores) / len(scores)#scores are in log scale

        normalized_truth_value = sigmoid_normalization(-entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_truth_value':normalized_truth_value,  'entropy': entropy,  "score_for_each_generation": scores, 'generated_texts_for_entropy': generated_texts}#this output format should be same for all truth methods

    
    def __str__(self):
        return "Entropy Truth Method with " + str(self.number_of_generations) + " generations. Scoring function: " + str(self.scoring_function) + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)