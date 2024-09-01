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


class Inside(TruthMethod):
    def __init__(self, threshold:float=0.0, std:float=1.0, number_of_generations: int = 10, alpha:float = 0.001): 
        super().__init__(threshold = threshold, std = std)
        self.number_of_generations = number_of_generations
        self.alpha = alpha

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        scores = []
        generated_texts = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        sentence_embeddings = torch.Tensor().to(model.device)
        for i in range(self.number_of_generations):
            
            model_output = model.generate(input_ids, num_return_sequences = 1, do_sample = True, **kwargs)
            
            tokens = model_output[0][len(input_ids[0]):]
            generated_texts.append(tokenizer.decode(tokens))

            with torch.no_grad():
                outputs = model(model_output, output_hidden_states=True)
                layer_activations = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
                sentence_embedding = layer_activations[0, -2:-1, :]

            sentence_embeddings = torch.cat((sentence_embeddings, sentence_embedding), dim=0)

        hidden_dim = sentence_embeddings.shape[-1]
        centering_matrix = torch.eye(hidden_dim) - (torch.ones((hidden_dim, hidden_dim)) / hidden_dim)
        centering_matrix = centering_matrix.to(model.device)

        covariance = sentence_embeddings @ centering_matrix @ sentence_embeddings.T
        regularized_covarience = covariance + torch.eye(hidden_dim).to(model.device) * self.alpha

        eigenvalues, _ = torch.linalg.eig(regularized_covarience)
        eigenvalues = eigenvalues.real

        eigen_score = torch.mean(torch.log(eigenvalues))


        normalized_truth_value = sigmoid_normalization(-eigen_score, self.threshold, self.std)
        return {"truth_value": -eigen_score, 'normalized_truth_value':normalized_truth_value,  'entropy': entropy,  "score_for_each_generation": scores, 'generated_texts_for_entropy': generated_texts}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
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