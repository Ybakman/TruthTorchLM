import copy
import torch
from typing import Union

from .truth_method import TruthMethod
from TruthTorchLLM.utils import sigmoid_normalization
from TruthTorchLLM.availability import ACTIVATION_AVAILABLE_API_MODELS 
from ..generation import sample_generations_batch_hf_local, sample_generations_sequential_hf_local

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

class Inside(TruthMethod):
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_ACTIVATIONS = True

    def __init__(self, threshold:float=0.0, std:float=1.0, number_of_generations: int = 10, alpha:float = 0.001): 
        super().__init__(threshold = threshold, std = std)
        self.number_of_generations = number_of_generations
        self.alpha = alpha

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_sequential_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_activations=True, **kwargs)
        
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        sentence_embeddings = torch.stack([hidden_states[-1][int(len(hidden_states[-1])/2)][0] for hidden_states in sampled_generations_dict["activations"][:self.number_of_generations]]) #TODO: check this part is correct or not

        hidden_dim = sentence_embeddings.shape[-1]
        centering_matrix = torch.eye(hidden_dim) - (torch.ones((hidden_dim, hidden_dim)) / hidden_dim)

        covariance = sentence_embeddings @ centering_matrix @ sentence_embeddings.T
        regularized_covarience = covariance + torch.eye(self.number_of_generations) * self.alpha
        eigenvalues, _ = torch.linalg.eig(regularized_covarience)

        eigenvalues = eigenvalues.real

        eigen_score = -torch.mean(torch.log(eigenvalues)).cpu().item()
        normalized_truth_value = sigmoid_normalization(eigen_score, self.threshold, self.std)
        return {"truth_value": eigen_score, 'normalized_truth_value':normalized_truth_value,  'generated_texts_for_inside': generated_texts}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)

        if not model in ACTIVATION_AVAILABLE_API_MODELS:
            raise ValueError("Inside method cannot be used with black-box API models since it requires access to activations.")

        return {"truth_value": 0, 'normalized_truth_value':0,  'generated_texts_for_inside': []}#this output format should be same for all truth methods

    def __str__(self):
        return "Inside Truth Method with " + str(self.number_of_generations) + " generations. Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    