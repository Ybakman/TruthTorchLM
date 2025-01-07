import copy
import torch
from typing import Union

from .truth_method import TruthMethod
from TruthTorchLM.availability import ACTIVATION_AVAILABLE_API_MODELS 
from ..generation import sample_generations_hf_local

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

class Inside(TruthMethod):
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_ACTIVATIONS = True

    def __init__(self, number_of_generations: int = 10, alpha:float = 0.001, batch_generation = True): 
        super().__init__()
        self.number_of_generations = number_of_generations
        self.alpha = alpha
        self.batch_generation = batch_generation

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_activations=True, **kwargs)
        
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        sentence_embeddings = torch.stack([hidden_states[-1][int(len(hidden_states[-1])/2)][0] for hidden_states in sampled_generations_dict["activations"][:self.number_of_generations]]) #TODO: check this part is correct or not

        hidden_dim = sentence_embeddings.shape[-1]
        centering_matrix = torch.eye(hidden_dim) - (torch.ones((hidden_dim, hidden_dim)) / hidden_dim)

        covariance = sentence_embeddings @ centering_matrix.to(sentence_embeddings.dtype) @ sentence_embeddings.T
        regularized_covarience = covariance + torch.eye(self.number_of_generations) * self.alpha
        eigenvalues, _ = torch.linalg.eig(regularized_covarience)

        eigenvalues = eigenvalues.real

        eigen_score = -torch.mean(torch.log(eigenvalues)).cpu().item()
        return {"truth_value": eigen_score,   'generated_texts_for_inside': generated_texts}#this output format should be same for all truth methods

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):

        if not model in ACTIVATION_AVAILABLE_API_MODELS:
            raise ValueError("Inside method cannot be used with black-box API models since it requires access to activations.")

        return {"truth_value": 0, 'generated_texts_for_inside': []}#this output format should be same for all truth methods
