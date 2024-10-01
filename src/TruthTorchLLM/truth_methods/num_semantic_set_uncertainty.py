import copy
import torch
import random
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.utils import calculate_U_num_set, sigmoid_normalization
from .truth_method import TruthMethod
from ..generation import sample_generations_batch_hf_local, sample_generations_sequential_hf_local, sample_generations_api



class NumSemanticSetUncertainty(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True
    
    def __init__(self, method_for_similarity: str = "semantic", number_of_generations=5, threshold=0.0, std=1.0, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda'):
        super().__init__(threshold = threshold, std = std)

        if (model_for_entailment is None or tokenizer_for_entailment is None) and method_for_similarity == "semantic":
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')
       
        if method_for_similarity not in ["semantic", "jaccard"]:
            raise ValueError("method_for_similarity should be either semantic or jaccard. Please refer to https://arxiv.org/pdf/2305.19187 for more information.")
        
        self.model_for_entailment = None
        self.tokenizer_for_entailment = None
        
        if method_for_similarity == "semantic":
            print('There are 2 methods for similarity: semantic similarity and jaccard score. The default method is semantic similarity. If you want to use jaccard score, please set method_for_similarity="jaccard". Please refer to https://arxiv.org/pdf/2305.19187 for more information.')
            self.tokenizer_for_entailment = tokenizer_for_entailment
            self.model_for_entailment = model_for_entailment

        self.number_of_generations = number_of_generations
        self.method_for_similarity = method_for_similarity #jaccard or semantic

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)

        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_sequential_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        output_dict = {}
        
        output  = calculate_U_num_set(generated_texts, question_context, method_for_similarity=self.method_for_similarity, model_for_entailment=self.model_for_entailment, tokenizer_for_entailment=self.tokenizer_for_entailment)
        output_dict['U_num_set'] = output
        output_dict['generated_texts'] = generated_texts
        output_dict['truth_value'] = -output
        output_dict['normalized_truth_value'] = sigmoid_normalization(output, self.threshold, self.std)
        return output_dict

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        output_dict = {}
       
        output  = calculate_U_num_set(generated_texts, question_context, method_for_similarity=self.method_for_similarity, model_for_entailment=self.model_for_entailment, tokenizer_for_entailment=self.tokenizer_for_entailment)
        output_dict['U_num_set'] = output
        output_dict['generated_texts'] = generated_texts
        output_dict['truth_value'] = -output
        output_dict['normalized_truth_value'] = sigmoid_normalization(output, self.threshold, self.std)
        return output_dict


    def __str__(self):
        return f'Number of Semantic Set Uncertainty with {self.method_for_similarity} similarity method. Model for entailment: {self.model_for_entailment}, Tokenizer for entailment: {self.tokenizer_for_entailment}, Number of generations: {self.number_of_generations}, Threshold: {self.threshold}, Standard Deviation: {self.std}.'
    