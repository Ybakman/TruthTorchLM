
import torch
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import  DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLM.utils import calculate_U_ecc
from .truth_method import TruthMethod
from ..generation import sample_generations_hf_local, sample_generations_api


class EccentricityUncertainty(TruthMethod):
    
    REQUIRES_SAMPLED_TEXT = True

    def __init__(self, method_for_similarity: str = "semantic", number_of_generations=5, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None, temperature= 3.0, eigen_threshold = 0.9, entailment_model_device = 'cuda', batch_generation = True):
        super().__init__()

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
        self.eigen_threshold = eigen_threshold
        self.temperature = temperature #temperature for NLI model
        self.batch_generation = batch_generation

    def _eccentricity_uncertainty(self, sampled_generations_dict:dict, question_context:str):
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        output_dict = {}
        output  = calculate_U_ecc(generated_texts, question_context, method_for_similarity=self.method_for_similarity, temperature = self.temperature, eigen_threshold = self.eigen_threshold, model_for_entailment=self.model_for_entailment, tokenizer_for_entailment=self.tokenizer_for_entailment)
        output_dict['U_ecc'] = output
        output_dict['generated_texts'] = generated_texts
        output_dict['truth_value'] = -output
        return output_dict

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=False, batch_generation = self.batch_generation, **kwargs)
            
        return self._eccentricity_uncertainty(sampled_generations_dict, question_context)

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, **kwargs)
            
        return self._eccentricity_uncertainty(sampled_generations_dict, question_context)

