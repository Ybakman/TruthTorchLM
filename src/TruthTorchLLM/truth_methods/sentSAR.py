import copy
import torch
import numpy as np
import random
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.utils import sigmoid_normalization, bidirectional_entailment_clustering
from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from ..generation import sample_generations_hf_local, sample_generations_api
from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence_transformers.cross_encoder import CrossEncoder


class SentSAR(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True
    
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations=5, t=0.001, threshold=0.0, std=1.0, 
                 model_for_similarity=None, similarity_model_device = 'cuda', batch_generation=True):#normalization
        super().__init__(threshold = threshold, std = std)

        if model_for_similarity is None:
            self.model_for_similarity = CrossEncoder('cross-encoder/stsb-roberta-large', num_labels=1, device=similarity_model_device)
        else:
            self.model_for_similarity = model_for_similarity

        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.t = t
        self.batch_generation = batch_generation


    def _sentsar(self, generated_texts:list[str], question_context:str, scores:list[float], sampled_generations_dict:dict):

        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []

        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                gen_i = question_context + generated_texts[i]
                gen_j = question_context + generated_texts[j]
                similarity_i_j = self.model_for_similarity.predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        probs = torch.exp(torch.tensor(scores))
        assert len(probs) == len(similarities)

        sentence_scores = []
        for idx, prob in enumerate(probs):
            w_ent = -torch.log(
                prob + ((torch.tensor(similarities[idx]) / self.t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            sentence_scores.append(w_ent)
        sentence_scores = torch.tensor(sentence_scores)

        entropy = (torch.sum(sentence_scores, dim=0) / torch.tensor(sentence_scores.shape[0])).item()
        return {"truth_value": -entropy, 'sentSAR': entropy, "score_for_each_generation": scores, 'generated_texts': generated_texts, "similarities": similarities}
        

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, batch_generation=self.batch_generation **kwargs)


        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        scores = []
        for i in range(self.number_of_generations):
            tokens_text = [tokenizer.decode(token) for token in sampled_generations_dict["tokens"][i]]
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale

        return self._sentsar(generated_texts, question_context, scores, sampled_generations_dict)
        

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):

        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Semantic Entropy method is not applicable to given model")
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, **kwargs)


        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        scores = []
        for i in range(self.number_of_generations):
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale

        return self._sentsar(generated_texts, question_context, scores, sampled_generations_dict)

    def __str__(self):
        return "SentSAR Truth Method with " + str(self.number_of_generations) + " generations and t="+ str(self.t)+". Model for checking sentence similarity: " + self.model_for_similarity.config._name_or_path + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    

    

