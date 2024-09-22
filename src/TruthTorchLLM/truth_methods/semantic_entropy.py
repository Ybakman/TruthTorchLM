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
from ..generation import sample_generations_batch_hf_local, sample_generations_sequential_hf_local, sample_generations_api
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_total_log(generated_outputs : dict[str, float],clusters : list[set[str]]):
    total_output_for_log = 0
    for i, cluster in enumerate(clusters):
        total_output_for_log -= torch.logsumexp(torch.tensor([generated_outputs[elem] for elem in cluster]), dim=0).item()
    return total_output_for_log / len(clusters)


class SemanticEntropy(TruthMethod):
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True

    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations=5, threshold=0.0, std=1.0, 
                 model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda'):#normalization
        super().__init__(threshold = threshold, std = std)

        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)

        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_sequential_hf_local(model, input_text, tokenizer, [self], generation_seed, **kwargs)

            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        generated_outputs = {}
        scores = []

        for i in range(self.number_of_generations):
            #check if the text is already sampled
            text = generated_texts[i]
            if text in generated_texts[:i]:
                continue
            tokens_text = [tokenizer.decode(token) for token in sampled_generations_dict["tokens"][i]]
            score = self.scoring_function(question_context, tokens_text, sampled_generations_dict["logprobs"][i], generated_texts[i], sampled_generations_dict["tokens"][i]) 
            scores.append(score) #scores are in log scale
            generated_outputs[generated_texts[i]] = score
        
        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, list(generated_outputs.keys()))   
        total_output_for_log = calculate_total_log(generated_outputs,clusters)
        normalized_truth_value = sigmoid_normalization(total_output_for_log, self.threshold, self.std)
        return {"truth_value": -total_output_for_log, 'normalized_truth_value': normalized_truth_value, 'semantic_entropy': total_output_for_log, "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}


    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)

        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Semantic Entropy method is not applicable to given model")
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model, messages, [self], generation_seed, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        generated_outputs = {}
        scores = []

        for i in range(self.number_of_generations):
            text = generated_texts[i]
            if text in generated_texts[:i]:
                continue
            score = self.scoring_function(question_context, sampled_generations_dict["tokens"][i], sampled_generations_dict["logprobs"][i], generated_texts[i]) 
            scores.append(score) #scores are in log scale
            generated_outputs[generated_texts[i]] = score

        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, list(generated_outputs.keys()))
        total_output_for_log = calculate_total_log(generated_outputs,clusters) 
        normalized_truth_value = sigmoid_normalization(total_output_for_log, self.threshold, self.std)
        return {"truth_value": -total_output_for_log, 'normalized_truth_value': normalized_truth_value, 'semantic_entropy': total_output_for_log, "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}

    def __str__(self):
        return "Semantic Entropy Truth Method with " + str(self.number_of_generations) + " generations. Model for checking semantic: " + self.model_for_entailment.config._name_or_path + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    

    

