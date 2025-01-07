from .truth_method import TruthMethod
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from .truth_method import TruthMethod

from TruthTorchLM.availability import ACTIVATION_AVAILABLE_API_MODELS 
import torch
import numpy as np
import copy
import random


class AttentionScore(TruthMethod):
    def __init__(self, layer_index:int = -1):#normalization, 
        super().__init__()
        self.layer_index = layer_index


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        model_output = all_ids.to(model.device)

        with torch.no_grad():
            output = model(model_output, output_attentions=True)
            scores = []
            for head_index in range(output.attentions[self.layer_index].shape[1]):#for each head
                attention = output.attentions[self.layer_index][0][head_index]#this values are after softmax
                diag_entries = torch.diagonal(attention)
                log_diag_entries = torch.log(diag_entries)
                score = log_diag_entries.sum().item()
                score = score/len(diag_entries)
                scores.append(score)
            scores = torch.tensor(scores)
            result = torch.mean(scores)
                
        return {"truth_value": result,  "attention_score": result}# we shouldn't return generated text. remove it from the output format
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        if not model in ACTIVATION_AVAILABLE_API_MODELS:
            raise ValueError("Attention Score method cannot be used with black-box API models since it requires access to activations.")

        return {"truth_value": 0}#this output format should be same for all truth methods




