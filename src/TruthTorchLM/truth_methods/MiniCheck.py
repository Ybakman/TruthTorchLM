import torch
from typing import Union
import copy
from .truth_method import TruthMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from minicheck.minicheck import MiniCheck

# download minicheck by pip install "minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main"
# paper link: https://arxiv.org/abs/2404.10774
# hugginface link: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B

"""
    Input: Question (question_context), Answer (generated_text), Document (context)
    Output: Support score in [0, 1]
"""


class MiniCheck(TruthMethod):
    REQUIRES_SAMPLED_TEXT = False  
    REQUIRES_SAMPLED_LOGPROBS = False   
    REQUIRES_NORMALIZATION = False
    
    def __init__(self, minicheck_model:str = 'flan-t5-large', **generation_kwargs):
        super().__init__()
        
        if minicheck_model not in ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']:
            raise ValueError("Available Minicheck models are one of: 'roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B'")
        else:
            minicheck_model = self.minicheck_model
        
        
    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        truth_score, truth_label = self.minicheck(self, context, generated_text) 
        return {'truth_value:': truth_score, 'truth_label': truth_label}


    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        
        truth_score, truth_label = self.minicheck(self, context, generated_text) 
        return {'truth_value:': truth_score, 'truth_label': truth_label}


    def minicheck(self, context, generated_text):
        # Loading Minicheck model
        scorer = MiniCheck(model_name=self.minicheck_model, cache_dir='./ckpts')
        print(f'Using Minicheck model: {self.minicheck_model}')
        
        pred_label, raw_prob, _, _ = scorer.score(docs=[context], claims=[generated_text]) 
        return raw_prob[0], pred_label[0]
