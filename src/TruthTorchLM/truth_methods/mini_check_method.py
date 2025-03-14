import torch
from typing import Union
import copy
from .truth_method import TruthMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
import contextlib
import io


# paper link: https://arxiv.org/abs/2404.10774
# hugginface link: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B

"""
    Input: Question (question_context), Answer (generated_text), Document (context)
    Output: Support score in [0, 1]
"""


class MiniCheckMethod(TruthMethod):
    #REQUIRES_NORMALIZATION = True
    
    def __init__(self, minicheck_model:str = 'flan-t5-large'):
        super().__init__()
        try:
            from minicheck.minicheck import MiniCheck
        except ImportError:
            raise ImportError("minicheck is not installed. Please install it using 'pip install minicheck[llm]@git+https://github.com/Liyan06/MiniCheck.git@main' ")
        if minicheck_model not in ['roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B']:
            raise ValueError("Available Minicheck models are one of: 'roberta-large', 'deberta-v3-large', 'flan-t5-large', 'Bespoke-MiniCheck-7B'")
        else:
            self.minicheck_model = MiniCheck(model_name=minicheck_model)
            
        

        
    def forward_hf_local(
            self,
            model: PreTrainedModel,
            input_text: str,
            generated_text: str,
            question: str,
            all_ids: Union[list, torch.Tensor],
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            generation_seed=None,
            sampled_generations_dict: dict = None,
            messages: list = [],
            context: str = "",
            **kwargs,
        ):
        if context == "":
            raise ValueError("Context is required for MiniCheck method")        
        truth_score, truth_label = self.minicheck(context, generated_text) 
        return {'truth_value': truth_score, 'binary_prediction': truth_label}


    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs,
    ):
        if context == "":
            raise ValueError("Context is required for MiniCheck method") 
                
        truth_score, truth_label = self.minicheck(context, generated_text) 
        return {'truth_value': truth_score, 'truth_label': truth_label}


    def minicheck(self, context, generated_text):
        # Loading Minicheck model
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pred_label, raw_prob, _, _ = self.minicheck_model.score(docs=[context], claims=[generated_text]) 
            return raw_prob[0], pred_label[0]