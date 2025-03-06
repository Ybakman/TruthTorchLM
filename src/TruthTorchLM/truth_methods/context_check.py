import torch
from typing import Union
import copy
from .truth_method import TruthMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from litellm import completion

# Inspired by Minicheck - paper link: https://arxiv.org/abs/2404.10774

"""
    Input: Question (question_context), Answer (generated_text), Document (context)
    Output: Support score in [0, 1]
"""

DOCUMENT = "Document: <document>"
CLAIM = "Claim: <claim>"
JUDGE_PROMPT = """You are an expert in fact-checking and entailment classification. 
                Given a background document and a claim, your task is to assess how well the claim is supported by the document.

                ### Task:
                1. Read the provided **document** carefully.
                2. Analyze the **claim** and compare it with the document to determine the level of support.
                3. Assign a **truth score** between 0 and 1, where:
                   - **0** means the document **does not support** the claim.
                   - **1** means the document **fully supports** the claim.
                   - Any value in between reflects partial support.
                
                ### Important:
                - Your response **must be a single number between 0 and 1**.
                - Do **not** include any explanation or additional text.
                - Only return the numerical score as plain text.
"""


class ContextCheck(TruthMethod):
    REQUIRES_SAMPLED_TEXT = False  
    REQUIRES_SAMPLED_LOGPROBS = False   
    REQUIRES_NORMALIZATION = False
    
    def __init__(self, check_model:Union[PreTrainedModel, str], 
                 check_tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, temperature=1.0, **generation_kwargs):
        super().__init__()
        self.check_model = check_model
        self.check_tokenizer = check_tokenizer
        self.temperature = temperature


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
        
        if isinstance(self.check_model, str):
            truth_score, truth_label = self.api_context_check(context, generated_text, self.check_model)
        else:
            truth_score, truth_label = self.hf_local_context_check(context, generated_text, self.check_model, self.check_tokenizer, **kwargs)
        
        return {'truth_value:': truth_score, 'truth_label': truth_label}
                
    
    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        
        if isinstance(self.check_model, str):
            truth_score, truth_label = self.api_context_check(context, generated_text, self.check_model)
        else:
            truth_score, truth_label = self.hf_local_context_check(context, generated_text, self.check_model, self.check_tokenizer, **kwargs)
            
        return {'truth_value:': truth_score, 'truth_label': truth_label}



    def hf_local_context_check(self, context, generated_text, model, tokenizer, **kwargs):
        judge_content = JUDGE_PROMPT
        judge_content += DOCUMENT.replace('<document>', context)
        judge_content += CLAIM.replace('<claim>', generated_text)
        judge_prompt = [{"role": "user", "content": judge_content}]
        
        text = tokenizer.apply_chat_template(judge_prompt, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=self.temperature,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )
        tokens = model_output[0][len(input_ids[0]):]
        truth_score = tokenizer.decode(tokens, skip_special_tokens=False)
        try:
            truth_score = float(truth_score)
            truth_label = 1 if truth_score >= 0.5 else 0
        except ValueError:
            raise ValueError(f"Expected a numerical response between 0 and 1, but got: {truth_score}")
        return truth_score, truth_label
    
    def api_context_check(self, context, generated_text, model):
        judge_content = JUDGE_PROMPT
        judge_content += DOCUMENT.replace('<document>', context)
        judge_content += CLAIM.replace('<claim>', generated_text)
        judge_prompt = [{"role": "user", "content": judge_content}]
    
        truth_score = completion(model = model, messages = judge_prompt, temperature=self.temperature)
        truth_score = truth_score.choices[0].message.content
        try:
            truth_score = float(truth_score)
            truth_label = 1 if truth_score >= 0.5 else 0
        except ValueError:
            raise ValueError(f"Expected a numerical response between 0 and 1, but got: {truth_score}")
        return truth_score, truth_label