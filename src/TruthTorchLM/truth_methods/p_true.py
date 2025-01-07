from .truth_method import TruthMethod
from TruthTorchLM.utils import find_token_indices, fix_tokenizer_chat
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import PTRUE_SYSTEM_PROMPT, PTRUE_USER_PROMPT, PTRUE_MODEL_OUTPUT
from ..generation import sample_generations_hf_local

import torch
import numpy as np


        
class PTrue(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True
    
    def __init__(self, number_of_ideas: int = 5, system_prompt:str = PTRUE_SYSTEM_PROMPT, user_prompt:str = PTRUE_USER_PROMPT, model_output:str = PTRUE_MODEL_OUTPUT, batch_generation = True):
        super().__init__()
        self.number_of_ideas= number_of_ideas
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output
        self.batch_generation = batch_generation

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs): 
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model, input_text, tokenizer, [self], generation_seed, **kwargs)
        
        generated_text = tokenizer.decode(tokenizer.encode(generated_text, return_tensors="pt").view(-1).tolist(), skip_special_tokens=True)#remove special tokens
   
        ideas = sampled_generations_dict["generated_texts"][:self.number_of_ideas]
        ideas = "\n".join(ideas)


       
        chat = [{"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.user_prompt.format(question_context = question_context, ideas = ideas, generated_text = generated_text)},
        {"role": "assistant", "content": self.model_output}]
        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)#in case some tokenizers don't have chat template and don't support system prompt

        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(prompt_tokens)
            logits = outputs.logits # Logits for each token in the input

        logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
        logprobs = logprobs[0, :-1, :]#logprobs for each token except the last one
        logprobs = torch.gather(logprobs, dim=1, index = prompt_tokens[0][1:].view(-1, 1))#logprobs for each token in the generated text
        logprobs = logprobs.view(-1).tolist()#convert to list

        #write a function to find the probability of token 'true' in the logprobs
        indices, texts = find_token_indices(prompt_tokens[0][1:], tokenizer, "true")
        loss_true = 0
        for index in indices[-1]:#only look at the last occurence of the word true
            loss_true += logprobs[index]

        loss_true = loss_true / len(indices[-1])#length normalization
        prob_true = np.exp(loss_true).item()

        return {"truth_value": prob_true, 'p_true': prob_true,  'generated_ideas': ideas}#this output format should be same for all truth methods

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        raise ValueError("PTrue is not applicable to API models. Please use a different truth method.")

