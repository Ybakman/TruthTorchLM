from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization, find_token_indices
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLLM.templates import PTRUE_SYSTEM_PROMPT, PTRUE_USER_PROMPT, PTRUE_MODEL_OUTPUT

import torch
import numpy as np
import copy
import random


        
class PTrue(TruthMethod):
    def __init__(self, number_of_ideas: int = 5, threshold:float = 0.0, std:float = 1.0, system_prompt:str = PTRUE_SYSTEM_PROMPT, user_prompt:str = PTRUE_USER_PROMPT, model_output:str = PTRUE_MODEL_OUTPUT):
        super().__init__(threshold = threshold, std = std)
        self.number_of_ideas= number_of_ideas
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output


    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_text = tokenizer.decode(tokenizer.encode(generated_text, return_tensors="pt").view(-1).tolist(), skip_special_tokens=True)#remove special tokens
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        ideas = []
        for i in range(self.number_of_ideas):
            model_output = model.generate(input_ids, num_return_sequences = 1, do_sample = True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            ideas.append(text)

        ideas = "\n".join(ideas)
        chat = [{"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.user_prompt.format(question_context = question_context, ideas = ideas, generated_text = generated_text)},
        {"role": "assistant", "content": self.model_output}]

        
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

        normalized_truth_value = sigmoid_normalization(prob_true, self.threshold, self.std)
        return {"truth_value": prob_true, 'normalized_truth_value':normalized_truth_value,  'p_true': prob_true,  'generated_ideas': ideas}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, **kwargs):
        raise ValueError("PTrue is not applicable to API models. Please use a different truth method.")

    def __str__(self):
        return f"PTrue with template {self.template} and number of ideas {self.number_of_ideas}."

        