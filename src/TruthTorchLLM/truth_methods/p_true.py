from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLLM.utils import find_keys_of_template

import torch
import numpy as np
import copy
import random

#for a target text, find the indices of the tokens that are in the target text. 
#If target text cannot be tokenized in the original form, return the indices of the tokens that contain the target text and has the shortest length
def find_token_indices(tokens:list, tokenizer:PreTrainedTokenizer, target_text:str, ):
    indices = []
    texts = []
    begin = 0
    found = False
    while begin < len(tokens):
        for end in range(begin+1, len(tokens)):
            if  target_text in tokenizer.decode(tokens[begin:end]):
                #move begin
                while target_text in tokenizer.decode(tokens[begin:end]):
                    begin += 1
                begin -= 1
                index_list = [i for i in range(begin, end)]
                indices.append(index_list)
                texts.append(tokenizer.decode(tokens[begin:end]))
                begin = end
                found = True
                break
        if not found:
            break
        else:
            found = False
    return indices, texts
        

#TODO:allow the user set their templates
class PTrue(TruthMethod):
    def __init__(self, number_of_ideas: int = 5, threshold:float = 0.0, std:float = 1.0):
        super().__init__()
        self.number_of_ideas= number_of_ideas
        self.threshold = threshold
        self.std = std


    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        scores = []
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
        chat = [{"role": "system", "content": 'You are a helpful, respectful and honest question-answer evaluator. You will be given a question, some brainstormed ideas and a generated answer. Evaluate the generate answer as true or false considering the question and brainstormed ideas. Output "The generated answer is true" or "The generated answer is false".'},
        {"role": "user", "content": f'Question:{question_context}\nHere are some ideas that were brainstormed:{ideas}\nGenerated answer:{generated_text}'},
        {"role": "assistant", "content": 'The generated answer is true'}]

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

    def completion_forward(self, model:str, input_text:str, generated_text:str, **kwargs):
        raise ValueError("PTrue is not applicable to API models. Please use a different truth method.")

    def __str__(self):
        return f"PTrue with template {self.template} and number of ideas {self.number_of_ideas}."

        