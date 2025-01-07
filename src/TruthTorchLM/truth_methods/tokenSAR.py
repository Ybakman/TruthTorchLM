from .truth_method import TruthMethod

from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from sentence_transformers import CrossEncoder
from .truth_method import TruthMethod

from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np



class TokenSAR(TruthMethod):

    REQUIRES_LOGPROBS = True

    def __init__(self, tokenizer:PreTrainedTokenizer=None, similarity_model=None, similarity_model_device = 'cuda'): #normalization, 
        super().__init__()
        self.tokenizer = tokenizer
        if similarity_model is None:
            self.similarity_model = CrossEncoder('cross-encoder/stsb-roberta-large', device=similarity_model_device)
        else:
            self.similarity_model = similarity_model


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        model_output = all_ids
        tokens = model_output[0][len(input_ids[0]):]
        tokens_text = [tokenizer.decode(token) for token in tokens]

        with torch.no_grad():
            outputs = model(model_output)
            logits = outputs.logits  # Logits for each token in the input

            # Calculate probabilities from logits
            logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
            logprobs = logprobs[0, len(input_ids[0])-1:-1, :]#logprobs for each token in the generated text
            logprobs = torch.gather(logprobs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
            logprobs = logprobs.view(-1).tolist()#convert to list

            importance_vector = []
            for i in range(len(tokens)):
                removed_answer_ids = tokens[:i] + tokens[i+1:]
                removed_answer = self.tokenizer.decode(removed_answer_ids , skip_special_tokens=True)
                score = self.similarity_model.predict([( question_context +" "+removed_answer, question_context + ' ' + generated_text)])
                score = 1 - score[0]
                importance_vector.append(score)

            importance_vector = importance_vector / np.sum(importance_vector)
            score =  np.dot(importance_vector, logprobs)
                
        return {"truth_value": score,  "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):

        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("TokenSAR method is not applicable to given model")

        importance_vector = []
        for i in range(len(generated_tokens)):
            removed_answer = "".join(generated_tokens[:i]) + "".join(generated_tokens[i+1:])
            score = self.similarity_model.predict([( question_context +" "+removed_answer, question_context + ' ' + generated_text)])
            score = 1 - score[0]
            importance_vector.append(score)

        importance_vector = importance_vector / np.sum(importance_vector)
        score =  np.dot(importance_vector, logprobs)

        return {"truth_value": score, "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
