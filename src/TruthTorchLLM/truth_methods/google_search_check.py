from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLLM.utils import sigmoid_normalization
from TruthTorchLLM.utils.google_search_utils import GoogleSerperAPIWrapper,extract_list_from_string,extract_dict_from_string,type_check
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLLM.templates import GOOGLE_CHECK_QUERY_SYSTEM_PROMPT, GOOGLE_CHECK_QUERY_USER_PROMPT, GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT, GOOGLE_CHECK_VERIFICATION_USER_PROMPT

import torch
import numpy as np
import copy


#TODO:move std and threshold to the superclass
class GoogleSearchCheck(TruthMethod):
    def __init__(self, threshold:float = 0.0, std:float = 1.0, number_of_snippets:int = 10, location:str = 'us', language:str = 'en') -> None:
        super().__init__()
        self.threshold = threshold
        self.std = std
        self.number_of_snippets = number_of_snippets
        self.location = location
        self.language = language
        self.google_serper = GoogleSerperAPIWrapper(snippet_cnt = self.number_of_snippets, location = self.location, language = self.language)


       


    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        generated_text = tokenizer.decode(tokenizer.encode(generated_text, return_tensors="pt").view(-1).tolist(), skip_special_tokens=True)#remove special tokens
        #first we need to generate search queries
        chat = [{"role": "system", "content": GOOGLE_CHECK_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_QUERY_USER_PROMPT.format(question_context = question_context, input = generated_text)}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        model_output = model.generate(input_ids)
        
        tokens = model_output[0][len(input_ids[0]):]
        query_text = tokenizer.decode(tokens, skip_special_tokens=True)
        query = extract_list_from_string(query_text)
        query_list = type_check(query, list)
        if query_list != None:
            #search the queries
            search_results = self.google_serper.run(query_list)
            evidences = [[output['content'] for output in search_result] for search_result in search_results]
                    
        else:
            evidences = []
            print("The model output didn't match the output format while creating the query")

        #Ask model to verify the claim
        chat = [{"role": "system", "content": GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_VERIFICATION_USER_PROMPT.format(question_context = question_context, claim = generated_text, evidence = evidences)}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
       
        model_output = model.generate(input_ids)
        
        tokens = model_output[0][len(input_ids[0]):]
        verification_text = tokenizer.decode(tokens, skip_special_tokens=True)
        verification = extract_dict_from_string(verification_text)
        verification_dict = type_check(verification, dict)

        if verification_dict == None:
            return {"truth_value": 0.5, 'normalized_truth_value': 0.5, 'evidences':evidences, 'query_text':query_text, 'evidences':evidences, 'verification_text':verification}
            print("The model output didn't match the output format in verification")
        else:
            try:
                if  verification_dict['factuality'] == True:
                    truth_value = 1.0
                    normalized_truth_value = 1.0
                else:
                    truth_value = 0.0
                    normalized_truth_value = 0.0
            except:
                truth_value = 0.5
                normalized_truth_value = 0.5
                print("The model output didn't match the output format in verification")
            

            return {"truth_value": truth_value, 'normalized_truth_value': normalized_truth_value, 'evidences':evidences, 'query_text':query_text, 'evidences':evidences, 'verification_text':verification, 'verification':verification_dict}





    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        #first we need to generate search queries
        chat = [{"role": "system", "content": GOOGLE_CHECK_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_QUERY_USER_PROMPT.format(question_context = question_context, input = generated_text)}]
        
        response = completion(
                model=model,
                messages=chat,
            )
        query_text = response.choices[0].message['content']
        
        query = extract_list_from_string(query_text)
        query_list = type_check(query, list)
        if query_list != None:
            #search the queries
            search_results = self.google_serper.run(query_list)
            evidences = []
            for search_result in search_results:
                for output in search_result:
                    evidences.append(output['content'])
            #evidences = [[output['content'] for output in search_result] for search_result in search_results]
                    
        else:
            evidences = []
            print("The model output didn't match the output format while creating the query")

        #Ask model to verify the claim
        chat = [{"role": "system", "content": GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_VERIFICATION_USER_PROMPT.format(question_context = question_context, claim = generated_text, evidence = evidences)}]
        response = completion(
                model=model,
                messages=chat,
            )
        verification_text = response.choices[0].message['content']
        verification = extract_dict_from_string(verification_text)

        #handle capital cases
        verification = verification.replace("true", "True")
        verification = verification.replace("false", "False")
        verification_dict = type_check(verification, dict)

        if verification_dict == None:
            print("The model output didn't match the output format in verification")
            return {"truth_value": 0.5, 'normalized_truth_value': 0.5, 'evidences':evidences, 'query_text':query_text, 'evidences':evidences, 'verification_text':verification}
        else:
            try:
                if  verification_dict['factuality'] == True:
                    truth_value = 1.0
                    normalized_truth_value = 1.0
                else:
                    truth_value = 0.0
                    normalized_truth_value = 0.0
            except:
                truth_value = 0.5
                normalized_truth_value = 0.5
                print("The model output didn't match the output format in verification")
            

            return {"truth_value": truth_value, 'normalized_truth_value': normalized_truth_value, 'evidences':evidences, 'query_text':query_text, 'evidences':evidences, 'verification_text':verification, 'verification':verification_dict}
        

    def __str__(self):
        return f"GoogleSearchCheck with threshold {self.threshold} and std {self.std} and number of snippets {self.number_of_snippets} and location {self.location} and language {self.language}"
        

    

        