from .truth_method import TruthMethod
from TruthTorchLM.utils import fix_tokenizer_chat
from TruthTorchLM.utils.google_search_utils import GoogleSerperAPIWrapper,extract_list_from_string,extract_dict_from_string,type_check
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import GOOGLE_CHECK_QUERY_SYSTEM_PROMPT, GOOGLE_CHECK_QUERY_USER_PROMPT, GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT, GOOGLE_CHECK_VERIFICATION_USER_PROMPT

import torch
import copy


class GoogleSearchCheck(TruthMethod):
    REQUIRES_NORMALIZATION = False

    def __init__(self, number_of_snippets:int = 10, location:str = 'us', language:str = 'en', 
    check_query_system_prompt:str = GOOGLE_CHECK_QUERY_SYSTEM_PROMPT, check_query_user_prompt:str = GOOGLE_CHECK_QUERY_USER_PROMPT,
    check_verification_system_prompt:str = GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT, check_verification_user_prompt:str = GOOGLE_CHECK_VERIFICATION_USER_PROMPT, 
    max_new_tokens=1024, temperature=1.0, top_k=50, num_beams=1, **generation_kwargs) -> None:
        super().__init__()
        self.number_of_snippets = number_of_snippets
        self.location = location
        self.language = language
        self.google_serper = GoogleSerperAPIWrapper(snippet_cnt = self.number_of_snippets, location = self.location, language = self.language)
        self.check_query_system_prompt = check_query_system_prompt
        self.check_query_user_prompt = check_query_user_prompt
        self.check_verification_system_prompt = check_verification_system_prompt
        self.check_verification_user_prompt = check_verification_user_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_beams = num_beams
        self.generation_kwargs = generation_kwargs


    def get_evidences(self, query_text:str):
        query = extract_list_from_string(query_text)
        query_list = type_check(query, list)
        if query_list != None:
            #search the queries
            search_results = self.google_serper.run(query_list)
            evidences = [[output['content'] for output in search_result] for search_result in search_results]
                    
        else:
            evidences = []
            print("The model output didn't match the output format while creating the query")
        return evidences

    def _google_search_check(self, verification_text:str, evidences:list, query_text:str):
        verification = extract_dict_from_string(verification_text)
        #handle capital cases
        if verification != None:
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
                    
                else:
                    truth_value = 0.0
                  
            except:
                truth_value = 0.5
             
                print("The model output didn't match the output format in verification")
            
            return {"truth_value": truth_value,  'evidences':evidences, 'query_text':query_text, 'evidences':evidences, 'verification_text':verification, 'verification':verification}

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):
     
        kwargs = copy.deepcopy(kwargs)
        generated_text = tokenizer.decode(tokenizer.encode(generated_text, return_tensors="pt").view(-1).tolist(), skip_special_tokens=True)#remove special tokens
        #first we need to generate search queries

        
        chat = [{"role": "system", "content": self.check_query_system_prompt},
        {"role": "user", "content": self.check_query_user_prompt.format(question_context = question_context, input = generated_text)}]
        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)


        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        model_output = model.generate(input_ids, max_new_tokens = self.max_new_tokens, temperature = self.temperature, top_k = self.top_k, num_beams = self.num_beams, **self.generation_kwargs)
        
        tokens = model_output[0][len(input_ids[0]):]
        query_text = tokenizer.decode(tokens, skip_special_tokens=True)

        evidences = self.get_evidences(query_text) 

        #Ask model to verify the claim
        
        chat = [{"role": "system", "content": self.check_verification_system_prompt},
        {"role": "user", "content": self.check_verification_user_prompt.format(question_context = question_context, claim = generated_text, evidence = evidences)}]
        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
        
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
       
        model_output = model.generate(input_ids, max_new_tokens = self.max_new_tokens, temperature = self.temperature, top_k = self.top_k, num_beams = self.num_beams, **self.generation_kwargs)
        
        tokens = model_output[0][len(input_ids[0]):]
        verification_text = tokenizer.decode(tokens, skip_special_tokens=True)

        return self._google_search_check(verification_text, evidences, query_text)


    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        #first we need to generate search queries
        chat = [{"role": "system", "content": GOOGLE_CHECK_QUERY_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_QUERY_USER_PROMPT.format(question_context = question_context, input = generated_text)}]
        
        response = completion(
                model=model,
                messages=chat,

            )
        query_text = response.choices[0].message['content']

        evidences = self.get_evidences(query_text)        
      

        #Ask model to verify the claim
        chat = [{"role": "system", "content": GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": GOOGLE_CHECK_VERIFICATION_USER_PROMPT.format(question_context = question_context, claim = generated_text, evidence = evidences)}]
        response = completion(
                model=model,
                messages=chat,
            )
        verification_text = response.choices[0].message['content']

        return self._google_search_check(verification_text, evidences, query_text)

        