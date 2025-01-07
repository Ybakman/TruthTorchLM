from .truth_method import TruthMethod
from TruthTorchLM.utils import fix_tokenizer_chat
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import PTRUE_SYSTEM_PROMPT, PTRUE_USER_PROMPT, PTRUE_MODEL_OUTPUT
from TruthTorchLM.utils.common_utils import generate
from ..generation import sample_generations_api

import torch
import copy




VC_SYSTEM_PROMPT = 'You are a helpful, respectful and honest confidence estimator.'
VC_USER_PROMPT = '''You will be provided with a question and a corresponding answer that you generated. Your task is to evaluate your confidence in the accuracy of the provided answer. The confidence indicates how likely you think your answer is true.

The output must be a single number between 0 and 100:
- 100 indicates maximum confidence.
- 0 indicates no confidence.

Output format: Only the number, without any additional text or explanation.

Question: {question_context}
Generated Answer: {generated_text}

Your confidence score:'''
        
class VerbalizedConfidence(TruthMethod):

    
    def __init__(self, system_prompt:str = VC_SYSTEM_PROMPT, user_prompt:str = VC_USER_PROMPT, max_new_tokens=1024, **generation_kwargs):
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs

    def extract_confidence(self, confidence_text):
        '''Extracts the confidence value from the confidence text. The text may include non-numeric characters.'''
        confidence_text = confidence_text.strip()
        confidence_text = ''.join([c for c in confidence_text if c.isdigit() or c == '.'])
        try:
            return float(confidence_text)
        except:
            return 0.0

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs): 
        
        generated_text = tokenizer.decode(tokenizer.encode(generated_text, return_tensors="pt").view(-1).tolist(), skip_special_tokens=True)#remove special tokens
        chat = [{"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.user_prompt.format(question_context = question_context, generated_text = generated_text)}]
        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)#in case some tokenizers don't have chat template and don't support system prompt
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)

        kwargs = copy.deepcopy(kwargs)
        kwargs.pop('do_sample', None)
        generation_dict = generate(prompt, model, tokenizer, do_sample = False, max_new_tokens=self.max_new_tokens, **self.generation_kwargs)
        confidence_text = generation_dict['generated_text_skip_specials']
        confidence = self.extract_confidence(confidence_text)
        confidence = confidence/100.0

        return {"truth_value": confidence, "confidence_text": confidence_text}

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None,  logprobs:list=None, generated_tokens:list=None, **kwargs):
        chat = [{"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.user_prompt.format(question_context = question_context, generated_text = generated_text)}]

        sampled_generations_dict = sample_generations_api(model = model, messages = chat, generation_seed = generation_seed, 
        number_of_generations=1, return_text = True, **kwargs)
        confidence_text = sampled_generations_dict["generated_texts"][0]
        confidence = self.extract_confidence(confidence_text)
        confidence = confidence/100.0
        return {"truth_value": confidence, "confidence_text": confidence_text}
