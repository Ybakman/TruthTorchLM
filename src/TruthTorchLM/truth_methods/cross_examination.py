from enum import Enum
from .truth_method import TruthMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.utils import fix_tokenizer_chat
from litellm import completion
import copy
from typing import Union
import torch


#paper link: https://arxiv.org/pdf/2305.13281

EXAMINEE_PROMPT = """Please answer the following questions regarding your claim: <questions>"""
SETUP_TEMPLATE = """Your goal is to try to verify the correctness of the following claim:<C>, based on the background information you will gather. To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. Please keep asking questions as long as you’re yet to be sure regarding the true veracity of the claim. Please start with the first questions."""
PROVIDE_ANSWERS_TEMPLATE = "Here are the answers to your questions: <answers>"
FOLLOWUP_TEMPLATE_1 = """Do you have any follow-up questions? Please answer with Yes or No."""
FOLLOWUP_TEMPLATE_2 = """What are the follow-up questions?"""
FACTUAL_DECISION_TEMPLATE = """Based on the interviewee’s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect?"""
class Examiner_Stage(Enum):
    SETUP, FOLLOWUP, DECISION = range(3)


class CrossExamination(TruthMethod):
    REQUIRES_NORMALIZATION = False

    def __init__(self, model_examiner = None,  tokenizer_examiner: PreTrainedTokenizer = None, follow_up_turns_threshold:int=2, 
    max_new_tokens=1024, temperature=1.0, top_k=50, num_beams=1, **generation_kwargs):
        super().__init__()
        if model_examiner == None:
            print("No examiner model is provided. Defaulting to GPT-4o-mini.")
            self.model_examiner = "gpt-4o-mini"
        
        if type(model_examiner) != str and model_examiner != None and tokenizer_examiner == None:
            raise ValueError("tokenizer_examiner is not provided.")
        
        self.model_examiner = model_examiner
        self.tokenizer_examiner = tokenizer_examiner
        self.follow_up_turns_threshold = follow_up_turns_threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_beams = num_beams
        self.generation_kwargs = generation_kwargs


    def examiner_inference(self, examiner_messages, output_from_examinee:str, examiner_stage:Examiner_Stage):
        is_it_over = False
        if examiner_stage.name == Examiner_Stage.SETUP.name:
            examiner_messages.append({"role":"user", "content":SETUP_TEMPLATE.replace("<C>",output_from_examinee)})
        elif examiner_stage.name == Examiner_Stage.FOLLOWUP.name:
            examiner_messages.append({"role":"user", "content":PROVIDE_ANSWERS_TEMPLATE.replace("<answers>",output_from_examinee)})
            examiner_messages.append({"role":"user", "content":FOLLOWUP_TEMPLATE_1})
        elif examiner_stage.name == Examiner_Stage.DECISION.name:
            examiner_messages.append({"role":"user", "content":FACTUAL_DECISION_TEMPLATE})
        examiner_messages = self._examiner_inference(examiner_messages)
        if examiner_stage.name == Examiner_Stage.SETUP.name:
            pass # Do nothing
        elif examiner_stage.name == Examiner_Stage.FOLLOWUP.name:
            if "yes" in examiner_messages[-1]['content'].lower():
                examiner_messages.append({"role":"user", "content":FOLLOWUP_TEMPLATE_2})
                examiner_messages = self._examiner_inference(examiner_messages)
            else:
                is_it_over=True
        elif examiner_stage.name == Examiner_Stage.DECISION.name:
            is_it_over=True        
        
        return is_it_over, examiner_messages
    
    def _examiner_inference(self, examiner_messages):
        if type(self.model_examiner) == str:
            examiner_response = completion(
                model = self.model_examiner,
                messages = examiner_messages
            )
            examiner_messages.append({"role":"assistant", "content":examiner_response.choices[0].message['content']})
            return examiner_messages
        else:
            text = self.tokenizer_examiner.apply_chat_template(examiner_messages, tokenize = False)
            input_ids = self.tokenizer_examiner.encode(text, return_tensors="pt").to(self.model_examiner.device)
            model_output = self.model_examiner.generate(input_ids, max_new_tokens=self.max_new_tokens, temperature = self.temperature, top_k = self.top_k, num_beams = self.num_beams, **self.generation_kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = self.tokenizer_examiner.decode(tokens, skip_special_tokens = False)
            examiner_messages.append({"role":"assistant", "content":generated_text})
            return examiner_messages

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        
        kwargs = copy.deepcopy(kwargs)
        
        messages.append({"role":"assistant", "content":generated_text})
        examiner_messages = []
        _, examiner_messages = self.examiner_inference(examiner_messages, messages[-1]['content'], Examiner_Stage.SETUP)
        
        messages.append({"role":"user", "content":EXAMINEE_PROMPT.replace("<questions>",examiner_messages[-1]['content'])})
        examinee_response = completion(
                model = model,
                messages = messages,
                **kwargs
        )
        messages.append({"role":"assistant", "content":examinee_response.choices[0].message['content']})
        for i in range(self.follow_up_turns_threshold):
            is_it_over, examiner_messages = self.examiner_inference(examiner_messages, messages[-1]['content'], Examiner_Stage.FOLLOWUP)
            if not is_it_over:
                messages.append({"role":"user", "content":EXAMINEE_PROMPT.replace("<questions>",examiner_messages[-1]['content'])})
                examinee_response = completion(
                        model = model,
                        messages = messages,
                        **kwargs
                )
                messages.append({"role":"assistant", "content":examinee_response.choices[0].message['content']})
                if i+1 == self.follow_up_turns_threshold:
                    examiner_messages.append({"role":"user", "content":PROVIDE_ANSWERS_TEMPLATE.replace("<answers>",messages[-1]['content'])})
            else:
                break
        _, examiner_messages = self.examiner_inference(examiner_messages, "", Examiner_Stage.DECISION)
        if "correct" in examiner_messages[-1]['content'].lower() and "incorrect" not in examiner_messages[-1]['content'].lower():
            score = 1
        else:
            score = 0

        return {"truth_value": score, "generated_text": generated_text, "examiner_messages": examiner_messages, 'messages': messages}
    
    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
                         tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs): 
             
        kwargs = copy.deepcopy(kwargs)
        messages.append({"role":"assistant", "content":generated_text})
        examiner_messages = []
        _, examiner_messages = self.examiner_inference(examiner_messages, messages[-1]['content'], Examiner_Stage.SETUP)
        
        messages.append({"role":"user", "content":EXAMINEE_PROMPT.replace("<questions>",examiner_messages[-1]['content'])})
        tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
        text = tokenizer.apply_chat_template(messages, tokenize = False)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        model_output = model.generate(input_ids, max_new_tokens=self.max_new_tokens, temperature = self.temperature, top_k = self.top_k, num_beams = self.num_beams, **self.generation_kwargs)
        tokens = model_output[0][len(input_ids[0]):]
        generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
        messages.append({"role":"assistant", "content":generated_text})
        for i in range(self.follow_up_turns_threshold):
            is_it_over, examiner_messages = self.examiner_inference(examiner_messages, messages[-1]['content'], Examiner_Stage.FOLLOWUP)
            if not is_it_over:
                messages.append({"role":"user", "content":EXAMINEE_PROMPT.replace("<questions>",examiner_messages[-1]['content'])})
                tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
                text = tokenizer.apply_chat_template(messages, tokenize = False)
                input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
                model_output = model.generate(input_ids, max_new_tokens=self.max_new_tokens, temperature = self.temperature, top_k = self.top_k, num_beams = self.num_beams, **self.generation_kwargs)
                tokens = model_output[0][len(input_ids[0]):]
                generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
                messages.append({"role":"assistant", "content":generated_text})
                if i+1 == self.follow_up_turns_threshold:
                    examiner_messages.append({"role":"user", "content":PROVIDE_ANSWERS_TEMPLATE.replace("<answers>",messages[-1]['content'])})
            else:
                break
        _, examiner_messages = self.examiner_inference(examiner_messages, "", Examiner_Stage.DECISION)
        if "correct" in examiner_messages[-1]['content'].lower() and "incorrect" not in examiner_messages[-1]['content'].lower():
            score = 1
        else:
            score = 0
        
        return {"truth_value": score, "generated_text": generated_text, "examiner_messages": examiner_messages, 'messages': messages}

   