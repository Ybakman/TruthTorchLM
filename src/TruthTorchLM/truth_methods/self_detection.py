import copy
import torch
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLM.availability import AVAILABLE_API_MODELS
from TruthTorchLM.utils import *
from .truth_method import TruthMethod
from TruthTorchLM.templates import SELF_DETECTION_QUESTION_PROMPT, SELF_DETECTION_SYSTEM_PROMPT, ENTAILMENT_PROMPT, DEFAULT_SYSTEM_PROMPT 
from TruthTorchLM.generation import sample_generations_hf_local, sample_generations_api


#https://arxiv.org/pdf/2310.17918

class SelfDetection(TruthMethod):
    def __init__(self, output_type:str = 'entropy',method_for_similarity: str = "semantic", number_of_questions=5, model_for_entailment: PreTrainedModel = None, 
    tokenizer_for_entailment: PreTrainedTokenizer = None, prompt_for_generating_question = SELF_DETECTION_QUESTION_PROMPT, 
    system_prompt = SELF_DETECTION_SYSTEM_PROMPT, prompt_for_entailment:str = ENTAILMENT_PROMPT, system_prompt_for_entailment:str = DEFAULT_SYSTEM_PROMPT, batch_generation = True, question_max_new_tokens = 64, question_temperature = 1.0):
        super().__init__()

        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment

        if (model_for_entailment is None or tokenizer_for_entailment is None) and method_for_similarity == "semantic": 
            self.model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
            self.tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

        self.number_of_questions = number_of_questions
        self.prompt_for_generating_question = prompt_for_generating_question
        self.system_prompt = system_prompt
        self.prompt_for_entailment = prompt_for_entailment
        self.system_prompt_for_entailment = system_prompt_for_entailment
        self.batch_generation = batch_generation
        self.question_max_new_tokens = question_max_new_tokens
        self.question_temperature = question_temperature
        

        if output_type not in ['entropy', 'consistency']:
            raise ValueError("output_type should be either 'entropy' or 'consistency'")
        self.output_type = output_type

        if method_for_similarity not in ["generation", "semantic", "jaccard"]:
            raise ValueError("method_for_similarity should be either 'generation' or 'semantic' or 'jaccard'")
        self.method_for_similarity = method_for_similarity

    def generate_similar_questions(self, input_text: str, prompt_for_generating_question: str = None, system_prompt:str = None, model= None, tokenizer = None, generation_seed = 0):  
        generated_questions = [input_text]
        chat = [{"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.prompt_for_generating_question.format(question = input_text)}]
        if type(model) != str:
            tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
            input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt = True)
            sampled_generations_dict = sample_generations_hf_local(model, input_text, tokenizer, number_of_generations=self.number_of_questions, return_text=True, generation_seed=generation_seed, 
            max_new_tokens=self.question_max_new_tokens, temperature=self.question_temperature, batch_generation=self.batch_generation)
        if type(model) == str:
            sampled_generations_dict = sample_generations_api(model, chat, number_of_generations=self.number_of_questions, return_text=True, generation_seed=generation_seed, temperature=self.question_temperature)

        return sampled_generations_dict["generated_texts"]

    def _self_detection_output(self,model, tokenizer, generated_texts:list, question_context:str, generated_questions):#TODO: check the similarity method's correctness
        if self.method_for_similarity == "semantic":
            clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts, 
            self.method_for_similarity, entailment_prompt= self.prompt_for_entailment, system_prompt=self.system_prompt_for_entailment)
        else:
            clusters = bidirectional_entailment_clustering(model, tokenizer, question_context, generated_texts, self.method_for_similarity, entailment_prompt= self.prompt_for_entailment, system_prompt=self.system_prompt_for_entailment)

        entropy = 0
        for cluster in clusters:
            entropy -= len(cluster)/self.number_of_questions * np.log(len(cluster)/self.number_of_questions)
        consistency = (len(clusters[0])-1)/(self.number_of_questions - 1)

        if self.output_type == 'entropy':
            truth_value = -entropy
        elif self.output_type == 'consistency':
            truth_value = consistency

        return {"truth_value": truth_value, 'entropy': entropy, "consistency": consistency, 
        "generated_questions": generated_questions, 'generated_texts': generated_texts, "clusters": clusters}


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):       
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        generated_questions = self.generate_similar_questions(question_context, self.prompt_for_generating_question, model=model, tokenizer=tokenizer, generation_seed = generation_seed)

        for generated_question in generated_questions:
            
            chat = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": generated_question}]
            tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)#do sample?
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

        return self._self_detection_output(model, tokenizer, generated_texts, question_context, generated_questions)

        

    
    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        if model not in AVAILABLE_API_MODELS:
            raise ValueError("This method is not applicable to given model")
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        generated_questions = self.generate_similar_questions( input_text=question_context, prompt_for_generating_question=self.prompt_for_generating_question, system_prompt = self.system_prompt, model= model, generation_seed = generation_seed)

        for generated_question in generated_questions:
            if self.system_prompt is None:
                chat = [
                {"role": "user", "content": generated_question}]
            else:
                chat = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": generated_question}]
            response = completion(
                model=model,
                messages= chat,
                **kwargs
            )
            generated_texts.append(response.choices[0].message['content'])

       
        return self._self_detection_output(model, None, generated_texts, question_context, generated_questions)
        