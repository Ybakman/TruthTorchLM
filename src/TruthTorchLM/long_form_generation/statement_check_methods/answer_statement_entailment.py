from .statement_check_method import StatementCheckMethod
from TruthTorchLM.utils import check_entailment
from TruthTorchLM.truth_methods import TruthMethod
from TruthTorchLM.availability import AVAILABLE_API_MODELS, PROB_AVAILABLE_API_MODELS
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from TruthTorchLM.generation import get_sampling_properties, sample_generations_hf_local, sample_generations_api
from TruthTorchLM.normalizers import Normalizer, SigmoidNormalizer

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers import DebertaForSequenceClassification, DebertaTokenizer

import torch
from random import randint
from typing import Union
from copy import deepcopy
from litellm import completion


INSTRUCTION = [{"role": "system", "content": "You will be given a text and a follow-up sentence. Generate a question that, in the context of the preceding original text, might have generated the follow-up sentence. Please do not use specific facts that appear in the follow-up sentence when formulating the question. Provide only the text of the question with no additional text."},
                 {"role": "user", "content": '''Following this text: 
{text_so_far}

You see the sentence:

{statement}'''}]

FIRST_STATEMENT_INSTRUCTION = [{"role": "system", "content": "You will be given a question and a sentence. The sentence is part of the answer to the given question. Your goal is to generate a specific question that might have generated the sentence. Please do not use specific facts that appear in the sentence when formulating the question. The question must have a unique answer. Provide only the text of the question with no additional text."},
                 {"role": "user", "content": '''The original question:
                 
{question_context}

You see the sentence:

{statement}'''}]

GEN_ANSWER_INST = [{"role": "system", "content": 'You are a helpful assistant. Give short and precise answers.'},
                  {"role": "user", "content": ""},]

class AnswerStatementEntailment(StatementCheckMethod):
    def __init__(self, model:Union[PreTrainedModel, str], num_questions:int, num_answers_per_question:int=3, 
                 instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 first_statement_instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 generate_answer_instruction:list=GEN_ANSWER_INST,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                 entailment_model:PreTrainedModel=None, 
                 entailment_tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                 entailment_model_device = 'cuda', normalizer:Normalizer=SigmoidNormalizer(threshold = 0, std = 1.0),
                 add_generation_prompt = True, continue_final_message = False, **kwargs):
        super().__init__()

        # Check if the model is an API model
        if type(model) == str and model not in AVAILABLE_API_MODELS:
            raise ValueError(f"model {model} is not supported.")

        self.model = model
        self.tokenizer = tokenizer
        self.num_questions = num_questions
        self.instruction = instruction
        self.first_statement_instruction = first_statement_instruction
        self.generate_answer_instruction = generate_answer_instruction
        self.num_answers_per_question = num_answers_per_question
        self.entailment_model = entailment_model
        self.entailment_tokenizer = entailment_tokenizer
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message
        self.kwargs = {
                # "max_length": 50,
                "num_return_sequences": 1,
                "seed": 42,
                "do_sample":True}
        self.kwargs.update(kwargs)

        self.normalizer = normalizer

        if type(model) != str:
            self.kwargs.pop('seed', None) 
            eos_token_id = self.kwargs.pop("eos_token_id", None)
            if eos_token_id is None:    
                eos_token_id = model.config.eos_token_id
            self.kwargs['eos_token_id'] = eos_token_id

            pad_token_id = self.kwargs.pop("pad_token_id", None)
            if pad_token_id is None:
                if type(eos_token_id) == list:
                    pad_token_id = eos_token_id[0]
                else:
                    pad_token_id = eos_token_id
            self.kwargs['pad_token_id'] = pad_token_id 
        else:
            self.kwargs.pop('do_sample', None) 
            self.kwargs.pop('num_return_sequences', None) 
            self.kwargs.pop('max_length', None) 

        # print(self.kwargs)

        if self.entailment_model is None or self.entailment_tokenizer is None:
            self.entailment_model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            self.entailment_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')


    def _generate_question(self, statement:str, text_so_far:str, question_context:str):

        # print(self.kwargs)

        messages = deepcopy(self.first_statement_instruction) if text_so_far is None else deepcopy(self.instruction)
        messages[-1]["content"] = messages[-1]["content"].format(statement=statement, text_so_far=text_so_far, question_context=question_context)

        if type(self.model) == str:
            response = completion(
                model=self.model,
                messages=messages,
                **self.kwargs
            )
            question = response.choices[0].message['content']
        else:
            self.tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
            text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=self.add_generation_prompt, continue_final_message=self.continue_final_message)
            generated_output = generate(text, self.model, self.tokenizer, **self.kwargs)
            question = generated_output["generated_text_skip_specials"]

        return question.strip()
    
    def _get_questions(self, question_context:str, statement:str, text_so_far:str):
        #Generate questions
        questions = []
        question_check = []
        org_seed = self.kwargs.get('seed', None)
        for _ in range(self.num_questions):
            question = self._generate_question(statement=statement, text_so_far=text_so_far, question_context=question_context)
            if question.lower() not in question_check:
                question_check.append(question.lower())
                questions.append(question)
            if type(self.model) == str:
                seed = self.kwargs.pop('seed', None)
                self.kwargs['seed'] = seed + 1 #Increment seed to get different questions
        if org_seed is not None:
            self.kwargs['seed'] = org_seed
        # print("Questions generated:", questions)
        return questions
    
    def _does_entail(self, statement:str, question:str, answer:str)->bool:
        #Check if the question entails the answer
        implication_1 = check_entailment(self.entailment_model, self.entailment_tokenizer, question, answer, statement)
        implication_2 = check_entailment(self.entailment_model, self.entailment_tokenizer, question, statement, answer)

        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        implications = [implication_1, implication_2]
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        # semantically_equivalent = (implications[0] == 2) and (implications[1] == 2) #strict check
        return semantically_equivalent

    def check_statement_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, 
                              statement:str, text_so_far:str, all_ids:Union[list, torch.Tensor], 
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed=None, 
                              messages:list = [], add_generation_prompt = True, continue_final_message = False, **kwargs):

        questions = self._get_questions(question_context=question_context, statement=statement, text_so_far=text_so_far)
        #Get model answers for each question (generate answers until it entails the statement)
        answers = []
        entailment = []
        for i, question in enumerate(questions):
            messages = deepcopy(self.generate_answer_instruction)
            messages[1]["content"] = question
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
            #check if the answer aligns with the statement
            for _ in range(self.num_answers_per_question):
                generated_output = generate(text, model, tokenizer, **kwargs)
                answer = generated_output['generated_text_skip_specials']
                answers.append(answer)
                del generated_output

                if self._does_entail(statement=statement, question=question, answer=answer):
                    entailment.append(1)
                else:
                    entailment.append(0)

        return { "normalized_truth_values": self.normalizer(sum(entailment)/len(entailment)), "truth_values": sum(entailment)/len(entailment),
                       "questions": questions, "answers": answers, "entailment": entailment}

    def check_statement_api(self, model:str, messages:list, generated_text:str, 
                            question_context:str, statement:str, text_so_far:str, generation_seed=None, **kwargs):
        
        questions = self._get_questions(question_context=question_context, statement=statement, text_so_far=text_so_far)
        answers = []
        entailment = []
        q_messages = deepcopy(self.generate_answer_instruction)
        for i, question in enumerate(questions):
            q_messages[1]["content"] = question
            #check if the answer aligns with the statement
            for _ in range(self.num_answers_per_question):
                response = completion(
                    model=model,
                    messages=q_messages,
                    **kwargs
                )
                answer = response.choices[0].message['content']
                answers.append(answer)
                if self._does_entail(statement=statement, question=question, answer=answer):
                    entailment.append(1)
                else:
                    entailment.append(0)

        return { "normalized_truth_values": self.normalizer(sum(entailment)/len(entailment)), "truth_values": sum(entailment)/len(entailment),
                       "questions": questions, "answers": answers, "entailment": entailment}

    def __str__(self): 

        model_name = self.model.__class__ if type(self.model) != str else self.model
        ent_model_name = self.entailment_model.__class__ if type(self.entailment_model) != str else self.entailment_model

        return f"Statement Check Method by Entailment Check with Answers and statement. Answers generated to the generated questions.\n\
Question generation model: {model_name}\n\
Number of questions to be generated for a stament: {self.num_questions}\n\
Number of answers per question: {self.num_answers_per_question}\n\
Entailment check model: {ent_model_name}\n\n\
Question generation instruction for the first statement:\n  {self.first_statement_instruction}\n\n\
Question generation instruction for statements with preceeding text:\n  {self.instruction}\n\n\
Answer generation instruction:\n    {self.generate_answer_instruction}"