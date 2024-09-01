from .statement_check_method import StatementCheckMethod
from TruthTorchLLM.utils import check_entailment
from TruthTorchLLM.truth_methods import TruthMethod
from TruthTorchLLM.availability import AVAILABLE_API_MODELS

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

class QuestionGeneration(StatementCheckMethod):
    def __init__(self, model:Union[PreTrainedModel, str], num_questions:int, max_answer_trials:int=3, 
                 instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 first_statement_instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 generate_answer_instruction:list=GEN_ANSWER_INST,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                 truth_method:TruthMethod=None, entailment_model:PreTrainedModel=None, 
                 entailment_tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, **kwargs):
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
        self.max_answer_trials = max_answer_trials
        self.truth_method = truth_method
        self.entailment_model = entailment_model
        self.entailment_tokenizer = entailment_tokenizer
        self.kwargs = kwargs

        if self.entailment_model is None or self.entailment_tokenizer is None:
            self.entailment_model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
            self.entailment_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

    def __str__(self): 

        model_name = self.model.__class__ if type(self.model) != str else self.model
        ent_model_name = self.entailment_model.__class__ if type(self.entailment_model) != str else self.entailment_model

        return f"Statement Check Method by Generating Questions.\n\
Question generation model: {model_name}\n\
Number of questions to be generated for a stament: {self.num_questions}\n\
Number of trials to generate an answer that entails with the statement: {self.max_answer_trials}\n\
Entailment check model: {ent_model_name}\n\n\
Question generation instruction for the first statement:\n  {self.first_statement_instruction}\n\n\
Question generation instruction for statements with preceeding text:\n  {self.instruction}\n\n\
Answer generation instruction:\n    {self.generate_answer_instruction}\n\n\
Truth method to assign a score the question(s):\n   {self.truth_method}"

    def generate_question(self, statement:str, text_so_far:str, question_context:str, **kwargs):

        messages = deepcopy(self.first_statement_instruction) if text_so_far is None else deepcopy(self.instruction)
        messages[-1]["content"] = messages[-1]["content"].format(statement=statement, text_so_far=text_so_far, question_context=question_context)

        if type(self.model) == str:
            response = completion(
                model=self.model,
                messages=messages,
                # **kwargs
            )
            question = response.choices[0].message['content']
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize = False)
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
            model_output = self.model.generate(input_ids)
            tokens = model_output[0][len(input_ids[0]):]
            question = self.tokenizer.decode(tokens, skip_special_tokens = False)

        return question.strip()
    
    def does_entail(self, statement:str, question:str, answer:str)->bool:
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
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        #Generate questions
        questions = []
        question_check = []
        for _ in range(self.num_questions):
            question = self.generate_question(statement, text_so_far, question_context)
            if question.lower() not in question_check:
                question_check.append(question.lower())
                questions.append(question)
        del question_check
        print("Questions generated:", questions)

        #Get model answers for each question (generate answers until it entails the statement)
        answers = [None] * len(questions)
        texts = [None] * len(questions)
        model_outputs = [None] * len(questions)
        messages = deepcopy(self.generate_answer_instruction)
        for i, question in enumerate(questions):
            messages[1]["content"] = question
            text = tokenizer.apply_chat_template(messages, tokenize = False)
            input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

            #check if the answer aligns with the statement
            for _ in range(self.max_answer_trials):
                model_output = model.generate(input_ids)  
                tokens = model_output[0][len(input_ids[0]):]
                answer = tokenizer.decode(tokens, skip_special_tokens = True)
                if self.does_entail(statement, question, answer):
                    answers[i] = answer
                    texts[i] = text
                    model_outputs[i] = model_output
                    break
                print("     ", answer)

        print("Answers generated:", answers)
        print()

        #Get UE for each question
        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, text, answer, model_output in zip(questions, texts, answers, model_outputs):
            if answer is not None:
                truth_values = self.truth_method.generate_forward(model, text, answer, question, all_ids=model_output, tokenizer=tokenizer, **self.kwargs)
                normalized_truth_values.append(truth_values['normalized_truth_value'])
                unnormalized_truth_values.append(truth_values['truth_value'])
                method_spec_outputs.append(truth_values)
            else: 
                normalized_truth_values.append(0)
                unnormalized_truth_values.append(-torch.inf)
                method_spec_outputs.append(None)

        #Create single score for the statement
        #TODO: Implement a better way to combine the scores
        normalized_truth_value = sum(normalized_truth_values) / len(normalized_truth_values)
        unnormalized_truth_value = sum(unnormalized_truth_values) / len(unnormalized_truth_values)

        return {"normalized_truth_value": normalized_truth_value, "truth_value": unnormalized_truth_value,
                       "normalized_truth_values": normalized_truth_values, "truth_values": unnormalized_truth_values,
                       "questions": questions, "answers": answers, "truth_method_spec_outputs": method_spec_outputs}
        

    def check_statement_api(self, model:str, messages:list, generated_text:str, question_context:str, statement:str, text_so_far:str, **kwargs):
        
        #Generate questions
        questions = []
        question_check = []
        for _ in range(self.num_questions):
            question = self.generate_question(statement, text_so_far, question_context)
            if question.lower() not in question_check:
                question_check.append(question.lower())
                questions.append(question)
        del question_check
        print("Questions generated:", questions)

        #Get model answers for each question (generate answers until it entails the statement)
        answers = [None] * len(questions)
        q_messages = deepcopy(self.generate_answer_instruction)
        for i, question in enumerate(questions):
            q_messages[1]["content"] = question

            #check if the answer aligns with the statement
            for _ in range(self.max_answer_trials):
                response = completion(
                    model=model,
                    messages=q_messages,
                    seed = randint(0, 1000000),
                    # **kwargs
                )
                answer = response.choices[0].message['content']
                if self.does_entail(statement, question, answer):
                    answers[i] = answer
                    break
                print("     ", answer)

        print("Answers generated:", answers)
        print()

        #Get UE for each question
        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, answer in zip(questions, answers):
            q_messages[1]["content"] = question
            if answer is not None:
                truth_values = self.truth_method.completion_forward(model, q_messages, answer, question, **kwargs)
                normalized_truth_values.append(truth_values['normalized_truth_value'])
                unnormalized_truth_values.append(truth_values['truth_value'])
                method_spec_outputs.append(truth_values)
            else:
                normalized_truth_values.append(0)
                unnormalized_truth_values.append(-torch.inf)
                method_spec_outputs.append(None)

        #Create single score for the statement
        #TODO: Implement a better way to combine the scores
        normalized_truth_value = sum(normalized_truth_values) / len(normalized_truth_values)
        unnormalized_truth_value = sum(unnormalized_truth_values) / len(unnormalized_truth_values)

        return {"normalized_truth_value": normalized_truth_value, "truth_value": unnormalized_truth_value,
                       "normalized_truth_values": normalized_truth_values, "truth_values": unnormalized_truth_values,
                       "questions": questions, "answers": answers, "truth_method_spec_outputs": method_spec_outputs}
        

