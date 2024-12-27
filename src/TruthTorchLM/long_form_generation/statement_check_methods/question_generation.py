from .statement_check_method import StatementCheckMethod
from TruthTorchLM.utils import check_entailment
from TruthTorchLM.truth_methods import TruthMethod
from TruthTorchLM.availability import AVAILABLE_API_MODELS
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from TruthTorchLM.generation import get_sampling_properties, sample_generations_hf_local, sample_generations_api

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
    def __init__(self, model:Union[PreTrainedModel, str], num_questions:int, 
                 instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 first_statement_instruction:list=FIRST_STATEMENT_INSTRUCTION, 
                 generate_answer_instruction:list=GEN_ANSWER_INST,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
                 truth_methods:list[TruthMethod]=None, batch_generation:bool = True, 
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
        self.truth_methods = truth_methods
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message
        self.batch_generation = batch_generation
        self.kwargs = {
                # "max_length": 50,
                "num_return_sequences": 1,
                "seed": 42,
                "do_sample":True}
        self.kwargs.update(kwargs)

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

    def _generate_question(self, statement:str, text_so_far:str, question_context:str):

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
        return questions
    
    def _get_truth_value_local(self, truth_methods, model, tokenizer, question, text, answer, model_output, generation_seed, **kwargs):

        number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)

        sampled_gen_dict = sample_generations_hf_local(model, text, tokenizer, generation_seed, number_of_generations=number_of_generations, 
        return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs,return_attentions=return_attentions, return_activations=return_activations, batch_generation=self.batch_generation,  **kwargs)

        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for truth_method in truth_methods:
            truth_values = truth_method(model=model, input_text=text, generated_text=answer, question_context=question, all_ids=model_output, tokenizer=tokenizer, generation_seed = generation_seed, sampled_generations_dict=sampled_gen_dict, **kwargs)     
            normalized_truth_values.append(truth_values['normalized_truth_value'])
            unnormalized_truth_values.append(truth_values['truth_value'])
            method_spec_outputs.append(truth_values)

        return normalized_truth_values, unnormalized_truth_values, method_spec_outputs

    def check_statement_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, 
                              statement:str, text_so_far:str, all_ids:Union[list, torch.Tensor], 
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed=None, 
                              messages:list = [], add_generation_prompt = True, continue_final_message = False, **kwargs):

        questions = self._get_questions(question_context=question_context, statement=statement, text_so_far=text_so_far)
        answers = [statement] * len(questions)
        texts = [None] * len(questions)
        model_outputs = [None] * len(questions)
        for i, question in enumerate(questions):
            messages = deepcopy(self.generate_answer_instruction)
            messages[1]["content"] = question
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
            texts[i] = text
            messages.append({"role": "assistant", "content": statement})
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text_messsages = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
            model_outputs[i] = tokenizer.encode(text_messsages, return_tensors="pt").to(model.device)

        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, text, answer, model_output in zip(questions, texts, answers, model_outputs):
            if answer is not None:
                normalized_truth_value, unnormalized_truth_value, method_spec_output = self._get_truth_value_local(self.truth_methods, model=model, tokenizer=tokenizer, 
                                                                                                                        question=question, text=text, answer=answer, 
                                                                                                                        model_output=model_output, generation_seed=generation_seed, **kwargs)
                normalized_truth_values.append(normalized_truth_value)
                unnormalized_truth_values.append(unnormalized_truth_value)
                method_spec_outputs.append(method_spec_output)
            else: 
                normalized_truth_values.append([0]*len(self.truth_methods))
                # unnormalized_truth_values.append(-torch.inf)
                unnormalized_truth_values.append([0]*len(self.truth_methods))
                method_spec_outputs.append([{}]*len(self.truth_methods))
        
        final_normalized_truth_values = []
        final_unnormalized_truth_values = []
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {"normalized_truth_values":[], 'unnormalized_truth_values':[], "method_spec_outputs":[]}
            total = 0
            for truth_values in normalized_truth_values:
                total += truth_values[i]
                output_dict["normalized_truth_values"].append(truth_values[i])
            final_normalized_truth_values.append(total/len(normalized_truth_values))
            total = 0
            for truth_values in unnormalized_truth_values:
                total += truth_values[i]
                output_dict["unnormalized_truth_values"].append(truth_values[i])
            final_unnormalized_truth_values.append(total/len(unnormalized_truth_values))
            for truth_values in method_spec_outputs:
                output_dict["method_spec_outputs"].append(truth_values[i])
            final_method_specific_outputs.append(output_dict)

        return { "normalized_truth_values": final_normalized_truth_values, "truth_values": final_unnormalized_truth_values,
                       "questions": questions, "answers": answers, "truth_method_spec_outputs": final_method_specific_outputs}
        


    def _get_truth_value_api(self, truth_methods, model, q_messages, question, answer, generation_seed, **kwargs):

        #Get sampled generations to be used in truth methods
        number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)
        sampled_gen_dict = sample_generations_api(model, q_messages, generation_seed, number_of_generations=number_of_generations, 
        return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs,return_attentions=return_attentions, return_activations=return_activations, **kwargs)
      
        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for truth_method in truth_methods:
            truth_values = truth_method(model=model, messages=q_messages, generated_text=answer, question_context=question, generation_seed=generation_seed, sampled_generations_dict=sampled_gen_dict, **kwargs)
            normalized_truth_values.append(truth_values['normalized_truth_value'])
            unnormalized_truth_values.append(truth_values['truth_value'])
            method_spec_outputs.append(truth_values)

        return normalized_truth_values, unnormalized_truth_values, method_spec_outputs

    def check_statement_api(self, model:str, messages:list, generated_text:str, 
                            question_context:str, statement:str, text_so_far:str, generation_seed=None, **kwargs):
        

        questions = self._get_questions(question_context=question_context, statement=statement, text_so_far=text_so_far)

        requires_logprobs = False    
        for truth_method in self.truth_methods:
            if truth_method.REQUIRES_LOGPROBS:
                requires_logprobs = True

        if requires_logprobs:
            raise ValueError(f"Truth methods requiring logprobs cannot be used with QuestionGeneration statement check method.")

        #Get model answers for each question (generate answers until it entails the statement)
        answers = [statement] * len(questions)
        q_messages = deepcopy(self.generate_answer_instruction)

        #Get truth value for truth method
        normalized_truth_values = []
        unnormalized_truth_values = []
        method_spec_outputs = []
        for question, answer in zip(questions, answers):
            q_messages[1]["content"] = question
            if answer is not None:
                normalized_truth_value, unnormalized_truth_value, method_spec_output = self._get_truth_value_api(self.truth_methods, model=model, 
                                                                                             q_messages=q_messages, question=question, answer=answer, 
                                                                                             generation_seed=generation_seed, **kwargs)
                normalized_truth_values.append(normalized_truth_value)
                unnormalized_truth_values.append(unnormalized_truth_value)
                method_spec_outputs.append(method_spec_output)
            else: 
                normalized_truth_values.append([0]*len(self.truth_methods))
                # unnormalized_truth_values.append(-torch.inf)
                unnormalized_truth_values.append([0]*len(self.truth_methods))
                method_spec_outputs.append([{}]*len(self.truth_methods))
       
        final_normalized_truth_values = []
        final_unnormalized_truth_values = []
        final_method_specific_outputs = []
        for i in range(len(self.truth_methods)):
            output_dict = {"normalized_truth_values":[], 'unnormalized_truth_values':[], "method_spec_outputs":[]}
            total = 0
            for truth_values in normalized_truth_values:
                total += truth_values[i]
                output_dict["normalized_truth_values"].append(truth_values[i])
            final_normalized_truth_values.append(total/len(normalized_truth_values))
            total = 0
            for truth_values in unnormalized_truth_values:
                total += truth_values[i]
                output_dict["unnormalized_truth_values"].append(truth_values[i])
            final_unnormalized_truth_values.append(total/len(unnormalized_truth_values))
            for truth_values in method_spec_outputs:
                output_dict["method_spec_outputs"].append(truth_values[i])
            final_method_specific_outputs.append(output_dict)

        return { "normalized_truth_values": final_normalized_truth_values, "truth_values": final_unnormalized_truth_values,
                       "questions": questions, "answers": answers, "truth_method_spec_outputs": final_method_specific_outputs}
        


    def __str__(self): 

        model_name = self.model.__class__ if type(self.model) != str else self.model

        return f"Statement Check Method by Generating Questions.\n\
Question generation model: {model_name}\n\
Number of questions to be generated for a stament: {self.num_questions}\n\
Question generation instruction for the first statement:\n  {self.first_statement_instruction}\n\n\
Question generation instruction for statements with preceeding text:\n  {self.instruction}\n\n\
Answer generation instruction:\n    {self.generate_answer_instruction}\n\n\
Truth methods to assign a score the question(s):\n   {self.truth_methods}"