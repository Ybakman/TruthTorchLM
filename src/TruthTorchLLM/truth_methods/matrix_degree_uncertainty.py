import copy
import torch
import random
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.utils import calculate_U_deg, sigmoid_normalization
from .truth_method import TruthMethod



class MatrixDegreeUncertainty(TruthMethod):
    def __init__(self, method_for_similarity: str = "semantic", number_of_generations=5, threshold=0.0, std=1.0, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None, temperature = 3.0):
        
        super().__init__(threshold = threshold, std = std)

        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')
       
        if method_for_similarity not in ["semantic", "jaccard"]:
            raise ValueError("method_for_similarity should be either semantic or jaccard. Please refer to https://arxiv.org/pdf/2305.19187 for more information.")
        
        if method_for_similarity == "semantic":
            print('There are 2 methods for similarity: semantic similarity and jaccard score. The default method is semantic similarity. If you want to use jaccard score, please set method_for_similarity="jaccard". Please refer to https://arxiv.org/pdf/2305.19187 for more information.')
            self.tokenizer_for_entailment = tokenizer_for_entailment
            self.model_for_entailment = model_for_entailment

        self.number_of_generations = number_of_generations
        self.method_for_similarity = method_for_similarity #jaccard or semantic
        self.temperature = temperature #temperature for NLI model

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        for i in range(self.number_of_generations):
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

        output_dict = {}
       
        output = calculate_U_deg(generated_texts, question_context, temperature = self.temperature)
        output_dict['U_deg'] = output
        output_dict['generated_texts'] = generated_texts
        output_dict['truth_value'] = -output
        output_dict['normalized_truth_value'] = sigmoid_normalization(output, self.threshold, self.std)
        return output_dict

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        generated_outputs = {}
        scores = []
        for i in range(self.number_of_generations):
            kwargs.pop('logprobs', None)
            seed = kwargs.pop('seed', None) 
            seed = random.randint(0, 1000000)
            kwargs['seed'] = seed
            
            response = completion(
                model=model,
                messages=messages,
                logprobs=True,
                **kwargs
            )
            generated_texts.append(response.choices[0].message['content'])

        output_dict = {}
        
        output = calculate_U_deg(generated_texts, question_context)
        output_dict['U_deg'] = output
        output_dict['generated_texts'] = generated_texts
        output_dict['truth_value'] = -output
        output_dict['normalized_truth_value'] = sigmoid_normalization(output, self.threshold, self.std)
        return output_dict


    def __str__(self):
        return f'Matrix Degree Uncertainty with {self.method_for_similarity} similarity method and {self.output_method} output method. Model for entailment: {self.model_for_entailment}, Tokenizer for entailment: {self.tokenizer_for_entailment}, Number of generations: {self.number_of_generations}, Threshold: {self.threshold}, Standard Deviation: {self.std}, Temperature: {self.temperature}.'
    