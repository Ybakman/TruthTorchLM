import copy
import torch
import random
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
from TruthTorchLLM.utils import *
from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from transformers import AutoModelForCausalLM, AutoTokenizer


class SelfDetection(TruthMethod):
    def __init__(self, model_for_questions, method_for_similarity: str = "semantic", number_of_generations=5, threshold=0.5, std=1.0, model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None):
        super().__init__(threshold = threshold, std = std)

        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')


        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment
        self.number_of_generations = number_of_generations
        self.method_for_similarity=method_for_similarity #jaccard or semantic
        self.model_for_questions = model_for_questions #string or model

    def generate_similar_questions(self, input_text: str):
        generated_questions = []
        new_input_text = f"Given the following question {input_text}, paraphrase it to have different words and expressions but is semantically equivalent"
        for i in range(self.number_of_generations):
            if type(self.model_for_questions) != str:
                input_ids = self.tokenizer_for_entailment.encode(new_input_text, return_tensors="pt").to(self.model_for_questions.device)
                model_output = self.model_for_questions.generate(input_ids, num_return_sequences=1, do_sample=True)
                tokens = model_output[0][len(input_ids[0]):]
                generated_question = self.tokenizer_for_entailment.decode(tokens, skip_special_tokens=True)
            else:
                if self.model_for_questions not in PROB_AVAILABLE_API_MODELS:
                    raise ValueError("This method is not applicable to given model")
                response = completion(
                    model=self.model_for_questions,
                    messages=[{"content": new_input_text, "role": "user"}],
                    logprobs=True
                )
                generated_question = response.choices[0].message['content']
            generated_questions.append(generated_question)
        return generated_questions

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        #input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)

        generated_questions=self.generate_similar_questions(input_text)

        for qenerated_question in generated_questions:
            input_ids = tokenizer.encode(qenerated_question, return_tensors="pt").to(model.device)
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts, self.method_for_similarity)
        entropy = 0
        for cluster in clusters:
            entropy += len(cluster)/self.number_of_generations * np.log(len(cluster)/self.number_of_generations)
        normalized_entropy_value = sigmoid_normalization(entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_entropy_value': normalized_entropy_value, 'entropy': entropy, "generated_questions": generated_questions, 'generated_texts': generated_texts, "clusters": clusters}

    
    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("This method is not applicable to given model")
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        generated_questions=self.generate_similar_questions(messages)
        for generated_question in generated_questions:
            response = completion(
                model=model,
                messages=[{"content": generated_question, "role": "user"}],
                logprobs=True,
                **kwargs
            )
            generated_texts.append(response.choices[0].message['content'])

        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts, self.method_for_similarity)
        entropy = 0
        for cluster in clusters:
            entropy += len(cluster)/self.number_of_generations * np.log(len(cluster)/self.number_of_generations)

        normalized_entropy_value = sigmoid_normalization(entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_entropy_value': normalized_entropy_value, 'entropy': entropy, "generated_questions": generated_questions, 'generated_texts': generated_texts, "clusters": clusters}
        
    
    def __str__(self):
        return "SelfDetection Class with " + str(self.number_of_generations) + " generations. Method for similarity: " + str(self.method_for_similarity) + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std) + ". Model for entailment: " + str(self.model_for_entailment) + ". Tokenizer for entailment: " + str(self.tokenizer_for_entailment) + ". Model for questions: " + (self.model_for_questions if isinstance(self.model_for_questions, str) else str(self.model_for_questions))

