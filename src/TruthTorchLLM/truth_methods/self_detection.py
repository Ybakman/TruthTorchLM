import copy
import torch
import random
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.availability import AVAILABLE_API_MODELS
from TruthTorchLLM.utils import *
from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from transformers import AutoModelForCausalLM, AutoTokenizer


class SelfDetection(TruthMethod):
    def __init__(self, method_for_similarity: str = "semantic", number_of_generations=5, threshold=0.5, std=1.0, model_for_entailment: PreTrainedModel = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli'), tokenizer_for_entailment: PreTrainedTokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli'), model_for_questions = None, tokenizer_for_questions = None):
        super().__init__(threshold = threshold, std = std)
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment
        self.number_of_generations = number_of_generations
        self.model_for_questions = model_for_questions #string or model
        self.tokenizer_for_questions = tokenizer_for_questions

    def generate_similar_questions(self, input_text: str, prompt_for_generating_question: str = None, model= None, tokenizer = None):  
        generated_questions = [input_text]
        if prompt_for_generating_question is not None:
            new_input_text = prompt_for_generating_question + input_text
        else:
            new_input_text = f"Given a question, paraphrase it to have different words and expressions but have the same meaning as the original question. Please note that you should not answer the question, but rather provide a re-phrased." + input_text
        for i in range(self.number_of_generations-1):
            if self.model_for_questions is None:
                if model is None:
                    raise ValueError("model and tokenizer should be provided")
                elif type(model) != str:
                    input_ids = tokenizer.encode(new_input_text, return_tensors="pt").to(model.device)
                    model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, temperature=1.0)
                    tokens = model_output[0][len(input_ids[0]):]
                    generated_question = tokenizer.decode(tokens, skip_special_tokens=True)
                elif type(model) == str:
                    if model not in AVAILABLE_API_MODELS:
                        raise ValueError("This method is not applicable to given model")
                    response = completion(
                        model=model,
                        messages=[{"content": new_input_text, "role": "user"}],
                        temperature=1.0
                    )
                    generated_question = response.choices[0].message['content']
            
            elif type(self.model_for_questions) != str:
                input_ids = self.tokenizer_for_questions.encode(new_input_text, return_tensors="pt").to(self.model_for_questions.device)
                model_output = self.model_for_questions.generate(input_ids, num_return_sequences=1, do_sample=True, temperature=1.0)
                tokens = model_output[0][len(input_ids[0]):]
                generated_question = self.tokenizer_for_questions.decode(tokens, skip_special_tokens=True)
            else:
                if self.model_for_questions not in AVAILABLE_API_MODELS:
                    raise ValueError("This method is not applicable to given model")
                response = completion(
                    model=self.model_for_questions,
                    messages=[{"content": new_input_text, "role": "user"}],
                    temperature=1.0
                )
                generated_question = response.choices[0].message['content']
            generated_questions.append(generated_question)
        return generated_questions

    
    

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, entailment_method: str = "semantic", prompt_for_generating_question: str = None, prompt_for_entailment: str = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        #input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        generated_questions=self.generate_similar_questions(input_text, prompt_for_generating_question,model, tokenizer)

        for generated_question in generated_questions:
            input_ids = tokenizer.encode(generated_question, return_tensors="pt").to(model.device)
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        if entailment_method == "generation":
            clusters = bidirectional_entailment_clustering(model, tokenizer, question_context, generated_texts, "generation", prompt_for_entailment)
        elif entailment_method == "semantic" or entailment_method == "jaccard":
            clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts, entailment_method, prompt_for_entailment)
        else:
            raise ValueError("entailment_method should be either 'generation' or 'semantic' or 'jaccard'")
        entropy = 0
        for cluster in clusters:
            entropy += len(cluster)/self.number_of_generations * np.log(len(cluster)/self.number_of_generations)
        normalized_entropy_value = sigmoid_normalization(entropy, self.threshold, self.std)
        
        consistency = (len(clusters[0])-1)/(self.number_of_generations - 1)
        normalized_consistency = sigmoid_normalization(consistency, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_entropy_value': normalized_entropy_value, 'entropy': entropy, "consistency": consistency, "normalized_consistency_value": normalized_consistency, "generated_questions": generated_questions, 'generated_texts': generated_texts, "clusters": clusters}
        

    
    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, entailment_method: str = "semantic", prompt_for_generating_question: str = None, prompt_for_entailment: str = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)
        if model not in AVAILABLE_API_MODELS:
            raise ValueError("This method is not applicable to given model")
        kwargs = copy.deepcopy(kwargs)
        generated_questions = []
        generated_texts = []
        generated_questions=self.generate_similar_questions(messages[0], prompt_for_generating_question,model,None)
        for generated_question in generated_questions:
            response = completion(
                model=model,
                messages=[{"content": generated_question, "role": "user"}],
                **kwargs
            )
            generated_texts.append(response.choices[0].message['content'])
        if entailment_method == "completion":
            clusters = bidirectional_entailment_clustering(None, None, question_context, generated_texts, entailment_method, prompt_for_entailment, model)
        elif entailment_method == "semantic" or entailment_method == "jaccard":
            clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts, entailment_method, prompt_for_entailment) #entailment_method is semantic or jaccard
        else:
            raise ValueError("entailment_method should be either 'completion' or 'semantic' or 'jaccard'")
        entropy = 0
        for cluster in clusters:
            entropy += len(cluster)/self.number_of_generations * np.log(len(cluster)/self.number_of_generations)
        normalized_entropy_value = sigmoid_normalization(entropy, self.threshold, self.std)

        consistency = (len(clusters[0])-1)/(self.number_of_generations - 1)
        normalized_consistency = sigmoid_normalization(consistency, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_entropy_value': normalized_entropy_value, 'entropy': entropy, "consistency": consistency, "normalized_consistency_value": normalized_consistency, "generated_questions": generated_questions, 'generated_texts': generated_texts, "clusters": clusters}
        
    def __str__(self):
        return "SelfDetection Class with " + str(self.number_of_generations) + " generations"  ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std) + ". Model for entailment: " + str(self.model_for_entailment) + ". Tokenizer for entailment: " + str(self.tokenizer_for_entailment) + ". Model for questions: " + (self.model_for_questions if isinstance(self.model_for_questions, str) else str(self.model_for_questions)) + ". Tokenizer for questions: " + (self.tokenizer_for_questions if isinstance(self.tokenizer_for_questions, str) else str(self.tokenizer_for_questions))







