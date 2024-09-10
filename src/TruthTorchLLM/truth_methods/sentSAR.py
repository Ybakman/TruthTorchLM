import copy
import torch
import numpy as np
import random
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.utils import sigmoid_normalization, bidirectional_entailment_clustering
from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from transformers import AutoModelForCausalLM, AutoTokenizer

from sentence_transformers.cross_encoder import CrossEncoder


class SentSAR(TruthMethod):
    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations=5, t=0.001, threshold=0.0, std=1.0, 
                 model_for_similarity: str = "cross-encoder/stsb-roberta-large"):#normalization
        super().__init__(threshold = threshold, std = std)


        self.model_for_similarity =  CrossEncoder(model_name=model_for_similarity, num_labels=1)
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.t = t



    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        scores = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)

        for i in range(self.number_of_generations):
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            with torch.no_grad():
                outputs = model(model_output)
                logits = outputs.logits # Logits for each token in the input

            logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
            logprobs = logprobs[0, len(input_ids[0])-1:-1, :]#logprobs for each token in the generated text
            logprobs = torch.gather(logprobs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
            logprobs = logprobs.view(-1).tolist()#convert to list

            score = self.scoring_function(question_context, tokens, logprobs) 
            scores.append(score) #scores are in log scale
            generated_texts.append(generated_text)

        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []

        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                gen_i = question_context + generated_texts[i]
                gen_j = question_context + generated_texts[j]
                similarity_i_j = self.model_for_similarity .predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        probs = torch.exp(torch.tensor(scores))
        assert len(probs) == len(similarities)

        sentence_scores = []
        for idx, prob in enumerate(probs):
            w_ent = -torch.log(
                prob + ((torch.tensor(similarities[idx]) / self.t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            sentence_scores.append(w_ent)
        sentence_scores = torch.tensor(sentence_scores)

        entropy = (torch.sum(sentence_scores, dim=0) / torch.tensor(sentence_scores.shape[0])).item()
        
        normalized_truth_value = sigmoid_normalization(entropy, self.threshold, self.std)
        return {"truth_value": -entropy, 'normalized_truth_value': normalized_truth_value, 'sentSAR': entropy, "score_for_each_generation": scores, 'generated_texts': generated_texts, "similarities": similarities}

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)

        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Semantic Entropy method is not applicable to given model")
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

            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            generated_texts.append(response.choices[0].message['content'])

            score = self.scoring_function(question_context, tokens, logprobs)
            scores.append(score)#scores are in log scale
            generated_outputs[response.choices[0].message['content']] = score

        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []

        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                gen_i = question_context + generated_texts[i]
                gen_j = question_context + generated_texts[j]
                similarity_i_j = self.model_for_similarity .predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        probs = torch.exp(torch.tensor(scores))
        assert len(probs) == len(similarities)

        sentence_scores = []
        for idx, prob in enumerate(probs):
            w_ent = -torch.log(
                prob + ((torch.tensor(similarities[idx]) / self.t) * torch.cat([probs[:idx], probs[idx + 1:]])).sum())
            sentence_scores.append(w_ent)
        sentence_scores = torch.tensor(sentence_scores)

        entropy = (torch.sum(sentence_scores, dim=0) / torch.tensor(sentence_scores.shape[0])).item()
        normalized_truth_value = sigmoid_normalization(entropy, self.threshold, self.std)

        return {"truth_value": -entropy, 'normalized_truth_value': normalized_truth_value, 'sentSAR': entropy, "score_for_each_generation": scores, 'generated_texts': generated_texts, "similarities": similarities}

    def __str__(self):
        return "SentSAR Truth Method with " + str(self.number_of_generations) + " generations and t="+ str(self.t)+". Model for checking sentence similarity: " + self.model_for_similarity.config._name_or_path + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    

    

