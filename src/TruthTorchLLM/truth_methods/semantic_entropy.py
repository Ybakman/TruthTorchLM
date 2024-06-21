import copy
import torch
import numpy as np
import random
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLLM.utils import *
from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
from .truth_method import TruthMethod
from TruthTorchLLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculating_total_log(generated_outputs : dict[str, float],clusters : list[set[str]]):
    total_output_for_log = 0
    for i, cluster in enumerate(clusters):
        total_output_for_log -= sum(generated_outputs[elem] for elem in cluster)
    return total_output_for_log / len(clusters)


class SemanticEntropy(TruthMethod):
    def __init__(self, model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations=10, threshold=0.5, std=1.0):#normalization
        super().__init__()
        self.model_for_entailment=model_for_entailment
        self.tokenizer_for_entailment=tokenizer_for_entailment
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.threshold = threshold
        self.std = std
        if model_for_entailment is None:
            self.model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
        if tokenizer_for_entailment is None:
            self.tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')


    def generate_forward(self, model: PreTrainedModel, input_text: str, all_ids: Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        generated_outputs = {}
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

            score = self.scoring_function(input_text, tokens, logprobs) 
            scores.append(score) #scores are in log scale
            generated_texts.append(generated_text)
            generated_outputs[generated_text] = score
        
        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, input_text, list(generated_outputs.keys()))   
        total_output_for_log = calculating_total_log(generated_outputs,clusters)
        normalized_truth_value = sigmoid_normalization(total_output_for_log, self.threshold, self.std)
        return {"truth_value": total_output_for_log, 'normalized_truth_value': normalized_truth_value, 'semantic_entropy': -total_output_for_log, "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}

    def completion_forward(self, model: str, input_text: str, generated_text: str, **kwargs):
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
                messages=[{"content": input_text, "role": "user"}],
                logprobs=True,
                **kwargs
            )

            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            generated_texts.append(response.choices[0].message['content'])

            score = self.scoring_function(input_text, tokens, logprobs)
            scores.append(score)#scores are in log scale
            generated_outputs[response.choices[0].message['content']] = score

        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, input_text, list(generated_outputs.keys()))
        total_output_for_log = calculating_total_log(generated_outputs,clusters) 
        normalized_truth_value = sigmoid_normalization(total_output_for_log, self.threshold, self.std)
        return {"truth_value": total_output_for_log, 'normalized_truth_value': normalized_truth_value, 'semantic_entropy': -total_output_for_log, "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}

    def __str__(self):
        return "Semantic Entropy Truth Method with " + str(self.number_of_generations) + " generations. Model for checking semantic: " + str(self.model_for_entailment) + ". Tokenizer for checking semantic: " + str(self.tokenizer_for_entailment) + ". Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    

    

