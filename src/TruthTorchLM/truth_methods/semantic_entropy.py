
import torch
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLM.utils import bidirectional_entailment_clustering
from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
from .truth_method import TruthMethod
from TruthTorchLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from ..generation import sample_generations_hf_local, sample_generations_api



def calculate_total_log(generated_outputs : list[str, float],clusters : list[set[str]]):
    total_output_for_log = 0
    for i, cluster in enumerate(clusters):
        score_list = []
        for elem in cluster:
            for output in generated_outputs:
                if elem == output[0]:
                    score_list.append(output[1])
        total_output_for_log -= torch.logsumexp(torch.tensor(score_list), dim=0).item()
    return total_output_for_log / len(clusters)


class SemanticEntropy(TruthMethod):
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True

    def __init__(self, scoring_function : ScoringMethod = LengthNormalizedScoring(), number_of_generations=5, 
                 model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda', batch_generation = True):#normalization
        super().__init__()

        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.scoring_function = scoring_function
        self.number_of_generations = number_of_generations
        self.batch_generation = batch_generation

    def _semantic_entropy(self, generated_texts:list[str], question_context:str, scores:list[float], generated_outputs:list):

        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, generated_texts)
        total_output_for_log = calculate_total_log(generated_outputs,clusters)

        return {"truth_value": -total_output_for_log, 'semantic_entropy': total_output_for_log, "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, batch_generation = self.batch_generation, **kwargs)

        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        generated_outputs = []
        scores = []

        for i in range(self.number_of_generations):
            text = generated_texts[i]
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale
            generated_outputs.append((text,score))

        return self._semantic_entropy(generated_texts, question_context, scores, generated_outputs)
        

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("Semantic Entropy method is not applicable to given model")
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        generated_outputs = []
        scores = []

        for i in range(self.number_of_generations):
            text = generated_texts[i]
            score = self.scoring_function(sampled_generations_dict["logprobs"][i]) 
            scores.append(score) #scores are in log scale
            generated_outputs.append((text,score))

        return self._semantic_entropy(generated_texts, question_context, scores, generated_outputs)

