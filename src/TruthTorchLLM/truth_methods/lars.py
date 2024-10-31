from .truth_method import TruthMethod
from TruthTorchLLM.utils import sigmoid_normalization, bidirectional_entailment_clustering
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from .truth_method import TruthMethod
from .semantic_entropy import calculate_total_log
# from ..evaluators.correctness_evaluator import CorrectnessEvaluator 
# from ..evaluators.rouge import ROUGE
# from TruthTorchLLM.utils.dataset_utils import get_dataset

from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
import torch
import numpy as np
import copy
import random

from ..generation import sample_generations_hf_local, sample_generations_api

class LARS(TruthMethod):
    def __init__(self, threshold:float=0.0, std:float = 1.0, 
                        device="cuda", lars_model:PreTrainedModel=None, lars_tokenizer:PreTrainedTokenizer=None, 
                        ue_type:str="confidence", number_of_generations:int=5,
                        model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda'):
        super().__init__(threshold = threshold, std = std)

        assert ue_type in ["confidence", "semantic_entropy", "se", "entropy"], f"ue_type must be one of ['confidence', 'semantic_entropy', 'se', 'entropy'] but it is {ue_type}."
        self.ue_type = ue_type
        self.number_of_generations = number_of_generations #number of generations for semantic entropy and entropy
        
        #lars model
        if lars_model is None or lars_tokenizer is None:
            lars_model = AutoModelForSequenceClassification.from_pretrained("duygunuryldz/deneme").to(device) #TODO
            lars_tokenizer = AutoTokenizer.from_pretrained("duygunuryldz/deneme") #TODO 
        self.lars_model = lars_model
        self.lars_tokenizer = lars_tokenizer
        self.device = device

        #lars params
        self.number_of_bins = lars_model.config.number_of_bins #number of bins for discretization of the probability space 
        self.edges = lars_model.config.edges #edges of bins, discretization of the probability space 

        #params for semantic entropy 
        if (ue_type == "se" or ue_type == "semantic_entropy") and (model_for_entailment is None or tokenizer_for_entailment is None):
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')
        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment


    def _find_bin(self, value):
        if self.edges is not None:
            bin_index = np.digitize(value, self.edges, right=False)
        else:
            bin_index = int(value*self.number_of_bins) #discretize the probability space equally
        return min(bin_index, (self.number_of_bins-1))

    def _lars(self, question, generation_token_texts, probs):

        a_text = ""
        for i, tkn_text in enumerate(generation_token_texts):
            bin_id = self._find_bin(probs[i])
            a_text += tkn_text+f"[prob_token_{bin_id}]"
        
        tokenized_input = self.lars_tokenizer(
            question, a_text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            is_split_into_words=False,  
            truncation=True,
            max_length = 512,           # Pad & truncate all sentences.
            pad_to_max_length = True,
        )

        input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(self.device)
        attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(self.device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(self.device)

        self.lars_model.eval()
        logits = self.lars_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits.detach()
            
        return torch.nn.functional.sigmoid(logits[:,  0]).item()


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        if self.ue_type == "confidence":
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
            model_output = all_ids
            tokens = model_output[0][len(input_ids[0]):]
            tokens_text = [tokenizer.decode(token) for token in tokens]

            # tokens_text = tokenizer.convert_ids_to_tokens(tokens)
            # tokens_text = [s.replace(space_char, " ") for s in tokens_text]  #requires space_char for each model

            with torch.no_grad():
                outputs = model(model_output)
                logits = outputs.logits  # Logits for each token in the input

                # Calculate probabilities from logits
                probs = torch.nn.functional.softmax(logits, dim=-1)#probs for each token
                probs = probs[0, len(input_ids[0])-1:-1, :]#probs for each token in the generated text
                probs = torch.gather(probs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#probs for each token in the generated text
                probs = probs.view(-1).tolist()#convert to list

                lars_score = self._lars(question_context, tokens_text, probs)

        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            if sampled_generations_dict is None:
                sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
                                                                        number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, batch_generation=self.batch_generation, **kwargs)
            scores = []
            generated_outputs = []
            generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        
            for i in range(self.number_of_generations):
                tokens_text = [tokenizer.decode(token) for token in sampled_generations_dict["tokens"][i]]
                score = torch.log(self._lars(question_context, tokens_text, torch.exp(sampled_generations_dict["logprobs"][i])) )
                scores.append(score) #scores are in log scale
                generated_outputs.append((generated_texts[i],score))

            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, sampled_generations_dict["generated_texts"])
                lars_score = -calculate_total_log(generated_outputs,clusters)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}
            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts}

        return {"truth_value": lars_score,  "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):

        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("LARS method is not applicable to given model")

        if self.ue_type == "confidence":
            kwargs = copy.deepcopy(kwargs)
            response = completion(
                model=model,
                messages=messages,
                logprobs = True,
                **kwargs
                )
                
            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            generated_text = response.choices[0].message['content']
            lars_score = self._lars(question_context, tokens, torch.exp(logprobs))

        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            if sampled_generations_dict is None:
                sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
                                                                    number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, **kwargs)
            scores = []
            generated_outputs = []
            generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        
            for i in range(self.number_of_generations):
                score = torch.log(self._lars(question_context, sampled_generations_dict["tokens"][i], torch.exp(sampled_generations_dict["logprobs"][i])) )
                scores.append(score) #scores are in log scale
                generated_outputs.append((generated_texts[i],score))

            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, sampled_generations_dict["generated_texts"])
                lars_score = -calculate_total_log(generated_outputs,clusters)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}
            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts}

        return {"truth_value": lars_score, "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format


    # def prepare_data_for_training(self, datasets: list[Union[str, list]], model:Union[str,PreTrainedModel], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
    #                               size_of_data = 1.0, val_ratio=0.1, num_gen_per_question:int=5, batch_generation = True,
    #                               correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), seed:int = 0):
        
    #     '''
    #         dataset: list of datasets, each dataset can be a string correspoing to dataset name that the library offers, or it can be a list of data samples 
    #     '''
        
    #     dataset = get_dataset(dataset, size_of_data=size_of_data, seed=seed, split = "train")
    #     pass 

    def train_lars_model(self, train_data, test_data, **kwargs):

        #preprocess data
            #params: datasets, dataset fractions, val_ratio, num answer per question, qa evaluator, model, tokenizer, generation params
            #return: question, generation token text list, probs, labels for both train and val
        #train model
            #params: dataset, val set, model, tokenizer, num of bins, train params (epoch, lr, batch size, test iter)
        #save model
        pass #not implemented yet

    def __str__(self):
        if self.ue_type == "confidence":
            return "LARS Truth Method with type: " + self.ue_type + " LARS model: " + self.lars_model.config._name_or_path + " number of bins: " + str(self.number_of_bins) + " probability range borders: " + str(self.edges)
        elif self.ue_type == "entropy":
            return "LARS Truth Method with type: " + self.ue_type + " number of generations: " + str(self.number_of_generations) + " LARS model: " + self.lars_model.config._name_or_path + " number of bins: " + str(self.number_of_bins) + " probability range borders: " + str(self.edges)
        elif self.ue_type in ["semantic_entropy", "se"]:
            return "LARS Truth Method with type: " + self.ue_type + " number of generations: " + str(self.number_of_generations) + " LARS model: " + self.lars_model.config._name_or_path + " number of bins: " + str(self.number_of_bins) + " probability range borders: " + str(self.edges) + " entailment model: " + self.model_for_entailment.config._name_or_path
