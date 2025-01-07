import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from datasets import  Dataset
from sklearn.model_selection import train_test_split


from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.availability import ACTIVATION_AVAILABLE_API_MODELS 

from .truth_method import TruthMethod
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT

from ..evaluators.correctness_evaluator import CorrectnessEvaluator 
from TruthTorchLM.utils.dataset_utils import get_dataset
from ..generation import sample_generations_batch_hf_local, sample_generations_sequential_hf_local
from TruthTorchLM.utils.eval_utils import metric_score

from sklearn.neural_network import MLPClassifier
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat
from sklearn.neural_network import MLPClassifier
from TruthTorchLM.utils.eval_utils import metric_score



#https://arxiv.org/pdf/2304.13734

class SAPLMA(TruthMethod):
    def __init__(self, saplma_model:MLPClassifier =None, layer_index = -1):

        super().__init__()
        self.saplma_model = saplma_model
        self.layer_index = layer_index
        

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        with torch.no_grad():
            outputs = model(all_ids, output_hidden_states=True)#get
            embedding = outputs['hidden_states'][self.layer_index][0, -2].cpu().float().numpy().reshape(1,-1)#look at the last position, -2 because 

        out = self.saplma_model.predict_proba(embedding)
        score = out[0][1]
  

        return {"truth_value": score}
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None,  logprobs:list=None, generated_tokens:list=None, **kwargs):

        if not model in ACTIVATION_AVAILABLE_API_MODELS:
            raise ValueError("SAPLMA method cannot be used with black-box API models since it requires access to activations.")

        return {"truth_value": 0}

        

    @staticmethod
    def _get_datasets(datasets:list, size_for_each_dataset:list, val_ratio:float, seed:int):
        print("Creating train and validation datasets...")
        all_data = []
        for i, dataset in enumerate(datasets):
            all_data.append(get_dataset(dataset, size_of_data=size_for_each_dataset[i], seed=seed, split = "train"))

        all_data = sum(all_data,[]) #list of dict, each dict contains "question" and "ground_truths"
        train_data, val_data = train_test_split(all_data, test_size=val_ratio, random_state=seed)
        return train_data, val_data
    
    @staticmethod
    def _generate_answers_and_label(train_data:list, val_data:list, model:PreTrainedModel, tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                               correctness_evaluator:CorrectnessEvaluator, previous_context:list=[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], 
                               user_prompt:str = DEFAULT_USER_PROMPT, num_gen_per_question:int=5, layer_index = -1, **kwargs):
        print("Generating answers and labels for training data...")
        print(kwargs)
        for i in tqdm(range(len(train_data))):
            messages = previous_context.copy()
            messages.append({'role': 'user', 'content': user_prompt.format(question_context = train_data[i]['question'])})

            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, continue_final_message=False)
            
            sampled = sample_generations_batch_hf_local(model=model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=num_gen_per_question-1, return_activations=True, return_text=True, return_model_output=False, **kwargs)

            most_likely = sample_generations_sequential_hf_local(model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=1, return_activations=True, return_text=True, top_p=1, temperature=None, return_model_output=False, **kwargs)
            
            train_data[i]['generated_texts'] = most_likely['generated_texts']
            train_data[i]['hidden_states'] = [most_likely['activations'][0][-1][layer_index].reshape(-1).float().numpy()]
            for j in range(len(sampled['generated_texts'])):
                if sampled['generated_texts'][j] in train_data[i]['generated_texts']:
                    continue
                train_data[i]['generated_texts'].append(sampled['generated_texts'][j])
                train_data[i]['hidden_states'].append(sampled['activations'][j][-1][layer_index].reshape(-1).float().numpy())

            train_data[i]["labels"] = [correctness_evaluator(train_data[i]['question'], answer, train_data[i]['ground_truths']) for answer in train_data[i]['generated_texts']]

        print("Generating answers and labels for validation data...")
        for i in tqdm(range(len(val_data))):
            messages = previous_context.copy()
            messages.append({'role': 'user', 'content': user_prompt.format(question_context = val_data[i]['question'])})

            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, continue_final_message=False)

            most_likely = sample_generations_sequential_hf_local(model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=1, return_activations=True, return_text=True, return_model_output=False, **kwargs)
            
            val_data[i]['generated_text'] = most_likely['generated_texts'][0]
            val_data[i]['hidden_states'] = most_likely['activations'][0][-1][layer_index].reshape(-1).float().numpy()
            val_data[i]["label"] = correctness_evaluator(val_data[i]['question'], most_likely['generated_texts'][0], val_data[i]['ground_truths'])
        
        return train_data, val_data


    @staticmethod
    def _prepare_data(train_data:list, val_data:list):
        print("Preparing data for training...")
        train_data_features = []
        train_data_labels = []
        for i in range(len(train_data)):
            for j in range(len(train_data[i]['generated_texts'])):
                if train_data[i]['labels'][j] == -1:
                    continue
                else:
                    train_data_features.append(train_data[i]['hidden_states'][j])
                    train_data_labels.append(train_data[i]['labels'][j])

        train_data_features = np.array(train_data_features)
        train_data_labels = np.array(train_data_labels)

        print("Preparing data for validation...")
        val_data_features = []
        val_data_labels = []
        for i in range(len(val_data)):
            val_data_features.append(val_data[i]['hidden_states'])
            val_data_labels.append(val_data[i]['label'])
        
        val_data_features = np.array(val_data_features)
        val_data_labels = np.array(val_data_labels)

        return (train_data_features, train_data_labels), (val_data_features, val_data_labels)    


    
    @staticmethod
    def _train(train_dataset:tuple, val_dataset:tuple, model:MLPClassifier, save_path:str=None,
            test_metrics:list[str]=['auroc']):
        
        

        print("SAPLMA training started...")

        train_data_features, train_data_labels = train_dataset
        val_data_features, val_data_labels = val_dataset

        model.fit(train_data_features, train_data_labels)

        scores = model.predict_proba(val_data_features)[:,1]
        labels = val_data_labels
        metric_scores = metric_score(metric_names=test_metrics, generation_correctness =  labels,  truth_values = scores, normalized_truth_values=scores)

        return {'model': model, 'metric_scores': metric_scores, 'train_data': train_dataset, 'val_data': val_dataset}
    
    @staticmethod
    def train_saplma_model(datasets:list, size_for_each_dataset:list, val_ratio:float,  chat_model:PreTrainedModel, 
                        chat_tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                        correctness_evaluator, seed:int=0, save_path:str=None,previous_context:list=[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}],
                        user_prompt:str = DEFAULT_USER_PROMPT, num_gen_per_question:int=5,
                        saplma_model_architecture:tuple=(256,128,64), test_metrics:list[str]=['auroc'],
                        layer_index:int=-1, mlp_args:dict={}, **kwargs):
        
        assert val_ratio > 0

        #get all datasets as a whole and split to train and val
        train_data, val_data = SAPLMA._get_datasets(datasets, size_for_each_dataset, val_ratio, seed)

        #set pad tokens
        pad_token_id = chat_tokenizer.pad_token_id
        if pad_token_id == None:
            pad_token_id = chat_model.config.eos_token_id
            chat_tokenizer.pad_token_id = pad_token_id

        #generate answers to the questions and get their labels
        train_data, val_data = SAPLMA._generate_answers_and_label(train_data=train_data, val_data=val_data, model=chat_model, tokenizer=chat_tokenizer, 
                                correctness_evaluator=correctness_evaluator, previous_context=previous_context,
                                user_prompt=user_prompt, num_gen_per_question=num_gen_per_question, layer_index=layer_index, **kwargs)

        # del chat_model
        # del chat_tokenizer
        
        #create MLP model
        saplma_model = MLPClassifier(hidden_layer_sizes=saplma_model_architecture, random_state=seed, **mlp_args)


        #prepare data for LARS training, return train and val datasets
        train_dataset, val_dataset = SAPLMA._prepare_data(train_data=train_data, val_data=val_data)

        output_dict = SAPLMA._train(train_dataset=train_dataset, val_dataset=val_dataset, model=saplma_model, save_path=save_path, test_metrics=test_metrics)
                
        return output_dict
