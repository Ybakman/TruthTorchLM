from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from datasets import load_dataset
from TruthTorchLLM.truth_methods import TruthMethod
from TruthTorchLLM.evaluators import CorrectnessEvaluator, ROUGE
from TruthTorchLLM.generation import generate_with_truth_value, completion_with_truth_value
from TruthTorchLLM.availability import AVAILABLE_DATASETS
from TruthTorchLLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT 
import numpy as np
import wandb


def find_threshold_std(correctness: list, truth_values: list, precision: float = -1, recall: float = -1):
    std = np.std(truth_values)
    precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
    if precision != -1:
        # Find the index of the smallest precision that is greater than or equal to the target precision
        index = np.where(precisions >= precision)[0][0]
        # Since precisions is always one element longer than thresholds, we need to adjust the index
        threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    elif recall != -1:
        # Find the index of the smallest recall that is greater than or equal to the target recall
        index = np.where(recalls >= recall)[0][0]
        # Since recalls is always one element longer than thresholds, we need to adjust the index
        threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    else:
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        
        # Remove NaN values and find the index of the highest F1 score
        f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
        max_index = np.argmax(f1_scores)
        
        # The thresholds array is of length len(precisions) - 1, so we use max_index-1 to get the corresponding threshold
        threshold = thresholds[max_index - 1] if max_index > 0 else thresholds[0]

    return threshold, std

def calibrate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] =None, precision:float = -1, 
                           recall:float = -1, correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data:float = 1.0, system_prompt:str = DEFAULT_SYSTEM_PROMPT, 
                           user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, wandb_run = None, return_method_details:bool = False, **kwargs):
    if type(dataset) == str and dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")
    
    if dataset == "trivia_qa":
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
        dataset = []
        for i in range(len(raw_dataset)):
            ground_truths = raw_dataset['answer'][i]['aliases']
            dataset.append({'question': raw_dataset['question'][i], 'ground_truths': ground_truths})

    output_dict = {}
    output_dict['generation'] = []
    output_dict['generation_correctness'] = []

    for i in range(len(truth_methods)):
        output_dict[i] = {}
        output_dict[i]['truth_values'] = []
        output_dict[i]['normalized_truth_values'] = []  
        if return_method_details:
            output_dict[i]['method_specific_details'] = []

    if wandb_run is not None:
        logged_data = []
        #add method names to the columns
        names = []
        columns = []
        for i in range(len(truth_methods)):
            names.append(str(truth_methods[i]))
            columns.append(f'truth_method_{i}')
        
        names_table = wandb.Table(data = [names], columns=columns)
        wandb_run.log({"method_names": names_table})


    for i in tqdm(range(len(dataset))):
        messages = [{'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])}]
        if type(model) == str:
            truth_dict = completion_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = truth_methods, generation_seed = seed, **kwargs)
        else:
            truth_dict = generate_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = truth_methods, tokenizer=tokenizer, generation_seed = seed, **kwargs)
        is_correct = correctness_evaluator(truth_dict['generated_text'], dataset[i]['ground_truths'])
        output_dict['generation_correctness'] .append(is_correct)
        
        for j in range(len(truth_methods)):
            output_dict[j]['truth_values'].append(truth_dict['unnormalized_truth_values'][j])
            output_dict[j]['normalized_truth_values'].append(truth_dict['normalized_truth_values'][j])
            if return_method_details:
                output_dict[j]['method_specific_details'].append(truth_dict['method_specific_outputs'][j])


        if wandb_run is not None:
            for j in range(len(truth_methods)):
                wandb_run.log({
                    f'truth_values_{j}': truth_dict['unnormalized_truth_values'][j],
                    f'normalized_truth_values_{j}': truth_dict['normalized_truth_values'][j],
                })
            if return_method_details:
                columns=['truth_values','normalized_truth_values', 'generation_correctness', 
            'question_text', 'ground_truths', 'generated_text', 'index', 'method_specific_details']

                data = [str(truth_dict['unnormalized_truth_values']), str(truth_dict['normalized_truth_values']), is_correct, dataset[i]['question'], 
                    (', ').join(dataset[i]['ground_truths']) , truth_dict['generated_text'], i, str(truth_dict['method_specific_outputs'])]
            else:
                columns= ['truth_values','normalized_truth_values', 'generation_correctness', 
            'question_text', 'ground_truths', 'generated_text', 'index']
                data = [str(truth_dict['unnormalized_truth_values']), str(truth_dict['normalized_truth_values']), is_correct, dataset[i]['question'], (', ').join(dataset[i]['ground_truths']) , truth_dict['generated_text'], i]
            
            logged_data.extend([data])
            summary_table = wandb.Table(data = logged_data, columns=columns)
            wandb_run.log({
                'accuracy': is_correct,
                'index': i,
            })
            wandb.log({"run_summary" : summary_table})

    for i, truth_method in enumerate(truth_methods):
        truth_values = output_dict[i]['truth_values']
        correctness = output_dict['generation_correctness']
        threshold, std = find_threshold_std(correctness, truth_values, precision, recall)
        truth_method.set_threshold(threshold)
        truth_method.set_std(std)
        print(f'Truth method_{i} is calibrated with the following parameters: threshold =', threshold, 'std =', std)
        if wandb_run is not None:
            wandb_run.log({f'threshold_of_method_{i}': threshold})
            wandb_run.log({f'std_of_method_{i}': std})


    return truth_method


    



    