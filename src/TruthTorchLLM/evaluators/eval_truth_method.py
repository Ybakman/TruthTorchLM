from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from TruthTorchLLM.truth_methods import TruthMethod
from .correctness_evaluator import CorrectnessEvaluator 
from .rouge import ROUGE
from TruthTorchLLM.generation import generate_with_truth_value, completion_with_truth_value
from TruthTorchLLM.availability import AVAILABLE_DATASETS, AVAILABLE_EVALUATION_METRICS
from TruthTorchLLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
import wandb


def evaluate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, eval_metrics:list[str] = ['auroc'],
                          correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data = 1.0,  system_prompt:str = DEFAULT_SYSTEM_PROMPT, 
                           user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, return_method_details:bool = False, wandb_run = None,  **kwargs):
    if type(dataset) == str and dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")

    for eval_metric in eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}")

    
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
        output_dict['generation_correctness'].append(is_correct)
        output_dict['generation'].append(truth_dict['generated_text'])

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

    
    eval_list = []
    for i in range(len(truth_methods)):
        eval_dict = {}
        if 'auroc' in eval_metrics:
            try:
                auroc = roc_auc_score(output_dict['generation_correctness'], output_dict[i]['truth_values'])
            except:
                print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
                auroc = 0.5
            eval_dict['auroc'] = auroc

        eval_list.append(eval_dict)

    if wandb_run:
        # wandb_run.log({"run_summary" : text_table})
        for i, eval_dict in enumerate(eval_list):
            for key, value in eval_dict.items():
                wandb_run.log({f'{key}_of_method_{i}': value})

    return {'eval_list': eval_dict, 'output_dict': output_dict}