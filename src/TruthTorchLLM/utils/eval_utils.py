from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLLM.truth_methods import TruthMethod
from TruthTorchLLM.generation import generate_with_truth_value, completion_with_truth_value
from TruthTorchLLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np



def area_under_accuracy_coverage_curve(t_s, acc):
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    df = pd.DataFrame({"t_s": t_s, 'acc': acc}).sort_values('t_s', ascending=False)#that should be false in case of truth values
    df['acc_mean'] = df['acc'].expanding().mean()
    return auc(np.linspace(0,1,len(df)), df['acc_mean'])

def metric_score(metric_names:list[str], generation_correctness:list, truth_values:list[float], normalized_truth_values:list[float] = [],  seed:int = 0) -> dict:
    eval_dict = {}
    #replace NaN values with a big number
    truth_values = np.array(truth_values)
    truth_values[np.isnan(truth_values)] = 1000000
    normalized_truth_values = np.array(normalized_truth_values)
    normalized_truth_values[np.isnan(normalized_truth_values)] = 1000000

    truth_values = list(truth_values) #convert to list
    normalized_truth_values = (normalized_truth_values) #convert to list

    if "auroc" in metric_names:
        try:
            auroc = roc_auc_score(generation_correctness, truth_values)
        except:
            print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
            auroc = 0.5
        eval_dict['auroc'] = auroc

    if 'auprc' in metric_names:
        precision, recall, thresholds = precision_recall_curve(generation_correctness, truth_values)
        auprc = auc(recall, precision)
        eval_dict['auprc'] = auprc

    if 'auarc' in metric_names:
        #area under accuracy-coverage curve
        auarc = area_under_accuracy_coverage_curve(truth_values, generation_correctness)
        eval_dict['auarc'] = auarc

    if 'accuracy' in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        accuracy = np.mean((normalized_truth_values > 0.5) == generation_correctness)
        eval_dict['accuracy'] = accuracy
    
    return eval_dict


def run_over_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          correctness_evaluator = None, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, return_method_details:bool = False, wandb_run = None, 
                          wandb_push_method_details:bool = False, batch_generation=True,  add_generation_prompt = True, continue_final_message = False, **kwargs):
    output_dict = {}
    output_dict['previous_context'] = previous_context
    output_dict['user_prompt'] = user_prompt
    output_dict['generation'] = []
    output_dict['generation_correctness'] = []
    output_dict['question_text'] = []
    output_dict['ground_truths'] = []

    
    for i in range(len(truth_methods)):
        output_dict[f'truth_method_{i}'] = {}
        output_dict[f'truth_method_{i}']['name'] = str(truth_methods[i])
        output_dict[f'truth_method_{i}']['truth_values'] = []
        output_dict[f'truth_method_{i}']['normalized_truth_values'] = []  
        if return_method_details:
            output_dict[f'truth_method_{i}']['method_specific_details'] = []

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
        messages = previous_context.copy()
        messages.append({'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])})
        if type(model) == str:
            truth_dict = completion_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = truth_methods, generation_seed = seed, **kwargs)
        else:
            truth_dict = generate_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = truth_methods, tokenizer=tokenizer, 
            generation_seed = seed,batch_generation=batch_generation, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message, **kwargs)

        is_correct = correctness_evaluator(dataset[i]['question'], truth_dict['generated_text'], dataset[i]['ground_truths'])
        output_dict['generation_correctness'].append(is_correct)
        output_dict['generation'].append(truth_dict['generated_text'])
        output_dict['question_text'].append(dataset[i]['question'])
        output_dict['ground_truths'].append(dataset[i]['ground_truths'])
        
        for j in range(len(truth_methods)):
            output_dict[f'truth_method_{j}']['truth_values'].append(truth_dict['unnormalized_truth_values'][j])
            output_dict[f'truth_method_{j}']['normalized_truth_values'].append(truth_dict['normalized_truth_values'][j])
            if return_method_details:
                output_dict[f'truth_method_{j}']['method_specific_details'].append(truth_dict['method_specific_outputs'][j])
            
            if wandb_push_method_details and wandb_run is not None:
                columns=['truth_values','normalized_truth_values', 'generation_correctness', 
            'question_text', 'ground_truths', 'generated_text', 'index', 'method_specific_details']

                data = [str(truth_dict['unnormalized_truth_values']), str(truth_dict['normalized_truth_values']), is_correct, dataset[i]['question'], 
                    (', ').join(dataset[i]['ground_truths']) , truth_dict['generated_text'], i, str(truth_dict['method_specific_outputs'])]
            elif wandb_run is not None:
                columns= ['truth_values','normalized_truth_values', 'generation_correctness', 
            'question_text', 'ground_truths', 'generated_text', 'index']
                data = [str(truth_dict['unnormalized_truth_values']), str(truth_dict['normalized_truth_values']), is_correct, dataset[i]['question'], (', ').join(dataset[i]['ground_truths']) , truth_dict['generated_text'], i]
        if wandb_run is not None and wandb_push_method_details:
            logged_data.extend([data])
            summary_table = wandb.Table(data = logged_data, columns=columns)
            wandb_run.log({
                'accuracy': is_correct,
                'index': i,
            })
            wandb.log({"run_summary" : summary_table})


    return output_dict
    
    

    
   
