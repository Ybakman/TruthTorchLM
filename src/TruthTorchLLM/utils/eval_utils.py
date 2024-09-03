from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from sklearn.metrics import roc_auc_score
from TruthTorchLLM.truth_methods import TruthMethod
from TruthTorchLLM.generation import generate_with_truth_value, completion_with_truth_value
from TruthTorchLLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
import wandb
from sklearn.metrics import roc_auc_score



def metric_score(metric_names:list[str], generation_correctness:list, truth_values:list[float], normalized_truth_values:list[float] = [],  seed:int = 0) -> dict:
    eval_dict = {}
    if "auroc" in metric_names:
        try:
            auroc = roc_auc_score(generation_correctness, truth_values)
        except:
            print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
            auroc = 0.5
        eval_dict['auroc'] = auroc
    
    return eval_dict


def run_over_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          correctness_evaluator = None, system_prompt:str = DEFAULT_SYSTEM_PROMPT, 
                           user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, return_method_details:bool = False, wandb_run = None,  **kwargs):
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

    return output_dict
    
    

    
   
