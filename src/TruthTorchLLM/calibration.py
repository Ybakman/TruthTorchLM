from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLLM.truth_methods import TruthMethod
from TruthTorchLLM.evaluators import CorrectnessEvaluator, ROUGE
from TruthTorchLLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT 
from TruthTorchLLM.utils.dataset_utils import get_dataset
from TruthTorchLLM.utils.eval_utils import run_over_dataset
from TruthTorchLLM.utils.common_utils import find_threshold_std


def calibrate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] =None, precision:float = -1, 
                           recall:float = -1, correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data:float = 1.0, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}], 
                          user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, wandb_run = None, return_method_details:bool = False, wandb_push_method_details:bool = False, **kwargs):
    
    dataset = get_dataset(dataset, fraction_of_data=fraction_of_data, seed=seed)

    output_dict = run_over_dataset(dataset, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = correctness_evaluator, 
                                   previous_context = previous_context, user_prompt = user_prompt, seed = seed, return_method_details = return_method_details, 
                                   wandb_run = wandb_run, wandb_push_method_details = wandb_push_method_details, **kwargs)

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

    return output_dict 


    



    