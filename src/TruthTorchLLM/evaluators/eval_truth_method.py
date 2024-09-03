from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLLM.truth_methods import TruthMethod
from .correctness_evaluator import CorrectnessEvaluator 
from .rouge import ROUGE
from TruthTorchLLM.availability import AVAILABLE_EVALUATION_METRICS
from TruthTorchLLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLLM.utils.dataset_utils import get_dataset
from TruthTorchLLM.utils.eval_utils import metric_score, run_over_dataset


def evaluate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, eval_metrics:list[str] = ['auroc'],
                          correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data = 1.0,  system_prompt:str = DEFAULT_SYSTEM_PROMPT, 
                           user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, return_method_details:bool = False, wandb_run = None,  **kwargs):
    
    dataset = get_dataset(dataset, fraction_of_data=fraction_of_data, seed=seed)

    for eval_metric in eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}")
                
    output_dict = run_over_dataset(dataset, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = correctness_evaluator, 
                                   system_prompt = system_prompt, user_prompt = user_prompt, seed = seed, return_method_details = return_method_details, 
                                   wandb_run = wandb_run, **kwargs)

    eval_list = []
    for i in range(len(truth_methods)):
        eval_dict = metric_score(eval_metrics, output_dict['generation_correctness'], output_dict[i]['truth_values'], output_dict[i]['normalized_truth_values'], seed=seed)
        eval_list.append(eval_dict)

    if wandb_run:
        for i, eval_dict in enumerate(eval_list):
            for key, value in eval_dict.items():
                wandb_run.log({f'{key}_of_method_{i}': value})

    return {'eval_list': eval_list, 'output_dict': output_dict}