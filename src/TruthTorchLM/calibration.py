from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.truth_methods import TruthMethod
from TruthTorchLM.evaluators import CorrectnessEvaluator, ROUGE
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT 
from TruthTorchLM.utils.dataset_utils import get_dataset
from TruthTorchLM.utils.eval_utils import run_over_dataset
import numpy as np



def calibrate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list[TruthMethod], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] =None,
correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), size_of_data:float = 1.0, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], 
 user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, wandb_run = None, return_method_details:bool = False, wandb_push_method_details:bool = False, split = 'train', **kwargs):
    
    dataset = get_dataset(dataset, size_of_data=size_of_data, seed=seed, split = split)

    output_dict = run_over_dataset(dataset, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = correctness_evaluator, 
                                   previous_context = previous_context, user_prompt = user_prompt, seed = seed, return_method_details = return_method_details, 
                                   wandb_run = wandb_run, wandb_push_method_details = wandb_push_method_details, **kwargs)

    for i, truth_method in enumerate(truth_methods):
        truth_values = output_dict[f'truth_method_{i}']['truth_values']
        truth_values = np.array(truth_values)
        truth_values[np.isnan(truth_values)] = 0
        correctness = output_dict['generation_correctness']
        #if generation_correctness is -1, it means that the model didn't attempt to generate an answer, remove those from the evaluation
        truth_method.normalizer.calibrate(generation_performance_scores=correctness, truth_values=truth_values)
    return output_dict 


    



    