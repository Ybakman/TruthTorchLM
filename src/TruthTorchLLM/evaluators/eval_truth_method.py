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



def evaluate_truth_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_method: TruthMethod, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, eval_metrics:list[str] = ['auroc'],
                          correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data = 1.0,  system_prompt:str = DEFAULT_SYSTEM_PROMPT, 
                           user_prompt:str = DEFAULT_USER_PROMPT, seed = 31, return_model_generations = False, **kwargs):
    if type(dataset) == str and dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")

    for eval_metric in eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}")

    
    if dataset == "trivia_qa":
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=31)['train']
        dataset = []
        for i in range(len(raw_dataset)):
            ground_truths = raw_dataset['answer'][i]['aliases']
            dataset.append({'question': raw_dataset['question'][i], 'ground_truths': ground_truths})

    truth_values = []
    correctness = []
    if return_model_generations:
        model_generations = []
    for i in tqdm(range(len(dataset))):
        messages = [{'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])}]
        if type(model) == str:
            truth_dict = completion_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = [truth_method], **kwargs)
        else:
            truth_dict = generate_with_truth_value(model, messages, question_context = dataset[i]['question'], truth_methods = [truth_method], tokenizer=tokenizer, **kwargs)

        is_correct = correctness_evaluator(truth_dict['generated_text'], dataset[i]['ground_truths'])
        correctness.append(is_correct)
        truth_values.append(truth_dict['unnormalized_truth_values'][0])
        if return_model_generations:
            generation_dict = {}
            generation_dict['generated_text'] = truth_dict['generated_text']
            generation_dict['truth_value'] = truth_dict['unnormalized_truth_values'][0]
            generation_dict['is_correct'] = is_correct
            generation_dict['input_text'] = dataset[i]['input_text']
            generation_dict['ground_truths'] = dataset[i]['ground_truths']
            model_generations.append(generation_dict)

    eval_dict = {}
    if 'auroc' in eval_metrics:
        try:
            auroc = roc_auc_score(correctness, truth_values)
        except:
            print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
            auroc = 0.5
        eval_dict['auroc'] = auroc

    if return_model_generations:
        return {'eval_dict': eval_dict, 'accuracy_of_the_model': sum(correctness) / len(correctness), 'model_generations': model_generations}
    else:
        return {'eval_dict': eval_dict, 'accuracy_of_the_model': sum(correctness) / len(correctness)}