from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from TruthTorchLLM.trust_methods import TrustMethod
from .correctness_evaluator import CorrectnessEvaluator 
from .rouge import ROUGE
from TruthTorchLLM.generation import generate_with_trust_value
from TruthTorchLLM.availability import AVAILABLE_DATASETS, AVAILABLE_EVALUATION_METRICS
from TruthTorchLLM.templates import DEFAULT_TEMPLATE



def evaluate_trust_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  trust_method: TrustMethod, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None, eval_metric:str = 'auroc', 
                          correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data = 1.0, prompt_template = DEFAULT_TEMPLATE,seed = 31, return_model_generations = False, **kwargs):
    if type(dataset) == str and dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")
    if  eval_metric not in AVAILABLE_EVALUATION_METRICS:
        raise ValueError(f"Evaluation metric is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}")
    
    if dataset == "trivia_qa":
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=31)['train']
        dataset = []
        for i in range(len(raw_dataset)):
            input_text = prompt_template.format(question = raw_dataset['question'][i])
            ground_truths = raw_dataset['answer'][i]['aliases']

            dataset.append({'input_text': input_text, 'ground_truths': ground_truths})

    trust_values = []
    correctness = []
    if return_model_generations:
        model_generations = []
    for i in tqdm(range(len(dataset))):
        trust_dict = generate_with_trust_value(model, dataset[i]['input_text'], [trust_method], tokenizer=tokenizer, **kwargs)
        is_correct = correctness_evaluator.forward(trust_dict['generated_text'], dataset[i]['ground_truths'])
        correctness.append(is_correct)
        trust_values.append(trust_dict['unnormalized_trust_values'][0])
        if return_model_generations:
            generation_dict = {}
            generation_dict['generated_text'] = trust_dict['generated_text']
            generation_dict['trust_value'] = trust_dict['unnormalized_trust_values'][0]
            generation_dict['is_correct'] = is_correct
            generation_dict['input_text'] = dataset[i]['input_text']
            generation_dict['ground_truths'] = dataset[i]['ground_truths']
            model_generations.append(generation_dict)

    if eval_metric == 'auroc':
        try:
            auroc = roc_auc_score(correctness, trust_values)
        except:
            print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
            auroc = 0.5
        if return_model_generations:
            return {'auroc': auroc, 'accuracy_of_the_model': sum(correctness) / len(correctness), 'model_generations': model_generations}
        else:
            return {'auroc': auroc, 'accuracy_of_the_model': sum(correctness) / len(correctness)}