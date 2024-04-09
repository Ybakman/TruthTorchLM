from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from datasets import load_dataset
from TruthTorchLLM.trust_methods import TrustMethod
from TruthTorchLLM.evaluators import CorrectnessEvaluator, ROUGE
from TruthTorchLLM.generation import generate_with_trust_value
from TruthTorchLLM.availability import AVAILABLE_DATASETS
from TruthTorchLLM.templates import DEFAULT_TEMPLATE 
import numpy as np

def calibrate_trust_method(dataset: Union[str, list], model:Union[str,PreTrainedModel],  trust_method: TrustMethod, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] =None, precision:float = -1, 
                           recall:float = -1, correctness_evaluator:CorrectnessEvaluator = ROUGE(0.7), fraction_of_data:float = 1.0, prompt_template = DEFAULT_TEMPLATE,seed:int = 31, **kwargs):
    if type(dataset) == str and dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")
    
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
    for i in tqdm(range(len(dataset))):
        trust_dict = generate_with_trust_value(model, dataset[i]['input_text'], [trust_method], tokenizer=tokenizer, **kwargs)
        is_correct = correctness_evaluator(trust_dict['generated_text'], dataset[i]['ground_truths'])
        correctness.append(is_correct)
        trust_values.append(trust_dict['unnormalized_trust_values'][0])
      

    print(trust_values)
    
    std = np.std(trust_values)
    precisions, recalls, thresholds = precision_recall_curve(correctness, trust_values)
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

    trust_method.threshold = threshold
    trust_method.std = std
    print('The model is calibrated with the following parameters: threshold =', threshold, 'std =', std)
    return trust_method