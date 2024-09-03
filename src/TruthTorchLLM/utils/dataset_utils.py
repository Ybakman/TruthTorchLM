from datasets import load_dataset
from TruthTorchLLM.availability import AVAILABLE_DATASETS
from typing import Union



def get_dataset(dataset:Union[str, list], fraction_of_data:float = 1.0, seed:int = 0):
    if type(dataset) != str:
        if len(dataset) == 0:
            raise ValueError("Dataset list is empty.")
        if 'question' not in dataset[0].keys() or 'ground_truths' not in dataset[0].keys():
            raise ValueError("Dataset should have 'question' and 'ground_truths' keys.")
        return dataset
    
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")
    
    if dataset == "trivia_qa":
        dataset = get_trivia_qa(fraction_of_data, seed)
    
    return dataset


def get_trivia_qa(fraction_of_data:float = 1.0, seed:int = 0):
    raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
    dataset = []
    for i in range(len(raw_dataset)):
        ground_truths = raw_dataset['answer'][i]['aliases']
        dataset.append({'question': raw_dataset['question'][i], 'ground_truths': ground_truths})

    return dataset