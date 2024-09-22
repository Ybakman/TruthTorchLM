from datasets import load_dataset
from TruthTorchLLM.availability import AVAILABLE_DATASETS
from typing import Union
from tqdm import tqdm



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
    elif dataset == "gsm8k":
        dataset = get_gsm8k(fraction_of_data, seed)
    elif dataset == "natural_qa":
        dataset = get_natural_qa(fraction_of_data, seed)
    elif dataset == "pop_qa":
        dataset = get_pop_qa(fraction_of_data, seed)
    
    return dataset


def get_trivia_qa(fraction_of_data:float = 1.0, seed:int = 0):
    raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    if fraction_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
    dataset = []
    answers = raw_dataset['answer']
    questions = raw_dataset['question']
    for i in tqdm(range(len(raw_dataset))):
        ground_truths = answers[i]['aliases']
        dataset.append({'question': questions[i], 'ground_truths': ground_truths})

    return dataset

def get_gsm8k(fraction_of_data:float = 1.0, seed:int = 0):
    raw_dataset = load_dataset("openai/gsm8k", "main", split='test')
    if fraction_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
    dataset = []
    answers = raw_dataset['answer']
    questions = raw_dataset['question']
    for i in tqdm(range(len(raw_dataset))):
        answer = answers[i].split('####')[1].strip()
        dataset.append({'question': questions[i], 'ground_truths': [answer]})

    return dataset


def get_natural_qa(fraction_of_data:float = 1.0, seed:int = 0):
    raw_dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    if fraction_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
    dataset = []
    questions = raw_dataset['question']
    answers = raw_dataset['answer']
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({'question': questions[i], 'ground_truths': answers[i]})

    return dataset

def get_pop_qa(fraction_of_data:float = 1.0, seed:int = 0):
    raw_dataset = load_dataset("akariasai/PopQA", split='test')
    if fraction_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(test_size=(1 - fraction_of_data), seed=seed)['train']
    dataset = []
    questions = raw_dataset['question']
    answers = raw_dataset['possible_answers']
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({'question': questions[i], 'ground_truths': answers[i]})

    return dataset