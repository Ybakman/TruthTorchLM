from datasets import load_dataset
from TruthTorchLM.availability import AVAILABLE_DATASETS
from typing import Union
from tqdm import tqdm



def get_dataset(dataset:Union[str, list], size_of_data:float = 1.0, seed:int = 0, split = 'test'):
    if type(dataset) != str:
        if len(dataset) == 0:
            raise ValueError("Dataset list is empty.")
        if 'question' not in dataset[0].keys() or 'ground_truths' not in dataset[0].keys():
            raise ValueError("Dataset should have 'question' and 'ground_truths' keys.")
        return dataset
    
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}")

    print("Loading dataset from Huggingface Datasets, split:", split, "fraction of data:", size_of_data)
    
    if dataset == "trivia_qa":
        dataset = get_trivia_qa(size_of_data=size_of_data, seed=seed, split=split)  
    elif dataset == "gsm8k":
        dataset = get_gsm8k(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "natural_qa":
        dataset = get_natural_qa(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "pop_qa":
        dataset = get_pop_qa(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "simple_qa":
        dataset = get_simple_qa(size_of_data=size_of_data, seed=seed, split=split)
    
    return dataset



def get_trivia_qa(size_of_data:float = 1.0, seed:int = 0, split = 'test'):

    if split == 'test':
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    elif split == 'train':
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")

    if size_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']
    dataset = []
    answers = raw_dataset['answer']
    questions = raw_dataset['question']
    for i in tqdm(range(len(raw_dataset))):
        ground_truths = answers[i]['aliases']
        dataset.append({'question': questions[i], 'ground_truths': ground_truths})

    return dataset

def get_gsm8k(size_of_data:float = 1.0, seed:int = 0, split = 'test'):
    if split == 'test':
        raw_dataset = load_dataset("openai/gsm8k", "main", split='test')
    elif split == 'train':
        raw_dataset = load_dataset("openai/gsm8k", "main", split='train')
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']
    dataset = []
    answers = raw_dataset['answer']
    questions = raw_dataset['question']
    for i in tqdm(range(len(raw_dataset))):
        answer = answers[i].split('####')[1].strip()
        dataset.append({'question': questions[i], 'ground_truths': [answer]})

    return dataset


def get_natural_qa(size_of_data:float = 1.0, seed:int = 0, split = 'test'):
    if split == 'test':
        raw_dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    elif split == 'train':
        raw_dataset = load_dataset("google-research-datasets/nq_open", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']
    dataset = []
    questions = raw_dataset['question']
    answers = raw_dataset['answer']
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({'question': questions[i], 'ground_truths': answers[i]})

    return dataset

def get_pop_qa(size_of_data:float = 1.0, seed:int = 0, split = 'test'):
    if split == 'test':
        raw_dataset = load_dataset("akariasai/PopQA", split='test')
    elif split == 'train':
        raw_dataset = load_dataset("akariasai/PopQA", split='test')
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']
    dataset = []
    questions = raw_dataset['question']
    answers = raw_dataset['possible_answers']
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({'question': questions[i], 'ground_truths': [answers[i]]})

    return dataset


def get_simple_qa(size_of_data:float = 1.0, seed:int = 0, split = 'test'):
    if split == 'test':
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split='test')
    elif split == 'train':
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split='test')
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']
    dataset = []
    questions = raw_dataset['problem']
    answers = raw_dataset['answer']
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({'question': questions[i], 'ground_truths': [answers[i]]})

    return dataset