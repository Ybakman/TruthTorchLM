from datasets import load_dataset
from TruthTorchLM.availability import AVAILABLE_DATASETS
from typing import Union
from tqdm import tqdm


def get_dataset(
    dataset: Union[str, list], size_of_data: float = 1.0, seed: int = 0, split="test"
):
    if type(dataset) != str:
        if len(dataset) == 0:
            raise ValueError("Dataset list is empty.")
        if (
            "question" not in dataset[0].keys()
            or "ground_truths" not in dataset[0].keys()
        ):
            raise ValueError(
                "Dataset should have 'question' and 'ground_truths' keys.")
        return dataset

    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}"
        )

    print(
        "Loading dataset from Huggingface Datasets, split:",
        split,
        "fraction of data:",
        size_of_data,
    )

    if dataset == "trivia_qa":
        dataset = get_trivia_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "gsm8k":
        dataset = get_gsm8k(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "natural_qa":
        dataset = get_natural_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "pop_qa":
        dataset = get_pop_qa(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "simple_qa":
        dataset = get_simple_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "wikipedia_factual":
        dataset = get_wikipedia_factual(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "narrative_qa":
        dataset = get_narrative_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "web_questions":
        dataset = get_web_questions(
            size_of_data=size_of_data, seed=seed, split=split)

    return dataset


def get_trivia_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):

    if split == "test":
        raw_dataset = load_dataset(
            "trivia_qa", "rc.nocontext", split="validation")
    elif split == "train":
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    answers = raw_dataset["answer"]
    questions = raw_dataset["question"]
    for i in tqdm(range(len(raw_dataset))):
        ground_truths = answers[i]["aliases"]
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": ground_truths})

    return dataset


def get_gsm8k(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("openai/gsm8k", "main", split="test")
    elif split == "train":
        raw_dataset = load_dataset("openai/gsm8k", "main", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    answers = raw_dataset["answer"]
    questions = raw_dataset["question"]
    for i in tqdm(range(len(raw_dataset))):
        answer = answers[i].split("####")[1].strip()
        dataset.append({"context": "", "question": questions[i], "ground_truths": [answer]})

    return dataset


def get_natural_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset(
            "google-research-datasets/nq_open", split="validation"
        )
    elif split == "train":
        raw_dataset = load_dataset(
            "google-research-datasets/nq_open", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["question"]
    answers = raw_dataset["answer"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({"context": "", "question": questions[i], "ground_truths": answers[i]})

    return dataset


def get_pop_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("akariasai/PopQA", split="test")
    elif split == "train":
        raw_dataset = load_dataset("akariasai/PopQA", split="test")
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["question"]
    answers = raw_dataset["possible_answers"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": [answers[i]]})

    return dataset


def get_simple_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split="test")
    elif split == "train":
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split="test")
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["problem"]
    answers = raw_dataset["answer"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": [answers[i]]})

    return dataset


def get_wikipedia_factual(size_of_data: float = 1.0, seed: int = 0, split='train'):
    raw_dataset = load_dataset("achorn123/wikipedia_factual_dataset_500", split='train')

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']

    dataset = []
    for data in tqdm(raw_dataset, desc="Processing Wikipedia factual 500"):
        context = data["context"].strip()
        question = data["question"].strip()
        answer = data["answer"].strip()
        dataset.append({
            'context': context,
            'question': question,
            'ground_truths': [answer]
        })

    return dataset


def get_narrative_qa(size_of_data: float = 1.0, seed: int = 0, split='test'):
    raw_dataset = load_dataset("deepmind/narrativeqa", split=split)

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']

    dataset = []
    for data in tqdm(raw_dataset, desc="Processing NarrativeQA"):
        context = data["document"]["text"].strip()
        question = data["question"]["text"].strip()
        answers = []
        for answer in data["answers"]:
            answers.append(answer["text"].strip())

        dataset.append({
            "context": context,
            "question": question,
            "ground_truths": answers
        })
        
    return dataset

def get_web_questions(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("stanfordnlp/web_questions", split="test")
    elif split == "train":
        raw_dataset = load_dataset("stanfordnlp/web_questions", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["question"]
    answers = raw_dataset["answers"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": answers[i]})

    return dataset