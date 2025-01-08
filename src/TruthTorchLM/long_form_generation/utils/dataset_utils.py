from datasets import load_dataset
from TruthTorchLM.availability import LONG_FORM_AVAILABLE_DATASETS
from typing import Union
from tqdm import tqdm

import requests
import os

import json
import random
from typing import List



def get_dataset(dataset:Union[str, list], size_of_data:int, seed:int = 0):
    if type(dataset) != str:
        assert len(dataset) > 0, "Dataset list is empty."
        assert 'question' in dataset[0], "Dataset should have 'question' key."
        if "statements" in dataset[0]:
            assert 'statement_correctness' in dataset[0], "Statement correctness scores is missing."
        if "statement_correctness" in dataset[0]:
            assert 'statements' in dataset[0], "Statements are missing."
        return dataset
    
    if dataset not in LONG_FORM_AVAILABLE_DATASETS:
        raise ValueError(f"Dataset is not available. Available datasets are: {LONG_FORM_AVAILABLE_DATASETS}")

    print("Loading dataset... Size of data:", size_of_data)
    
    if dataset == "longfact_concepts":
        dataset = get_longfact(branch="longfact-concepts_gpt4_01-10-2024", size_of_data=size_of_data, seed=seed)  
    elif dataset == "longfact_objects":
        dataset = get_longfact(branch="longfact-objects_gpt4_01-12-2024", size_of_data=size_of_data, seed=seed)
    return dataset

def get_longfact(branch:str, size_of_data:int=100, seed:int = 0):
    #Download data
    download_github_folder('google-deepmind', 'long-form-factuality', 'main', f'longfact/{branch}_noduplicates', f'./datasets/{branch}', None)
    #Load data
    questions = []
    for file_name in os.listdir(f'./datasets/{branch}'):
        with open(f'./datasets/{branch}/' + file_name, 'r') as file:
            for line in file:
                questions.append({'question': json.loads(line)['prompt']})

    if size_of_data < len(questions):
        return random.sample(questions, size_of_data)
    return questions


def download_file(file_info, local_file_path, headers):
    """Download a single file from GitHub."""
    download_url = file_info['download_url']
    response = requests.get(download_url, headers=headers)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as f:
            f.write(response.content)
        # print(f"Downloaded: {file_info['path']}")
    else:
        print(f"Failed to download {file_info['path']}: {response.status_code}")

def download_contents(owner, repo, branch, path, local_path, headers):
    """Recursively download contents from a GitHub folder."""
    api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}'
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        items = response.json()
        if isinstance(items, dict) and items.get('type') == 'file':
            # It's a single file
            download_file(items, local_path, headers)
        else:
            # It's a directory
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            for item in items:
                item_path = item['path']
                item_type = item['type']
                item_local_path = os.path.join(local_path, item['name'])
                if item_type == 'file':
                    download_file(item, item_local_path, headers)
                elif item_type == 'dir':
                    download_contents(owner, repo, branch, item_path, item_local_path, headers)
    else:
        print(f"Failed to get contents of {path}: {response.status_code}")

def download_github_folder(owner, repo, branch, folder_path, local_dir, token=None):
    headers = {'Authorization': f'token {token}'} if token else {}
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    download_contents(owner, repo, branch, folder_path, local_dir, headers)
