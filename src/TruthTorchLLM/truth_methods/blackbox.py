import copy
import torch
import random
import numpy as np
from typing import Union
from litellm import completion
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DebertaForSequenceClassification, DebertaTokenizer
from scipy.sparse.csgraph import laplacian
from TruthTorchLLM.utils import *
from TruthTorchLLM.availability import PROB_AVAILABLE_API_MODELS
from scipy.linalg import eigh
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer


def check_entailment(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str):
    inputs = tokenizer_for_entailment.encode_plus(
        text=context + " " + seq1,
        text_pair=context + " " + seq2,
        return_tensors='pt',
        truncation=True
    )
    outputs = model_for_entailment(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1) # contradiction, neutral, entailment
    return probs[0]

class BlackBoxMethods():
    def __init__(self, method_for_similarity: str = "semantic", number_of_generations=5, threshold=0.5, std=1.0, model_for_entailment: PreTrainedModel = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli'), tokenizer_for_entailment: PreTrainedTokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')):
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.model_for_entailment = model_for_entailment
        self.number_of_generations = number_of_generations
        self.threshold = threshold
        self.std = std
        self.method_for_similarity=method_for_similarity #jaccard or semantic

    def get_D_mat(self, W):
        D = np.diag(np.sum(W, axis=1))
        return D

    def get_L_mat(self, W, symmetric=True):
        # Compute the normalized Laplacian matrix from the degree matrix and weighted adjacency matrix
        D = self.get_D_mat(W)
        if symmetric:
            L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
        else:
            raise NotImplementedError()
            L = np.linalg.inv(D) @ (D - W)
        return L.copy()

    def calculate_affinity_matrix(self, texts: list[str], context: str):
        n = len(texts)
        affinity_matrix = np.zeros((n, n))
        
        if self.method_for_similarity == "semantic":
            for i in range(n):
                for j in range(i + 1, n):
                    left = check_entailment(self.model_for_entailment, self.tokenizer_for_entailment, context, texts[i], texts[j])[2].item()
                    right = check_entailment(self.model_for_entailment, self.tokenizer_for_entailment, context, texts[j], texts[i])[2].item()
                    affinity_matrix[i][j] = affinity_matrix[j][i] = (left + right) / 2
        
        elif self.method_for_similarity  == "jaccard":
            vectorizer = CountVectorizer().fit_transform(texts)
            vectors = vectorizer.toarray()
            for i in range(n):
                for j in range(i + 1, n):
                    affinity_matrix[i][j] = affinity_matrix[j][i] = jaccard_score(vectors[i], vectors[j], average='macro')
        
        return affinity_matrix

    def calculate_U_eigv(self, texts: list[str], context:str):
        W = self.calculate_affinity_matrix(texts, context)
        L = self.get_L_mat(W)
        eigvals = np.linalg.eigvalsh(L)
        U_eigv = sum(max(0, 1 - eig) for eig in eigvals)
        return U_eigv

    def calculate_U_deg(self, texts: list[str], context: str):
        W = self.calculate_affinity_matrix(texts, context)
        D = self.get_D_mat(W)
        m = len(W)
        U_deg = np.trace(m * np.identity(m) - D) / (m ** 2)
        return U_deg

    def calculate_U_ecc(self, texts: list[str], context: str):
        W = self.calculate_affinity_matrix(texts, context)
        L = self.get_L_mat(W, symmetric=True)
        eigvals, eigvecs = eigh(L)
        k = 5  
        V = eigvecs[:, :k]
        
        V_mean = np.mean(V, axis=0)
        V_prime = V - V_mean
        
        U_ecc = np.linalg.norm(V_prime, axis=1).sum()
        return U_ecc

    def calculate_U_num_set(self, texts: list[str], context: str):
        clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, context, texts, self.method_for_similarity)
        return len(clusters)

    def generate_forward(self, model: PreTrainedModel, input_text: str, generated_text:str, question_context:str, all_ids: Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, argument: str = "all", **kwargs):
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        kwargs.pop('do_sample', None)
        kwargs.pop('num_return_sequences', None)
        for i in range(self.number_of_generations):
            model_output = model.generate(input_ids, num_return_sequences=1, do_sample=True, **kwargs)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        if argument == "eigv":
            U_eigv = self.calculate_U_eigv(generated_texts, question_context)
            return {"U_eigv": U_eigv}
        elif argument == "deg":
            U_deg = self.calculate_U_deg(generated_texts, question_context)
            return {"U_deg": U_deg}
        elif argument == "ecc":
            U_ecc = self.calculate_U_ecc(generated_texts, question_context)
            return {"U_ecc": U_ecc}
        elif argument == "num_set":
            U_numset = self.calculate_U_num_set(generated_texts, question_context)
            return {"U_numset": U_numset}
        else:
            U_eigv = self.calculate_U_eigv(generated_texts, question_context)
            U_deg = self.calculate_U_deg(generated_texts, question_context)
            U_ecc = self.calculate_U_ecc(generated_texts, question_context)
            U_numset = self.calculate_U_num_set(generated_texts, question_context)
            return {"U_eigv": U_eigv, "U_deg": U_deg, "U_ecc": U_ecc,"U_numset": U_numset}

    def completion_forward(self, model: str, input_text: str, generated_text: str, question_context:str, argument: str = "all", **kwargs):
        if model not in PROB_AVAILABLE_API_MODELS:
            raise ValueError("This method is not applicable to given model")

        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        for _ in range(self.number_of_generations):
            kwargs.pop('logprobs', None)
            seed = kwargs.pop('seed', None) 
            seed = random.randint(0, 1000000)
            kwargs['seed'] = seed
            
            response = completion(
                model=model,
                messages=[{"content": input_text, "role": "user"}],
                logprobs=True,
                **kwargs
            )

            logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']]
            tokens = [token['token'] for token in response.choices[0].logprobs['content']]
            generated_texts.append(response.choices[0].message['content'])

        if argument == "eigv":
            U_eigv = self.calculate_U_eigv(generated_texts, question_context)
            return {"U_eigv": U_eigv}
        elif argument == "deg":
            U_deg = self.calculate_U_deg(generated_texts, question_context)
            return {"U_deg": U_deg}
        elif argument == "ecc":
            U_ecc = self.calculate_U_ecc(generated_texts, question_context)
            return {"U_ecc": U_ecc}
        elif argument == "num_set":
            U_numset = self.calculate_U_num_set(generated_texts, question_context)
            return {"U_numset": U_numset}
        else:
            U_eigv = self.calculate_U_eigv(generated_texts, question_context)
            U_deg = self.calculate_U_deg(generated_texts, question_context)
            U_ecc = self.calculate_U_ecc(generated_texts, question_context)
            U_numset = self.calculate_U_num_set(generated_texts, question_context)
            return {"U_eigv": U_eigv, "U_deg": U_deg, "U_ecc": U_ecc,"U_numset": U_numset}

    
    def __str__(self):
        return (
            f"BlackBoxMethods with {self.number_of_generations} generations. "
            f"Model for checking semantic: {self.model_for_entailment.__class__.__name__} "
            f"(ID: {self.model_for_entailment.config._name_or_path}). "
            f"Tokenizer for checking semantic: {self.tokenizer_for_entailment.__class__.__name__} "
            f"(ID: {self.tokenizer_for_entailment.pretrained_vocab_files_map['tokenizer_file']}). "
            f"Threshold: {self.threshold}. Standard Deviation: {self.std}"
            f"Method for similarity: {self.method_for_similarity}"
        )



