import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import re
from sklearn.metrics import jaccard_score
from scipy.linalg import eigh
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from sklearn.metrics import precision_recall_curve

#logging.set_verbosity(40)


def find_threshold_std(correctness: list, truth_values: list, precision: float = -1, recall: float = -1):
    std = np.std(truth_values)
    precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
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

    return threshold, std




def sigmoid_normalization(x: float, threshold: float = 0.0, std: float = 1.0):
    return 1 / (1 + np.exp(- (x - threshold) / std))

#for a target text, find the indices of the tokens that are in the target text. 
#If target text cannot be tokenized in the original form, return the indices of the tokens that contain the target text and has the shortest length
def find_token_indices(tokens:list, tokenizer:PreTrainedTokenizer, target_text:str, ):
    indices = []
    texts = []
    begin = 0
    found = False
    while begin < len(tokens):
        for end in range(begin+1, len(tokens)):
            if  target_text in tokenizer.decode(tokens[begin:end]):
                #move begin
                while target_text in tokenizer.decode(tokens[begin:end]):
                    begin += 1
                begin -= 1
                index_list = [i for i in range(begin, end)]
                indices.append(index_list)
                texts.append(tokenizer.decode(tokens[begin:end]))
                begin = end
                found = True
                break
        if not found:
            break
        else:
            found = False
    return indices, texts


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    vectorizer = CountVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2]).toarray()
    
    intersection = (vectors[0] & vectors[1]).sum()
    union = (vectors[0] | vectors[1]).sum()
    print(intersection)
    print(union)
    return intersection / union if union != 0 else 0



# Function to check entailment between two sequences
def check_entailment(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str):
    inputs = tokenizer_for_entailment.encode_plus(
        text=context + " " + seq1,
        text_pair=context + " " + seq2,
        return_tensors='pt',
        truncation=True
    )
    outputs = model_for_entailment(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    out_class = torch.argmax(probs[0], dim=-1).item()
    
    return out_class




# Function to perform bidirectional entailment clustering
def bidirectional_entailment_clustering(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context : str, sequences: list[str],method = "semantic"):
    clusters = [{sequences[0]}]
    for s_m in sequences[1:]:
        added_to_class = False
        for c in clusters:
            s_c = next(iter(c))  # Use the first sequence in the class for comparison
            if method == "semantic":
                left = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_c, s_m)
                right = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_m, s_c)
                
                if left != 0 and right != 0:#it shows there is no contradiction
                    c.add(s_m)
                    added_to_class = True
                    break
            elif method == "jaccard":
                similarity = calculate_jaccard_similarity(s_c, s_m)
                if similarity >= 0.7:
                    c.add(s_m)
                    added_to_class = True
                    break
        
        if not added_to_class:
            clusters.append({s_m})
    
    return clusters


def entailment_probability(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str, mode='minus_contradiction', temperature: float = 3.0):
    inputs = tokenizer_for_entailment.encode_plus(
        text=context + " " + seq1,
        text_pair=context + " " + seq2,
        return_tensors='pt',
        truncation=True
    )
    outputs = model_for_entailment(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits/temperature, dim=-1) # contradiction, neutral, entailment
    if mode == 'minus_contradiction':
        return 1 - probs[0][0]
    elif mode == 'entailment':
        return probs[0][2]
   

def get_D_mat(W):
    D = np.diag(np.sum(W, axis=1))
    return D

def get_L_mat(W, symmetric=True):
    # Compute the normalized Laplacian matrix from the degree matrix and weighted adjacency matrix
    D = get_D_mat(W)
    if symmetric:
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError()
        #L = np.linalg.inv(D) @ (D - W)
    return L.copy()

def calculate_affinity_matrix(texts: list[str], context: str, method_for_similarity: str = 'semantic', model_for_entailment: PreTrainedModel = None,
                 tokenizer_for_entailment: PreTrainedTokenizer = None, temperature: float = 3.0):
    
    if model_for_entailment is None or tokenizer_for_entailment is None:
        model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
        tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

    n = len(texts)
    affinity_matrix = np.ones((n, n))
    
    if method_for_similarity == "semantic":
        for i in range(n):
            for j in range(i + 1, n):
                left = entailment_probability(model_for_entailment, tokenizer_for_entailment, context, texts[i], texts[j], temperature=temperature).item()
                right = entailment_probability(model_for_entailment, tokenizer_for_entailment, context, texts[j], texts[i], temperature=temperature).item()
                affinity_matrix[i][j] = affinity_matrix[j][i] = (left + right) / 2
    elif method_for_similarity  == "jaccard":
        vectorizer = CountVectorizer().fit_transform(texts)
        vectors = vectorizer.toarray()
        for i in range(n):
            for j in range(i + 1, n):
                affinity_matrix[i][j] = affinity_matrix[j][i] = jaccard_score(vectors[i], vectors[j], average='macro')
    
    return affinity_matrix


def get_eig(L, thres=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def calculate_U_eigv(texts: list[str], context:str, temperature: float = 3.0):
    W = calculate_affinity_matrix(texts, context, temperature=temperature)
    L = get_L_mat(W)
    eigvals = np.linalg.eigvalsh(L)
    U_eigv = sum(max(0, 1 - eig) for eig in eigvals)
    return U_eigv

def calculate_U_deg(texts: list[str], context: str, temperature: float = 3.0):
    W = calculate_affinity_matrix(texts, context, temperature=temperature)
    D = get_D_mat(W)
    m = len(W)
    U_deg = np.trace(m * np.identity(m) - D) / (m ** 2)
    return U_deg

def calculate_U_ecc(texts: list[str], context: str, temperature: float = 3.0, eigen_threshold: float = 0.9):
    W = calculate_affinity_matrix(texts, context)
    L = get_L_mat(W, symmetric=True)
    eigvals, eigvecs = get_eig(L, thres=eigen_threshold)
    V = eigvecs
    V_mean = np.mean(V, axis=0)
    V_prime = V - V_mean
    
    U_ecc = np.linalg.norm(V_prime, axis=1).sum()
    return U_ecc

def calculate_U_num_set(texts: list[str], context: str,method_for_similarity: str = 'semantic', model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    
    if model_for_entailment is None or tokenizer_for_entailment is None:
        model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli')
        tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

    clusters = bidirectional_entailment_clustering(model_for_entailment, tokenizer_for_entailment, context, texts, method_for_similarity)
    return len(clusters)
