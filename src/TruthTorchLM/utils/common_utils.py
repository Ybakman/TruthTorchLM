import numpy as np
import torch
import copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import re
from sklearn.metrics import jaccard_score
from scipy.linalg import eigh
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DebertaForSequenceClassification, DebertaTokenizer

from litellm import completion
from TruthTorchLM.templates import ENTAILMENT_PROMPT, DEFAULT_SYSTEM_PROMPT 
from typing import Union


#logging.set_verbosity(40)


def check_system_prompt_support(tokenizer):
    chat = [{"role": "system", "content": 'Test'},]
    try:
        tokenizer(chat)
        return True
    except:
        return False

def fix_tokenizer_chat(tokenizer, chat):
    #tokenizer = copy.deepcopy(tokenizer)
    chat = copy.deepcopy(chat)
    if tokenizer.chat_template == None:
        tokenizer.chat_template ='''{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'system' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{ message['content'].strip() + '\n' }}
    {%- endif %}
{%- endfor %}'''.strip()
    else:
        if check_system_prompt_support(tokenizer) == False:
            #replace system prompt with the next user prompt
            for i in range(len(chat)):
                if chat[i]['role'] == 'system':
                    try:
                        if chat[i+1]['role'] == 'user':
                            chat[i]['role'] = 'user'
                            chat[i]['content'] = chat[i]['content'] + ' ' + chat[i+1]['content']
                            chat[i+1]['role'] = 'popped'
                        else:
                            chat[i]['role'] = 'user'
                        
                    except:
                        chat[i]['role'] = 'user'
            #remove popped elements
            chat = [chat[i] for i in range(len(chat)) if chat[i]['role'] != 'popped']
            
        
                        
    return tokenizer, chat


def generate(text:str, model:PreTrainedModel, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], **kwargs) -> dict:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model_output = model.generate(**inputs, **kwargs)
        tokens = model_output[0][len(inputs['input_ids'][0]):]
        generated_text = tokenizer.decode(tokens, skip_special_tokens = False)
        generated_text_return = tokenizer.decode(tokens, skip_special_tokens = True)

    return {'generated_text_skip_specials':generated_text_return, 'generated_text':generated_text, 'tokens':tokens, 'all_ids':model_output}


def sigmoid_normalization(x: float, threshold: float = 0.0, std: float = 1.0):
    z = (x - threshold) / std
    if z >= 0:
        # For positive z, compute sigmoid as 1 / (1 + exp(-z)) directly
        return 1 / (1 + np.exp(-z))
    else:
        # For negative z, to avoid overflow, use the identity: sigmoid(z) = exp(z) / (1 + exp(z))
        return np.exp(z) / (1 + np.exp(z))

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
    return intersection / union if union != 0 else 0



# Function to check entailment between two sequences
def check_entailment(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str):
    inputs = tokenizer_for_entailment.encode_plus(
        text=context + " " + seq1,
        text_pair=context + " " + seq2,
        return_tensors='pt',
        truncation=True,
        max_length=model_for_entailment.config.max_position_embeddings
    ).to(model_for_entailment.device)
    outputs = model_for_entailment(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    out_class = torch.argmax(probs[0], dim=-1).item()
    
    return out_class

def check_entailment_with_generation(model, seq1: str = None, seq2: str = None, context:str = None, entailment_prompt:str = ENTAILMENT_PROMPT, system_prompt:str = DEFAULT_SYSTEM_PROMPT,  tokenizer = None, max_new_tokens: int = 32):
    if system_prompt is None:#for some models there is no system prompt in their chat template such as gemma
        chat = [
        {"role": "user", "content": entailment_prompt.format(question = input_text)}]
    else:
        chat = [{"role": "system", "content": system_prompt},
        {"role": "user", "content": entailment_prompt.format(seq1=seq1, seq2=seq2, context=context)}]
    if type(model) != str:
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        generated_text = generate(input_text, model, tokenizer=tokenizer, do_sample = False, max_new_tokens = max_new_tokens)['generated_text_skip_specials']
    else:
        response = completion(
                model=model,
                messages= chat
            )
        generated_text = response.choices[0].message['content']
    if "same" in generated_text.lower():
        return 2
    elif "contradicted" in generated_text.lower():
        return 0
    else:
        return 1





def bidirectional_entailment_clustering(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context : str, sequences: list[str], method: str = "semantic", entailment_prompt:str = ENTAILMENT_PROMPT, system_prompt:str = DEFAULT_SYSTEM_PROMPT):
    clusters = [[sequences[0]]]
    for s_m in sequences[1:]:
        added_to_class = False
        for c in clusters:
            s_c = c[0]  # Use the first sequence in the class for comparison
            if method == "semantic":
                left = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_c, s_m)
                right = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_m, s_c)
                
                if left != 0 and right != 0:#it shows there is no contradiction
                    c.append(s_m)
                    added_to_class = True
                    break
            elif method == "jaccard":
                similarity = calculate_jaccard_similarity(s_c, s_m)
                if similarity >= 0.7:
                    c.append(s_m)
                    added_to_class = True
                    break
            elif method == "generation":
                check = check_entailment_with_generation(model_for_entailment, seq1=s_c, seq2=s_m, context=context, entailment_prompt=entailment_prompt, system_prompt=system_prompt, tokenizer=tokenizer_for_entailment)
                if check != 0:
                    c.append(s_m)
                    added_to_class = True
                    break
        if not added_to_class:
            clusters.append([s_m])
    
    return clusters

def entailment_probability(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str, mode='minus_contradiction', temperature: float = 3.0):
    inputs = tokenizer_for_entailment.encode_plus(
        text=context + " " + seq1,
        text_pair=context + " " + seq2,
        return_tensors='pt',
        truncation=True,
        max_length=model_for_entailment.config.max_position_embeddings
    ).to(model_for_entailment.device)

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


def check_entailment_one_hot(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context: str, seq1: str, seq2: str):
    out_class = check_entailment(model_for_entailment, tokenizer_for_entailment, context, seq1, seq2)
    one_hot = [0, 0, 0]
    if out_class == 2:      # Entailment
        one_hot[0] = 1
    elif out_class == 1:    # Neutral
        one_hot[1] = 1
    elif out_class == 0:    # Contradiction
        one_hot[2] = 1
    return one_hot

def calculate_affinity_matrix(texts: list[str], context: str, method_for_similarity: str = 'semantic', model_for_entailment: PreTrainedModel = None,
                 tokenizer_for_entailment: PreTrainedTokenizer = None, temperature: float = 3.0):
    
    if (model_for_entailment is None or tokenizer_for_entailment is None) and method_for_similarity == "semantic":
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

    elif method_for_similarity == "kernel":
        w = np.array([1, 0.5, 0])   # Pre-defined
        for i in range(n):
            for j in range(i + 1, n):
                left = check_entailment_one_hot(model_for_entailment, tokenizer_for_entailment, context, texts[i], texts[j])
                right = check_entailment_one_hot(model_for_entailment, tokenizer_for_entailment, context, texts[j], texts[i])
                affinity_matrix[i, j] = affinity_matrix[j, i] = np.dot(w, left) + np.dot(w, right)
            affinity_matrix[i,i] = 2
    
    return affinity_matrix


def get_eig(L, thres=None):
    # This function assumes L is symmetric
    # compute the eigenvalues and eigenvectors of the laplacian matrix
    eigvals, eigvecs = np.linalg.eigh(L)
    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

def calculate_U_eigv(texts: list[str], context:str, method_for_similarity:str = 'semantic', temperature: float = 3.0, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    W = calculate_affinity_matrix(texts, context, temperature=temperature, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment, method_for_similarity=method_for_similarity)
    L = get_L_mat(W)
    eigvals = np.linalg.eigvalsh(L)
    U_eigv = sum(max(0, 1 - eig) for eig in eigvals)
    return U_eigv

def calculate_U_deg(texts: list[str], context: str,  method_for_similarity:str = 'semantic',  temperature: float = 3.0, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    W = calculate_affinity_matrix(texts, context, temperature=temperature, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment, method_for_similarity=method_for_similarity)
    D = get_D_mat(W)
    m = len(W)
    U_deg = np.trace(m * np.identity(m) - D) / (m ** 2)
    return U_deg

def calculate_C_deg(texts: list[str], context: str, index:int = -1,  method_for_similarity:str = 'semantic',  temperature: float = 3.0, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    W = calculate_affinity_matrix(texts, context, temperature=temperature, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment, method_for_similarity=method_for_similarity)
    D = get_D_mat(W)
    m = len(W)
    C_deg = D[index, index]/m
    return C_deg

def calculate_U_ecc(texts: list[str], context: str,  method_for_similarity:str = 'semantic',  temperature: float = 3.0, eigen_threshold: float = 0.9, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    W = calculate_affinity_matrix(texts, context, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment , method_for_similarity=method_for_similarity, temperature=temperature)
    L = get_L_mat(W, symmetric=True)
    eigvals, eigvecs = get_eig(L, thres=eigen_threshold)
    V = eigvecs
    V_mean = np.mean(V, axis=0)
    V_prime = V - V_mean
    
    U_ecc = np.linalg.norm(V_prime, axis=1).sum()
    return U_ecc


def calculate_C_ecc(texts: list[str], context: str, index:int = -1,  method_for_similarity:str = 'semantic',  temperature: float = 3.0, eigen_threshold: float = 0.9, model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    W = calculate_affinity_matrix(texts, context, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment , method_for_similarity=method_for_similarity, temperature=temperature)
    L = get_L_mat(W, symmetric=True)
    eigvals, eigvecs = get_eig(L, thres=eigen_threshold)
    V = eigvecs
    V_mean = np.mean(V, axis=0)
    V_prime = V - V_mean
    
    C_ecc = np.linalg.norm(V_prime, axis=1)[index]
    return C_ecc


def calculate_U_num_set(texts: list[str], context: str ,method_for_similarity: str = 'semantic', model_for_entailment: PreTrainedModel = None, 
                 tokenizer_for_entailment: PreTrainedTokenizer = None):
    
    clusters = bidirectional_entailment_clustering(model_for_entailment, tokenizer_for_entailment, context, texts, method_for_similarity)
    return len(clusters)

def calculate_laplacian(graph: np.ndarray, normalize: bool):
    degree = get_D_mat(graph)
    laplacian = degree - graph
    if normalize: 
        degree_mp_inv = np.linalg.pinv(degree)
        sqrt_degree_inv = np.sqrt(degree_mp_inv)
        laplacian = sqrt_degree_inv @ laplacian @ sqrt_degree_inv
    return laplacian

def create_kernel(laplacian: np.ndarray, kernel_type='heat', temperature=0.3, smoothness=1, scale=1):
    # - laplacian: Graph laplacian
    # - kernel_type: 'heat' or 'matern'
    # - temperature: temperature paramter of Heat kernel
    # - smoothness: smoothness parameter of Matern kernel
    # - scale: scale parameter of Matern kernel
    
    if kernel_type == 'heat':
        kernel = np.exp(-temperature * laplacian)
    elif kernel_type == 'matern':
        identity_matrix = np.eye(laplacian.shape[0])
        kernel = np.linalg.inv((2 * smoothness / scale**2) * identity_matrix + laplacian) ** smoothness
    else:
        raise ValueError("Invalid kernel type. Choose 'heat' or 'matern'.")
    
    # Converting Kernel into unit trace PSD kernel
    diag_values = np.diag(kernel)
    normalization_factors = np.outer(np.sqrt(diag_values), np.sqrt(diag_values))
    N = kernel.shape[0]
    density_kernel = kernel / (normalization_factors * N)
    density_kernel = density_kernel / np.trace(density_kernel)
    return density_kernel

def calculate_VNE(kernel:np.ndarray):
    eigenvalues, _ = np.linalg.eigh(kernel)
    eigenvalues = eigenvalues[eigenvalues > 0]  # use only positive eigenvalues for log calculation
    vne = -np.sum(eigenvalues * np.log(eigenvalues))    # calculate von neumann entropy
    return vne
