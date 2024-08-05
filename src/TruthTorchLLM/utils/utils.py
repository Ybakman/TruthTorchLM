import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import re
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer

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