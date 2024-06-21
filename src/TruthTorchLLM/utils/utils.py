import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

def sigmoid_normalization(x: float, threshold: float = 0.0, std: float = 1.0):
    return 1 / (1 + np.exp(- (x - threshold) / std))

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
    entailment_prob = probs[0][2]  # entailment class is the third class in Deberta-large-mnli
    return entailment_prob > 0.5

# Function to perform bidirectional entailment clustering
def bidirectional_entailment_clustering(model_for_entailment: PreTrainedModel, tokenizer_for_entailment: PreTrainedTokenizer, context : str, sequences: list[str]):
    clusters = [{sequences[0]}]
    for s_m in sequences[1:]:
        added_to_class = False
        for c in clusters:
            s_c = next(iter(c))  # Use the first sequence in the class for comparison
            left = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_c, s_m)
            right = check_entailment(model_for_entailment, tokenizer_for_entailment, context, s_m, s_c)
            
            if left and right:
                c.add(s_m)
                added_to_class = True
                break
        
        if not added_to_class:
            clusters.append({s_m})
    
    return clusters
