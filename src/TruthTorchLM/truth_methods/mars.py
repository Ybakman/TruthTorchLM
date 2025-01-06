from .truth_method import TruthMethod
from TruthTorchLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from TruthTorchLM.utils import sigmoid_normalization
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification

from .truth_method import TruthMethod

from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
from scipy.special import softmax
import torch
import numpy as np
import re


class MARS(TruthMethod):

    REQUIRES_LOGPROBS = True

    def __init__(self, mars_temperature:float = 0.1, mars_model=None, mars_tokenizer = None, device = 'cuda'):
        super().__init__()
        self.mars_temperature = mars_temperature
        if mars_model is None:
            #load from HF
            mars_model = AutoModelForTokenClassification.from_pretrained("duygunuryldz/MARS").to(device)
            mars_tokenizer = AutoTokenizer.from_pretrained("duygunuryldz/MARS") 

        self.mars_model = mars_model
        self.mars_tokenizer = mars_tokenizer

    
    def get_importance_vector_MARS(self, model, tokenizer, question, answer):
        words = re.findall(r'\w+|[^\w\s]', answer)
        #first tokenize
        tokenized_input = tokenizer.encode_plus(
                                                [question],
                                                words,
                                                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                return_token_type_ids=True,
                                                is_split_into_words=True,
                                                truncation=True,
                                                max_length = 512,           # Pad & truncate all sentences.
                                                )
        attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(model.device)
        input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(model.device)

        unk_token_id = tokenizer.unk_token_id

        if unk_token_id in input_ids:
            print(words)
            return np.zeros(len(words)), words

        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(model.device)
        word_ids = tokenized_input.word_ids() 

        logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits[0].cpu()
        classes = logits[:,0:2]
        scores = torch.nn.functional.sigmoid(logits[:,2])

        phrases = []
        importance_scores = []
        i = 0
        while(i<len(scores)):
            if word_ids[i] == None or token_type_ids[0][i] == 0:
                i += 1 
                continue
            cl = torch.argmax(classes[i,:])
            if word_ids[i] == 0 or cl == 0: #we handle the edge case as well (beginning of the sentence)
                for j in range(i+1, len(scores)):
                    cl = torch.argmax(classes[j,:])
                    continue_word = False
                    for k in range(i,j):
                        if word_ids[k] == word_ids[j]:
                            continue_word = True
                    if (cl == 0 or  word_ids[j] == None) and continue_word == False:
                        break
                phrases.append(tokenizer.decode(input_ids[0][i:j]))
                importance_scores.append(scores[i].item())
                i = j 

        #maybe modify phrase with actual sentence
        real_phrases = []
        phrase_ind  = 0
        i = 0
        answer = answer.strip()

        while(i < len(answer)):
            last_token_place  = -1
            for j in range(i+1, len(answer)+1):
                if  phrases[phrase_ind].strip().replace(" ", "") == tokenizer.decode(tokenizer.encode(answer[i:j])[1:-1]).strip().replace(" ", ""):
                    last_token_place = j

            real_phrases.append(answer[i:last_token_place ].strip())
            i = last_token_place
            phrase_ind += 1

            if i < len(answer) and phrase_ind == len(phrases):
                print('Error in importance vector')
                print(phrases)
                print(answer)
                return np.ones(len(words))/ len(words) , words
            
        return np.array(importance_scores), real_phrases

    def compute_token_nll_importance_phrase(self, probs, tokens, importance_vector, phrases):#remove special tokens from tokens!
        neg_log_likelihoods = -np.log(probs) 
        #find probabilities of each word
        neg_log_likelihoods_word = []
        token_idx = 0
        merged_importance_vector  = []
        i= 0
        while i < len(phrases):
            found = False
            while found == False:
                for k in range(1,len(phrases)-i+1):
                    word  = "".join(phrases[i:i+k])
                    last_token = -1
                    for j in range(token_idx+1, len(tokens)+1):#importance should be summed I guess
                        #print("".join(tokens[token_idx:j]).strip().replace(" ", "").replace("\n", "").lower(), word.strip().replace(" ", "").lower())
                        if "".join(tokens[token_idx:j]).strip().replace(" ", "").replace("\n", "").lower() == word.strip().replace(" ", "").lower():
                            last_token = j
                        
                    if last_token != -1:
                        neg_log_likelihoods_word.append(np.mean(neg_log_likelihoods[token_idx:last_token]))
                        merged_importance_vector.append(np.mean(importance_vector[i:i+k]))
                        found = True
                        i += k
                        token_idx = last_token 
                        break
                if found == False and k == len(phrases)-i:
                    print('Error')
                    print(phrases)
                    print(tokens)
                    return -np.mean(neg_log_likelihoods), np.ones(len(neg_log_likelihoods))/ len(neg_log_likelihoods)

        merged_importance_vector = np.array(merged_importance_vector) / self.mars_temperature
        merged_importance_vector = softmax(merged_importance_vector, axis=0)
        score = 0.5 * np.dot( merged_importance_vector, neg_log_likelihoods_word) + 0.5 * np.mean(neg_log_likelihoods )
        return -score, merged_importance_vector



    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        if type(tokenizer.eos_token_id) == list:
            if all_ids[0,-1] in tokenizer.eos_token_id: 
                model_output = all_ids[:,:-1]
            else:
                print('exeeded max length')
                model_output = all_ids
        else:
            if all_ids[0,-1] == tokenizer.eos_token_id:
                model_output = all_ids[:,:-1]
            else:
                print('exeeded max length')
                model_output = all_ids
        #model_output = all_ids
        tokens = model_output[0][len(input_ids[0]):]
        tokens_text = [tokenizer.decode(token) for token in tokens]
        generated_text = tokenizer.decode(tokens, skip_special_tokens = False)

        with torch.no_grad():
            outputs = model(model_output)
            logits = outputs.logits  # Logits for each token in the input

            # Calculate probabilities from logits
            probs = torch.softmax(logits, dim=-1)#probs for each token
            probs = probs[0, len(input_ids[0])-1:-1, :]#logprobs for each token in the generated text
            probs = torch.gather(probs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
            probs = probs.view(-1).tolist()#convert to list
            probs = np.array(probs)
        
        importance_scores, phrases = self.get_importance_vector_MARS(self.mars_model, self.mars_tokenizer, question_context, generated_text)
        score, merged_importance_vector = self.compute_token_nll_importance_phrase(probs, tokens_text, importance_scores, phrases)
        
                
        return {"truth_value": score,  "generated_text": generated_text, 'phrases': phrases, 'importance_scores': importance_scores, 'probs': probs, 'merged_importance_vector': merged_importance_vector}
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):

        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("MARS method is not applicable to given model")

        probs = np.exp(np.array(logprobs))

        importance_scores, phrases = self.get_importance_vector_MARS(self.mars_model, self.mars_tokenizer, question_context, generated_text)
        score, merged_importance_vector = self.compute_token_nll_importance_phrase(probs, generated_tokens, importance_scores, phrases)
                
        return {"truth_value": score,  "generated_text": generated_text, 'phrases': phrases, 'importance_scores': importance_scores, 'probs': probs, 'merged_importance_vector': merged_importance_vector}
    
