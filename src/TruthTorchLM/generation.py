
import copy
import torch
import random
from typing import Union
from litellm import completion

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

#from .truth_methods.truth_method import TruthMethod
from TruthTorchLM.availability import AVAILABLE_API_MODELS, PROB_AVAILABLE_API_MODELS
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat

import time


def generate_with_truth_value(model:Union[PreTrainedModel, str], messages:list, question_context:str = None, truth_methods: list = [], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,  
generation_seed=None, batch_generation=True, add_generation_prompt = True, continue_final_message = False,  **kwargs)-> dict:
    if type(model) == str:
        return generate_with_truth_value_api(model = model, messages = messages, question_context = question_context, truth_methods = truth_methods, generation_seed = generation_seed, **kwargs)
    else:
        return generate_with_truth_value_hf_local(model = model, messages = messages, question_context = question_context, truth_methods = truth_methods, 
        tokenizer = tokenizer, generation_seed = generation_seed, batch_generation=batch_generation, add_generation_prompt = add_generation_prompt, continue_final_message = continue_final_message, **kwargs)



            
#TODO: remove number of generations from kwargs if exists
def generate_with_truth_value_hf_local(model:PreTrainedModel, messages:list, question_context:str = None, truth_methods: list = [], 
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed=None, batch_generation=True, add_generation_prompt = True, continue_final_message = False,  **kwargs) -> dict:


    tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message)
    if question_context == None:
        question_context = ''
        #search over last user message if exists
        for message in messages[::-1]:
            if message['role'] == 'user':
                question_context = message['content']
                break

    generated_output = generate(text, model, tokenizer, **kwargs)
    generated_text_return = generated_output['generated_text_skip_specials']
    generated_text = generated_output['generated_text']
    tokens = generated_output['tokens']
    model_output = generated_output['all_ids']

    #Get sampled generations to be used in truth methods
    number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)

    sampled_gen_dict = sample_generations_hf_local(model, text, tokenizer, generation_seed, number_of_generations=number_of_generations, 
    return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs,return_attentions=return_attentions, return_activations=return_activations, batch_generation=batch_generation,  **kwargs)


    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method(model=model,  input_text=text, generated_text=generated_text, question_context=question_context, all_ids=model_output, tokenizer=tokenizer, generation_seed = generation_seed, sampled_generations_dict=sampled_gen_dict, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)


    # 'all_ids': model_output.cpu(), 'generated_tokens':tokens
    # Create TruthObject
    truth_dict = {'generated_text':generated_text_return, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs, 'all_ids': model_output.cpu(), 'generated_tokens':tokens}

    # Return TruthObject
    return truth_dict


#for api-based models, we should write a wrapper function to handle exceptions during the api call
def generate_with_truth_value_api(model:str, messages:list, question_context:str = None, truth_methods: list = [], generation_seed=None, **kwargs) -> dict:
    # Check if the model is an API model
    if generation_seed is not None:
        random.seed(generation_seed)

    if type(model) == str and not model in AVAILABLE_API_MODELS:
        raise ValueError(f"model {model} is not supported.")

    requires_logprobs = False    
    for truth_method in truth_methods:
        if truth_method.REQUIRES_LOGPROBS:
            requires_logprobs = True

    if requires_logprobs and not model in PROB_AVAILABLE_API_MODELS:
        raise ValueError(f"model {model} is not supported for probability requiring truth methods.")
    
    if question_context == None:
        question_context = ''
        #search over last user message if exists
        for message in messages[::-1]:
            if message['role'] == 'user':
                question_context = message['content']
                break

    # Generate the main output
    seed = kwargs.pop('seed', None)
    if seed == None:
        seed = random.randint(0, 1000000)
    kwargs['seed'] = seed #a random seed is generated if seed is not specified

    response = completion(
        model=model,
        messages=messages,
        logprobs = requires_logprobs,
        **kwargs
    )
    generated_text = response.choices[0].message['content']

    logprobs = [token['logprob'] for token in response.choices[0].logprobs['content']] if requires_logprobs else None
    generated_tokens = [token['token'] for token in response.choices[0].logprobs['content']] if requires_logprobs else None
    
    #Get sampled generations to be used in truth methods
    number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)
    sampled_gen_dict = sample_generations_api(model, messages, generation_seed, number_of_generations=number_of_generations, 
    return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs,return_attentions=return_attentions, return_activations=return_activations, **kwargs)
      
    # Get scores from all truth methods
    normalized_truth_values = []
    unnormalized_truth_values = []
    method_spec_outputs = []
    
    for truth_method in truth_methods:
        truth_values = truth_method(model=model, messages=messages, generated_text=generated_text, question_context=question_context, generation_seed=generation_seed, sampled_generations_dict=sampled_gen_dict, logprobs=logprobs, generated_tokens=generated_tokens, **kwargs)
        normalized_truth_values.append(truth_values['normalized_truth_value'])
        unnormalized_truth_values.append(truth_values['truth_value'])
        method_spec_outputs.append(truth_values)

    # Create TruthObject
    truth_dict = {'generated_text':generated_text, 'normalized_truth_values':normalized_truth_values, 'unnormalized_truth_values':unnormalized_truth_values, 'method_specific_outputs' : method_spec_outputs}

    # Return TruthObject
    return truth_dict


def get_sampling_properties(truth_methods:list):
    number_of_generations = 0
    return_text = False
    return_logits = False
    return_logprobs = False
    return_attentions = False
    return_activations = False
    #search over all truth methods for number of generations
    for truth_method in truth_methods:
        if hasattr(truth_method, 'number_of_generations') and truth_method.number_of_generations > number_of_generations:
            number_of_generations = truth_method.number_of_generations
        if truth_method.REQUIRES_SAMPLED_TEXT:
            return_text = True
        if truth_method.REQUIRES_SAMPLED_LOGITS:
            return_logits = True
        if truth_method.REQUIRES_SAMPLED_LOGPROBS:
            return_logprobs = True
        if truth_method.REQUIRES_SAMPLED_ATTENTIONS:
            return_attentions = True
        if truth_method.REQUIRES_SAMPLED_ACTIVATIONS:
            return_activations = True
    return number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations


def sample_generations_hf_local(model:PreTrainedModel, input_text:str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], generation_seed:int=None, 
number_of_generations:int = 0, return_text:bool = False, return_logits:bool = False, return_logprobs:bool = False, return_attentions:bool = False, return_activations:bool = False, batch_generation = False,  **kwargs):

    if number_of_generations == 0 or (not return_text and not return_logprobs and not return_activations and not return_attentions and not return_logits):
        return None
    
    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        random.seed(generation_seed)
    
    if batch_generation == True:
        return sample_generations_batch_hf_local(model=model, input_text=input_text, tokenizer=tokenizer, number_of_generations=number_of_generations, 
        return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs, 
        return_attentions=return_attentions, return_activations=return_activations, **kwargs)
    if batch_generation == False:
        return sample_generations_sequential_hf_local(model=model, input_text=input_text, tokenizer=tokenizer, number_of_generations=number_of_generations, 
        return_text=return_text, return_logits=return_logits, return_logprobs=return_logprobs, 
        return_attentions=return_attentions, return_activations=return_activations, **kwargs)
        



def sample_generations_batch_hf_local(model:PreTrainedModel, input_text:str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
number_of_generations:int = 0, return_text:bool = False, return_logits:bool = False, return_logprobs:bool = False, return_attentions:bool = False, return_activations:bool = False, return_model_output:bool = True, **kwargs):

    #number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)

    if number_of_generations == 0 or (not return_text and not return_logprobs and not return_activations and not return_attentions and not return_logits):
        return None
        
    kwargs = copy.deepcopy(kwargs)
    kwargs.pop('do_sample', None)
    kwargs.pop('num_return_sequences', None)
    kwargs.pop('return_dict_in_generate', None)
    kwargs.pop('output_attentions', None)
    kwargs.pop('output_hidden_states', None)
    kwargs.pop('output_logits', None)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']

    eos_token_id = kwargs.pop("eos_token_id", None)

    if eos_token_id is None:    
        eos_token_id = model.config.eos_token_id

    pad_token_id = kwargs.pop("pad_token_id", None)
    if pad_token_id is None:
        if type(eos_token_id) == list:
            pad_token_id = eos_token_id[0]
        else:
            pad_token_id = eos_token_id

    generated_texts = []
    logits_list = []
    logprobs = []
    attentions_list = []
    activations_list = []
    tokens = []

    with torch.no_grad():
        model_output = model.generate(**inputs, num_return_sequences=number_of_generations, do_sample=True, return_dict_in_generate=True, 
                                        output_attentions=return_attentions, output_hidden_states=return_activations, output_logits=(return_logits or return_logprobs), 
                                        eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
        
        model_output.past_key_values=None
        model_output.sequences = model_output.sequences.cpu()
        if type(eos_token_id) == list:
            temp = torch.stack([torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos).to(dtype=torch.int), dim=-1) for eos in eos_token_id]).T
            indices = [torch.min(temp[i][temp[i]>0]).item() for i in range(len(temp))]
        else:
            indices = torch.argmax((model_output.sequences[:, len(input_ids[0]):] == eos_token_id).to(dtype=torch.int), dim=-1)
        indices[indices==0] = model_output.sequences.shape[1] - len(input_ids[0]) -1
        if return_text:
            tokens = [seq[len(input_ids[0]):indices[i] + len(input_ids[0])+1].tolist() for i, seq in enumerate(model_output.sequences)]
            generated_texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        if return_logprobs or return_logits:
            logits_list = torch.stack(model_output.logits).cpu().permute(1, 0, 2)
            model_output.logits = None
            if return_logprobs:
                logprobs = torch.log_softmax(logits_list, dim=-1) #logprobs for each token
                logprobs = torch.gather(logprobs, dim=-1, index = model_output.sequences[:, len(input_ids[0]):].unsqueeze(-1))#logprobs for each token in the generated text
                logprobs = logprobs.squeeze(-1).tolist()#convert to list
                logprobs = [logprobs[i][:indices[i]+1] for i in range(len(logprobs))]
            if return_logits:
                logits_list = [logits_list[i][:indices[i]+1] for i in range(len(logits_list))]
            else:
                logits_list = []
        if return_activations:
            activations_list = [] #shape = (num gen, num token, num_layer, hidden_state_shape)
            for i in range(number_of_generations): #generation id
                acts = []
                for j in range(indices[i]+1): #token id 
                    act = []
                    for k in range(len(model_output.hidden_states[0])): #layer id
                        act.append(model_output.hidden_states[j][k][i].cpu())
                    acts.append(act)
                activations_list.append(acts)
            model_output.hidden_states = None
        if return_attentions:
            attentions_list = model_output.attentions
            for i in range(number_of_generations): #generation id
                atts = []
                for j in range(indices[i]+1): #token id 
                    att = []
                    for k in range(len(model_output.attentions[0])): #layer id
                        att.append(model_output.attentions[j][k][i].cpu())
                    atts.append(att)
                attentions_list.append(atts)
            model_output.attentions = None
        
        if not return_model_output:
            model_output.sequences = None

    return {"generated_texts": generated_texts, "logprobs": logprobs, "activations": activations_list, "logits":logits_list, "attentions":attentions_list, "model_outputs": model_output.sequences, "tokens":tokens}
         


def sample_generations_sequential_hf_local(model:PreTrainedModel, input_text:str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
 number_of_generations:int = 0, do_sample:bool=True, return_text:bool = False, return_logits:bool = False, return_logprobs:bool = False, return_attentions:bool = False, return_activations:bool = False, return_model_output:bool = True, **kwargs):
    
    kwargs = copy.deepcopy(kwargs)
    kwargs.pop('do_sample', None)
    kwargs.pop('num_return_sequences', None)
    kwargs.pop('return_dict_in_generate', None)
    kwargs.pop('output_attentions', None)
    kwargs.pop('output_hidden_states', None)
    kwargs.pop('output_logits', None)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']

    eos_token_id = kwargs.pop("eos_token_id", None)

    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id

    generated_texts = []
    logits_list = []
    logprobs_list = []
    attentions_list = []
    activations_list = []
    model_outputs = []
    token_lists = []
    for i in range(number_of_generations):
        with torch.no_grad():
            model_output = model.generate(**inputs, num_return_sequences=1, do_sample=do_sample, return_dict_in_generate=True, 
                                            output_attentions=return_attentions, output_hidden_states=return_activations, 
                                            output_logits=(return_logits or return_logprobs), eos_token_id=eos_token_id, **kwargs)
            model_output.past_key_values=None
            model_output.sequences = model_output.sequences.cpu()
            if return_model_output:
                model_outputs.append(model_output.sequences)
            if return_text:
                tokens = model_output.sequences[0][len(input_ids[0]):]
                generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
                token_lists.append(tokens.tolist())
            if return_logprobs or return_logits:
                logits = torch.cat(model_output.logits).cpu()
                model_output.logits=None
                if return_logprobs:
                    logprobs = torch.log_softmax(logits, dim=-1)#logprobs for each token
                    logprobs = torch.gather(logprobs, dim=1, index = model_output.sequences[0][len(input_ids[0]):].view(-1, 1))#logprobs for each token in the generated text
                    logprobs = logprobs.view(-1).tolist()#convert to list
                    logprobs_list.append(logprobs)
                if return_logits:
                    logits_list.append(logits)
            if return_activations:
                acts = []
                for i in range(len(model_output.hidden_states)):
                    act = []
                    for j in range(len(model_output.hidden_states[i])):
                        act.append(model_output.hidden_states[i][j][0].cpu())
                    acts.append(act)
                activations_list.append(acts)
                model_output.hidden_states = None
            if return_attentions:
                atts = []
                for i in range(len(model_output.attentions)):
                    att = []
                    for j in range(len(model_output.attentions[i])):
                        att.append(model_output.attentions[i][j][0].cpu())
                    atts.append(att)
                attentions_list.append(atts)
                model_output.attentions = None

    return {"generated_texts": generated_texts, "logprobs": logprobs_list, "activations": activations_list, "logits":logits_list, "attentions":attentions_list, "model_outputs": model_outputs, "tokens":token_lists}
            


def sample_generations_api(model:str, messages:list, generation_seed:int=None, 
number_of_generations:int = 0, return_text:bool = False, return_logits:bool = False, return_logprobs:bool = False, return_attentions:bool = False, return_activations:bool = False, **kwargs):
    #number_of_generations, return_text, return_logits, return_logprobs, return_attentions, return_activations = get_sampling_properties(truth_methods)

    if number_of_generations == 0 or (not return_text and not return_logprobs):
        return None
    

    if generation_seed is not None:
        random.seed(generation_seed)
    
    kwargs = copy.deepcopy(kwargs)
    
    generated_texts = []
    logprobs_list = []
    token_lists = []
    for i in range(number_of_generations):
        kwargs.pop('logprobs', None)
        seed = kwargs.pop('seed', None) 
        seed = random.randint(0, 1000000)
        kwargs['seed'] = seed
            
        response = completion(
            model=model,
            messages=messages,
            logprobs=return_logprobs,
            **kwargs
        )
        
        if return_text:
            generated_texts.append(response.choices[0].message['content'])
        if return_logprobs:
            logprobs_list.append([token['logprob'] for token in response.choices[0].logprobs['content']])
            token_lists.append([token['token'] for token in response.choices[0].logprobs['content']])
     
    return {"generated_texts": generated_texts, "logprobs": logprobs_list, "tokens":token_lists}