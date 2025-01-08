from .decomposition_method import FactualDecompositionMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from typing import Callable
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat

from copy import deepcopy

CHAT = [{"role": "system", "content": 'You are a helpful assistant. List the specific factual propositions included in the given input. Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a separate bullet point, without adding explanations, introductions, or conversational responses. Each sentence must be standalone, containing all necessary details to be understood independently of the original text and other sentences. This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. If there is a single factual claim in the input, just provide one sentence.'},
        {"role": "user", "content": '''{TEXT}'''}]

def default_output_parser(text:str):
        statements = text.split("\n•")
        statements = [statement.strip() for statement in statements if statement.strip()]
        return statements

class UnstructuredDecompositionLocal(FactualDecompositionMethod):
    def __init__(self, model:PreTrainedModel, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], instruction:list=CHAT, decomposition_depth:int=1, 
                 output_parser:Callable[[str],list[str]]=default_output_parser, add_generation_prompt = True, continue_final_message = False, **kwargs):
        super().__init__()
    
        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.output_parser = output_parser
        self.decomposition_depth = decomposition_depth
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message

        default_kwargs = {"top_p":1, 
                          "do_sample" : False,
                          "temperature": None}
        default_kwargs.update(kwargs)

        default_kwargs.pop('seed', None) 
        eos_token_id = default_kwargs.pop("eos_token_id", None)
        if eos_token_id is None:    
            eos_token_id = model.config.eos_token_id
        default_kwargs['eos_token_id'] = eos_token_id

        pad_token_id = default_kwargs.pop("pad_token_id", None)
        if pad_token_id is None:
            if type(eos_token_id) == list:
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id
        default_kwargs['pad_token_id'] = pad_token_id 
        print(default_kwargs)
        self.kwargs = default_kwargs

    def decompose_facts(self, input_text:str):

        messages = deepcopy(self.instruction)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)
        self.tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=self.add_generation_prompt, continue_final_message=self.continue_final_message)
        generated_output = generate(text, self.model, self.tokenizer, **self.kwargs)
        generated_text = "\n" + generated_output["generated_text_skip_specials"].strip()
        statements = self.output_parser(generated_text)

        return statements
        
    def __str__(self):
        return "Factual decomposition by using LLMs method with " + self.model + " model. Chat template is:\n" +  str(self.instruction) + "\n Sentence seperator is: " + self.sentence_seperator
