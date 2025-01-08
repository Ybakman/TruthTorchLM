from .decomposition_method import FactualDecompositionMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from typing import Callable
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat

from copy import deepcopy
import outlines
from pydantic import BaseModel

CHAT = [{"role": "system", "content": 'You are a helpful assistant. List the specific factual propositions included in the given input. Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a separate bullet point, without adding explanations, introductions, or conversational responses. Each sentence must be standalone, containing all necessary details to be understood independently of the original text and other sentences. This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. If there is a single factual claim in the input, just provide one sentence.'},
        {"role": "user", "content": '''{TEXT}'''}]

class Statements(BaseModel):
    statements: list

class StructuredDecompositionLocal(FactualDecompositionMethod):
    def __init__(self, model:PreTrainedModel, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], instruction:list=CHAT, decomposition_depth:int=1, 
                 add_generation_prompt = True, continue_final_message = False):
        super().__init__()
    
        outlines.disable_cache() ##TODO where to put this?
        outlines_model = outlines.models.Transformers(model, tokenizer)
        self.generator = outlines.generate.json(outlines_model, Statements)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.decomposition_depth = decomposition_depth
        self.add_generation_prompt = add_generation_prompt
        self.continue_final_message = continue_final_message
    
    def decompose_facts(self, input_text:str):

        messages = deepcopy(self.instruction)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)
        self.tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=self.add_generation_prompt, continue_final_message=self.continue_final_message)
        resp = self.generator(text)

        return resp.statements
        
    def __str__(self):
        return "Factual decomposition by using LLMs.\nModel: " + self.model + "\nOutput structure is enforced with 'outlines' library.\nChat template is:\n" +  str(self.instruction) 
