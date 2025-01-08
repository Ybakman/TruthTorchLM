from .decomposition_method import FactualDecompositionMethod
from TruthTorchLM.availability import AVAILABLE_API_MODELS

from copy import deepcopy
from typing import Callable

import instructor
from litellm import completion
from pydantic import BaseModel


CHAT = [{"role": "system", "content": 'You are a helpful assistant. List the specific factual propositions included in the given input. Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a separate bullet point. Each sentence must be standalone, containing all necessary details to be understood independently of the original text and other sentences. This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. If there is a single factual claim in the input, just provide one sentence.'},
        {"role": "user", "content": '''{TEXT}'''}]


class Statements(BaseModel):
    statements: list

class StructuredDecompositionAPI(FactualDecompositionMethod):
    def __init__(self, model:str, instruction:list=CHAT, decomposition_depth:int=1, **kwargs):
        super().__init__()

        if type(model) == str and not model in AVAILABLE_API_MODELS:
            raise ValueError(f"model {model} is not supported.")
    
        self.model = model
        self.instruction = instruction
        self.decomposition_depth = decomposition_depth
        self.kwargs = kwargs
        self.client = instructor.from_litellm(completion)
   
        if "seed" not in kwargs:
            self.kwargs["seed"] = 42
    
    def decompose_facts(self, input_text:str):

        messages = deepcopy(self.instruction)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=Statements,
            **self.kwargs
        )
        return resp.statements
        
    def __str__(self):
        return "Factual decomposition by using LLMs with API calls.\nModel: " + self.model + "\nOutput structure is enforced with 'instructor' library.\nChat template is:\n" +  str(self.instruction) 
