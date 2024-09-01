from TruthTorchLLM.decomposition_methods import FactualDecompositionMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from typing import Callable

from copy import deepcopy

CHAT = [{"role": "system", "content": 'You are a helpful assistant. List the specific factual propositions included in the given input. Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a separate bullet point. Each sentence must be standalone, containing all necessary details to be understood independently of the original text and other sentences. This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. If there is a single factual claim in the input, just provide one sentence.'},
        {"role": "user", "content": '''{TEXT}'''}]

def default_output_parser(text:str):
        statements = text.split("\n•")
        statements = [statement.strip() for statement in statements if statement.strip()]
        return statements

class FactualDecompositionLocal(FactualDecompositionMethod):
    def __init__(self, model:PreTrainedModel, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], chat_template:list=CHAT, decomposition_depth:int=1, output_parser:Callable[[str],list[str]]=default_output_parser):
        super().__init__()
    
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.output_parser = output_parser
        self.decomposition_depth = decomposition_depth

    def _decompose_facts(self, input_text:str):
        messages = deepcopy(self.chat_template)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)

        text = self.tokenizer.apply_chat_template(messages, tokenize = False)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        model_output = self.model.generate(input_ids)
        tokens = model_output[0][len(input_ids[0]):]
        generated_text = "\n" + self.tokenizer.decode(tokens, skip_special_tokens = True).strip()

        statements = self.output_parser(generated_text)

        return {'statements_text': generated_text, "statements": statements}
    
    def decompose_facts(self, input_text:str):

        all_outputs = []
        first_run_output = self._decompose_facts(input_text)
        all_outputs.append(first_run_output)
        statements = first_run_output["statements"]
        for _ in range(self.decomposition_depth-1):
            temp_statements = []
            for statement in statements:
                new_output = self._decompose_facts(statement)
                all_outputs.append(new_output)
                temp_statements.extend(new_output["statements"])
            statements = temp_statements

        return {'all_outputs': all_outputs, "statements": statements}
        
    def __str__(self):
        return "Factual decomposition by using LLMs method with " + self.model + " model. Chat template is:\n" +  str(self.chat_template) + "\n Sentence seperator is: " + self.sentence_seperator
