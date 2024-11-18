from .correctness_evaluator import CorrectnessEvaluator
import evaluate
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from litellm import completion
import random
import torch
from TruthTorchLM.templates import DEFAULT_JUDGE_PROMPT, DEFAULT_JUDGE_SYSTEM_PROMPT


class ModelJudge(CorrectnessEvaluator):
    def __init__(self, model:Union[PreTrainedModel, str], tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, prompt:str = DEFAULT_JUDGE_PROMPT, system_prompt:str = DEFAULT_JUDGE_SYSTEM_PROMPT, num_retries:int = 1) -> None: 
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.num_retries = num_retries
        

    def __call__(self, question_text:str, generated_text: str,  ground_truths: list[str], seed:int = None) -> bool:
        if seed == None:
            seed = random.randint(0, 1000000)

        chat = [{"role": "system", "content": self.system_prompt},
        {"role": "user", "content": self.prompt.format(question = question_text, ground_truths = ', '.join(ground_truths), answer = generated_text)}]
        if type(self.model) == str:
            response = completion(
                        model=self.model,
                        messages=chat,
                        seed=seed,
                        num_retries=self.num_retries
                    )
            generated_text = response.choices[0].message['content']
        else:
            torch.manual_seed(seed)
            random.seed(seed)
            text = self.tokenizer.apply_chat_template(chat, tokenize = False)
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
            model_output = self.model.generate(input_ids)
            tokens = model_output[0][len(input_ids[0]):]
            generated_text = self.tokenizer.decode(tokens, skip_special_tokens = False)

        if 'incorrect' in generated_text.lower():
            return 0
        elif 'correct' in generated_text.lower(): 
            return 1
        elif "not_attempted" in generated_text.lower():
            return -1
        else:
            #output warning
            print("The output of the judge model is not in the expected format. Not attempted will be returned.")
            return -1


            
            
        
    
    def __str__(self):
        return f"ROUGE with threshold {self.threshold} and type {self.rouge_type}"