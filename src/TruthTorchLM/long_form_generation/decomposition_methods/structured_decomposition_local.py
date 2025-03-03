from ..templates import DECOMPOSITION_INSTRUCTION, DECOMPOSITION_INSTRUCTION_GRANULAR
from .decomposition_method import DecompositionMethod
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat

import outlines
from typing import Union
from copy import deepcopy
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class Claims(BaseModel):
    claims: list[str]


class StructuredDecompositionLocal(DecompositionMethod):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        instruction: list = DECOMPOSITION_INSTRUCTION,
        instruction_for_granular: list = DECOMPOSITION_INSTRUCTION_GRANULAR,
        decomposition_depth: int = 1,
        seed: int = 715,
        split_by_paragraphs=True,
    ):
        super().__init__()

        outlines.disable_cache()
        outlines_model = outlines.models.Transformers(model, tokenizer)
        self.generator = outlines.generate.json(outlines_model, Claims)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.instruction_for_granular = instruction_for_granular
        self.decomposition_depth = decomposition_depth
        self.split_by_paragraphs = split_by_paragraphs
        self.seed = seed
        self.model_name = model.name_or_path

    def decompose_facts(self, input_text: str, level: int = 1):

        messages = deepcopy(
            self.instruction if level == 1 else self.instruction_for_granular
        )

        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)
        self.tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        resp = self.generator(text, seed=self.seed)

        return resp.claims

    def __call__(self, input_text) -> list[str]:

        if self.split_by_paragraphs:
            paragraphs = [
                paragraph.strip()
                for paragraph in input_text.split("\n")
                if paragraph.strip()
            ]
        else:
            paragraphs = [input_text]
        all_claims = []
        for paragraph in paragraphs:
            claims = self.decompose_facts(paragraph, level=1)
            for d in range(1, self.decomposition_depth):
                temp_claims = []
                for claim in claims:
                    temp_claims.extend(
                        self.decompose_facts(claim, level=d + 1))
                claims = temp_claims
            all_claims.extend(claims)
        return all_claims

    def __str__(self):
        return (
            "Decomposition by using LLMs.\nModel: "
            + self.model_name
            + "\nOutput structure is enforced with 'outlines' library.\nInstruction for the first level of decomposition is:\n"
            + str(self.instruction)
            + "\nInstruction for the granular decomposition (level 2 and higher) is:\n"
            + str(self.instruction_for_granular)
        )
