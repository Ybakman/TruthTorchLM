from .decomposition_method import DecompositionMethod
from ..templates import UNSTRUCTURED_DECOMPOSITION_INSTRUCTION
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat

from copy import deepcopy
from typing import Union, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


def default_output_parser(text: str):
    claims = text.split("\n•")
    claims = [claim.strip() for claim in claims if claim.strip()]
    return claims


class UnstructuredDecompositionLocal(DecompositionMethod):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        instruction: list = UNSTRUCTURED_DECOMPOSITION_INSTRUCTION,
        decomposition_depth: int = 1,
        output_parser: Callable[[str], list[str]] = default_output_parser,
        split_by_paragraphs=True,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.output_parser = output_parser
        self.decomposition_depth = decomposition_depth
        self.split_by_paragraphs = split_by_paragraphs

        default_kwargs = {"top_p": 1, "do_sample": False, "temperature": None}
        default_kwargs.update(kwargs)

        eos_token_id = default_kwargs.pop("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = model.config.eos_token_id
        default_kwargs["eos_token_id"] = eos_token_id

        pad_token_id = default_kwargs.pop("pad_token_id", None)
        if pad_token_id is None:
            if type(eos_token_id) == list:
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id
        default_kwargs["pad_token_id"] = pad_token_id
        self.kwargs = default_kwargs

    def decompose_facts(self, input_text: str):

        messages = deepcopy(self.instruction)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)
        self.tokenizer, messages = fix_tokenizer_chat(self.tokenizer, messages)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        generated_output = generate(
            text, self.model, self.tokenizer, **self.kwargs)
        generated_text = "\n" + \
            generated_output["generated_text_skip_specials"].strip()
        claims = self.output_parser(generated_text)

        return claims

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
            claims = self.decompose_facts(paragraph)
            for _ in range(self.decomposition_depth - 1):
                temp_claims = []
                for claim in claims:
                    temp_claims.extend(self.decompose_facts(claim))
                claims = temp_claims
            all_claims.extend(claims)
        return all_claims

    def __str__(self):
        return (
            "Decomposition by using LLMs method with "
            + self.model
            + " model. Chat template is:\n"
            + str(self.instruction)
            + "\n Sentence seperator is: "
            + self.sentence_seperator
        )
