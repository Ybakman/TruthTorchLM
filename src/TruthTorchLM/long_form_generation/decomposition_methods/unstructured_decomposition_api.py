from .decomposition_method import DecompositionMethod
from ..templates import UNSTRUCTURED_DECOMPOSITION_INSTRUCTION

from copy import deepcopy
from typing import Callable
from litellm import completion


def default_output_parser(text: str):
    claims = text.split("\n-")
    claims = [claim.strip() for claim in claims if claim.strip()]
    return claims


class UnstructuredDecompositionAPI(DecompositionMethod):
    def __init__(
        self,
        model: str,
        instruction: list = UNSTRUCTURED_DECOMPOSITION_INSTRUCTION,
        decomposition_depth: int = 1,
        output_parser: Callable[[str], list[str]] = default_output_parser,
        split_by_paragraphs=True,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.instruction = instruction
        self.decomposition_depth = decomposition_depth
        self.split_by_paragraphs = split_by_paragraphs
        self.output_parser = output_parser
        self.kwargs = kwargs

        if "seed" not in kwargs:
            self.kwargs["seed"] = 42

    def decompose_facts(self, input_text: str):

        messages = deepcopy(self.instruction)
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)

        response = completion(
            model=self.model, messages=messages, **self.kwargs)
        generated_text = "\n" + response.choices[0].message["content"]
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
