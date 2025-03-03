from .decomposition_method import DecompositionMethod
from ..templates import DECOMPOSITION_INSTRUCTION, DECOMPOSITION_INSTRUCTION_GRANULAR

import instructor
from copy import deepcopy
from litellm import completion
from pydantic import BaseModel


class Claims(BaseModel):
    claims: list[str]


class StructuredDecompositionAPI(DecompositionMethod):
    def __init__(
        self,
        model: str,
        instruction: list = DECOMPOSITION_INSTRUCTION,
        instruction_for_granular: list = DECOMPOSITION_INSTRUCTION_GRANULAR,
        decomposition_depth: int = 1,
        split_by_paragraphs=True,
        **kwargs
    ):
        super().__init__()

        self.model = model
        self.instruction = instruction
        self.instruction_for_granular = instruction_for_granular
        self.decomposition_depth = decomposition_depth
        self.split_by_paragraphs = split_by_paragraphs
        self.kwargs = kwargs
        self.client = instructor.from_litellm(completion)

        if "seed" not in kwargs:
            self.kwargs["seed"] = 42

    def decompose_facts(self, input_text: str, level: int = 1):

        messages = deepcopy(
            self.instruction if level == 1 else self.instruction_for_granular
        )
        for item in messages:
            item["content"] = item["content"].format(TEXT=input_text)

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, response_model=Claims, **self.kwargs
        )
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
            "Decomposition by using LLMs with API calls.\nModel: "
            + self.model
            + "\nOutput structure is enforced with 'instructor' library.\nInstruction for the first level of decomposition is:\n"
            + str(self.instruction)
            + "\nInstruction for the granular decomposition (level 2 and higher) is:\n"
            + str(self.instruction_for_granular)
        )
