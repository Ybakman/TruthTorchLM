from .truth_method import TruthMethod
from TruthTorchLM.utils import fix_tokenizer_chat
from TruthTorchLM.utils.google_search_utils import GoogleSerperAPIWrapper
from litellm import completion
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import (
    GOOGLE_CHECK_QUERY_SYSTEM_PROMPT,
    GOOGLE_CHECK_QUERY_USER_PROMPT,
    GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT,
    GOOGLE_CHECK_VERIFICATION_USER_PROMPT,
)

from pydantic import BaseModel
import instructor
import outlines
import torch
import copy


class Verification(BaseModel):
    reasoning: str
    error: Union[None, str]
    correction: str
    factuality: bool


class Response(BaseModel):
    response: list[str]


class GoogleSearchCheck(TruthMethod):
    REQUIRES_NORMALIZATION = False

    def __init__(
        self,
        number_of_snippets: int = 10,
        location: str = "us",
        language: str = "en",
        check_query_system_prompt: str = GOOGLE_CHECK_QUERY_SYSTEM_PROMPT,
        check_query_user_prompt: str = GOOGLE_CHECK_QUERY_USER_PROMPT,
        check_verification_system_prompt: str = GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT,
        check_verification_user_prompt: str = GOOGLE_CHECK_VERIFICATION_USER_PROMPT,
        max_new_tokens=1024,
        temperature=1.0,
        top_k=50,
        num_beams=1,
        **generation_kwargs
    ) -> None:
        super().__init__()
        self.number_of_snippets = number_of_snippets
        self.location = location
        self.language = language
        self.google_serper = GoogleSerperAPIWrapper(
            snippet_cnt=self.number_of_snippets,
            location=self.location,
            language=self.language,
        )
        self.check_query_system_prompt = check_query_system_prompt
        self.check_query_user_prompt = check_query_user_prompt
        self.check_verification_system_prompt = check_verification_system_prompt
        self.check_verification_user_prompt = check_verification_user_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_beams = num_beams
        self.generation_kwargs = generation_kwargs

        outlines.disable_cache()

    def get_evidences(self, query_list: list):
        if len(query_list) > 0:
            # search the queries
            search_results = self.google_serper.run(query_list)
            evidences = [
                [output["content"] for output in search_result]
                for search_result in search_results
            ]
        else:
            evidences = []
            print("The model did not generate any queries.")
        return evidences

    def _google_search_check(
        self, verification_dict: str, evidences: list, queries: list[str]
    ):

        if type(verification_dict.factuality) != bool:
            print("The model did not return a boolean value for factuality.")
            return {
                "truth_value": 0.5,
                "normalized_truth_value": 0.5,
                "evidences": evidences,
                "queries": queries,
                "evidences": evidences,
                "verification": verification_dict,
            }
        else:
            try:
                if verification_dict.factuality == True:
                    truth_value = 1.0

                else:
                    truth_value = 0.0

            except:
                truth_value = 0.5

                print("The model did not produce a valid verification.")

            return {
                "truth_value": truth_value,
                "evidences": evidences,
                "queries": queries,
                "evidences": evidences,
                "verification": verification_dict,
            }

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        messages: list = [],
        context: str = "",
        **kwargs
    ):

        outlines_model = outlines.models.Transformers(model, tokenizer)
        outlines_generator = outlines.generate.json(outlines_model, Response)

        kwargs = copy.deepcopy(kwargs)
        # first we need to generate search queries

        chat = [
            {"role": "system", "content": self.check_query_system_prompt},
            {
                "role": "user",
                "content": self.check_query_user_prompt.format(
                    question=question, input=generated_text
                ),
            },
        ]

        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        query = outlines_generator(prompt, seed=generation_seed).response

        evidences = self.get_evidences(query)

        # Ask model to verify the claim
        outlines_generator = outlines.generate.json(
            outlines_model, Verification)

        chat = [
            {"role": "system", "content": self.check_verification_system_prompt},
            {
                "role": "user",
                "content": self.check_verification_user_prompt.format(
                    question=question,
                    claim=generated_text,
                    evidence=evidences,
                ),
            },
        ]
        tokenizer, chat = fix_tokenizer_chat(tokenizer, chat)

        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        verification = outlines_generator(prompt, seed=generation_seed)

        return self._google_search_check(verification, evidences, query)

    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)

        # cerare instructor client object for structured llm output
        client = instructor.from_litellm(completion)

        # first we need to generate search queries
        chat = [
            {"role": "system", "content": GOOGLE_CHECK_QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GOOGLE_CHECK_QUERY_USER_PROMPT.format(
                    question=question, input=generated_text
                ),
            },
        ]

        query = client.chat.completions.create(
            model=model,
            messages=chat,
            response_model=Response,
        ).response

        evidences = self.get_evidences(query)

        # Ask model to verify the claim
        chat = [
            {"role": "system", "content": GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GOOGLE_CHECK_VERIFICATION_USER_PROMPT.format(
                    question=question,
                    claim=generated_text,
                    evidence=evidences,
                ),
            },
        ]
        verification = client.chat.completions.create(
            model=model,
            messages=chat,
            response_model=Verification,
        )

        return self._google_search_check(verification, evidences, query)
