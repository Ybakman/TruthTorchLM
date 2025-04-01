from .truth_method import TruthMethod
from TruthTorchLM.utils import fix_tokenizer_chat
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.utils.common_utils import generate
from ..generation import sample_generations_api
from TruthTorchLM.templates import VC_SYSTEM_PROMPT, VC_USER_PROMPT, VC_USER_PROMPT_WITH_CONTEXT
import torch
import copy





class VerbalizedConfidence(TruthMethod):

    def __init__(
        self,
        system_prompt: str = VC_SYSTEM_PROMPT,
        user_prompt: str = VC_USER_PROMPT,
        with_context: bool = False,
        max_new_tokens=1024,
        **generation_kwargs
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = generation_kwargs
        self.with_context = with_context

        if with_context and '{context}' not in self.user_prompt:#check if the prompt has a context field
            print("Context field is required in user prompt for with_context=True, swithing to the default user prompt with context")
            self.user_prompt = VC_USER_PROMPT_WITH_CONTEXT

    def extract_confidence(self, confidence_text):
        """Extracts the confidence value from the confidence text. The text may include non-numeric characters."""
        confidence_text = confidence_text.strip()
        confidence_text = "".join(
            [c for c in confidence_text if c.isdigit() or c == "."]
        )
        try:
            return float(confidence_text)
        except:
            return 0.0

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
        if self.with_context == False:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, generated_text=generated_text
                    ),
                },
            ]
        else:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, generated_text=generated_text, context=context
                    ),
                },
            ]

        tokenizer, chat = fix_tokenizer_chat(
            tokenizer, chat
        )  # in case some tokenizers don't have chat template and don't support system prompt
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )

        kwargs = copy.deepcopy(kwargs)
        kwargs.pop("do_sample", None)
        generation_dict = generate(
            prompt,
            model,
            tokenizer,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            **self.generation_kwargs
        )
        confidence_text = generation_dict["generated_text_skip_specials"]
        confidence = self.extract_confidence(confidence_text)
        confidence = confidence / 100.0

        return {"truth_value": confidence, "confidence_text": confidence_text}

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
        if self.with_context == False:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, generated_text=generated_text
                    ),
                },
            ]
        else:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self.user_prompt.format(
                        question=question, generated_text=generated_text, context=context
                    ),
                },
            ]

        sampled_generations_dict = sample_generations_api(
            model=model,
            messages=chat,
            generation_seed=generation_seed,
            number_of_generations=1,
            return_text=True,
            **kwargs
        )
        confidence_text = sampled_generations_dict["generated_texts"][0]
        confidence = self.extract_confidence(confidence_text)
        confidence = confidence / 100.0
        return {"truth_value": confidence, "confidence_text": confidence_text}
