from .truth_method import TruthMethod
from TruthTorchLM.scoring_methods import ScoringMethod, LengthNormalizedScoring
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from TruthTorchLM.error_handler import handle_logprobs_error


class Confidence(TruthMethod):

    REQUIRES_LOGPROBS = True

    def __init__(
        self, scoring_function: ScoringMethod = LengthNormalizedScoring()
    ):  # normalization,
        super().__init__()
        self.scoring_function = scoring_function

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

        input_ids = tokenizer.encode(
            input_text, return_tensors="pt").to(model.device)
        model_output = all_ids.to(model.device)

        with torch.no_grad():
            outputs = model(model_output)
            logits = outputs.logits  # Logits for each token in the input

            # Calculate probabilities from logits
            logprobs = torch.log_softmax(
                logits, dim=-1)  # logprobs for each token
            logprobs = logprobs[
                0, len(input_ids[0]) - 1: -1, :
            ]  # logprobs for each token in the generated text
            logprobs = torch.gather(
                logprobs, dim=1, index=model_output[0][len(input_ids[0]):].view(-1, 1)
            )  # logprobs for each token in the generated text
            logprobs = logprobs.view(-1).tolist()  # convert to list

            score = self.scoring_function(logprobs)
            score = score
            generated_text = generated_text

        return {
            "truth_value": score,
            "generated_text": generated_text,
        }  # we shouldn't return generated text. remove it from the output format

    @handle_logprobs_error
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

        score = self.scoring_function(logprobs)

        return {
            "truth_value": score,
            "generated_text": generated_text,
        }  # we shouldn't return generated text. remove it from the output format
