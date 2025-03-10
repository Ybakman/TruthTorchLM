from .truth_method import TruthMethod
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

import torch
import numpy as np


class AttentionScore(TruthMethod):
    def __init__(self, layer_index: int = -1):  # normalization,
        super().__init__()
        self.layer_index = layer_index

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

        model_output = all_ids.to(model.device)
        with torch.no_grad():
            output = model(model_output, output_attentions=True)
            target_attention = output.attentions[self.layer_index].cpu()
            del output
            scores = []
            # for each head
            for head_index in range(target_attention.shape[1]):
                attention = target_attention[0][
                    head_index
                ]  # this values are after softmax
                diag_entries = torch.diagonal(attention)
                log_diag_entries = torch.log(diag_entries)
                score = log_diag_entries.sum().item()
                score = score / len(diag_entries)
                scores.append(score)
            result = np.mean(scores)

        return {
            "truth_value": result,
            "attention_score": result,
        }  # we shouldn't return generated text. remove it from the output format

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

        raise ValueError(
            "Attention Score method cannot be used with black-box API models since it requires access to activations."
        )

        return {
            "truth_value": 0
        }  # this output format should be same for all truth methods
