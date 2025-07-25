from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.truth_methods import TruthMethod
from .correctness_evaluator import CorrectnessEvaluator
from .exact_match import ExactMatch
from TruthTorchLM.availability import AVAILABLE_EVALUATION_METRICS
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.utils.dataset_utils import get_dataset
from TruthTorchLM.utils.eval_utils import metric_score, run_over_dataset
import wandb


def evaluate_truth_method(
    dataset: Union[str, list],
    model: Union[str, PreTrainedModel],
    truth_methods: list[TruthMethod],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    eval_metrics: list[str] = ["auroc"],
    correctness_evaluator: CorrectnessEvaluator = ExactMatch(),
    size_of_data=1.0,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_method_details: bool = False,
    batch_generation=True,
    add_generation_prompt=True,
    continue_final_message=False,
    split="test",
    **kwargs,
):

    dataset = get_dataset(
        dataset, size_of_data=size_of_data, seed=seed, split=split)

    for eval_metric in eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(
                f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}"
            )

    output_dict = run_over_dataset(
        dataset,
        model,
        truth_methods,
        tokenizer=tokenizer,
        correctness_evaluator=correctness_evaluator,
        previous_context=previous_context,
        user_prompt=user_prompt,
        seed=seed,
        return_method_details=return_method_details,
        batch_generation=batch_generation,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        **kwargs,
    )

    eval_list = get_metric_scores(
        output_dict=output_dict, eval_metrics=eval_metrics, seed=seed
    )

    return {"eval_list": eval_list, "output_dict": output_dict}


def get_metric_scores(output_dict: dict, eval_metrics: list[str], seed: int = 0):
    truth_methods = output_dict["truth_methods"]
    eval_list = []
    for i in range(len(truth_methods)):
        eval_dict = metric_score(
            eval_metrics,
            output_dict["generations_correctness"],
            output_dict[f"truth_method_{i}"]["truth_values"],
            output_dict[f"truth_method_{i}"]["normalized_truth_values"],
            seed=seed,
        )
        eval_list.append(eval_dict)
    return eval_list
