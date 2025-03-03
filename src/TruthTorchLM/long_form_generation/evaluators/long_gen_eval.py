from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.long_form_generation.decomposition_methods.decomposition_method import (
    DecompositionMethod,
)
from TruthTorchLM.long_form_generation.claim_check_methods.claim_check_method import (
    ClaimCheckMethod,
)
from TruthTorchLM.availability import AVAILABLE_EVALUATION_METRICS
from TruthTorchLM.templates import (
    DEFAULT_SYSTEM_BENCHMARK_PROMPT,
    DEFAULT_USER_PROMPT,
)  # TODO
from TruthTorchLM.utils.eval_utils import metric_score
from TruthTorchLM.long_form_generation.utils.dataset_utils import get_dataset
from TruthTorchLM.long_form_generation.utils.eval_utils import (
    run_over_dataset,
    run_over_labelled_dataset,
)

import wandb
import numpy as np
from typing import Union


def evaluate_truth_method_long_form(
    dataset: Union[str, list],
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    sample_level_eval_metrics: list[str] = ["f1"],
    dataset_level_eval_metrics: list[str] = ["auroc", "prr"],
    decomp_method: DecompositionMethod = None,
    claim_check_methods: list[ClaimCheckMethod] = None,
    claim_evaluator=None,
    size_of_data: int = 100,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_method_details: bool = False,
    wandb_run=None,
    return_calim_eval_details: bool = False,
    add_generation_prompt=True,
    continue_final_message=False,
    **kwargs,
):
    """
    available dataset formats:
                    str in LONG_FORM_AVAILABLE_DATASETS,
                    list of dicts with key "question",
                    list of dict with keys "question", "claims", "claim_correctness", optional key: "generated_text"
    """
    dataset = get_dataset(dataset, size_of_data=size_of_data, seed=seed)
    for eval_metric in dataset_level_eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(
                f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}"
            )
    for eval_metric in sample_level_eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(
                f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}"
            )

    if "claims" in dataset[0]:
        output_dict = run_over_labelled_dataset(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            claim_check_methods=claim_check_methods,
            previous_context=previous_context,
            user_prompt=user_prompt,
            seed=seed,
            return_method_details=return_method_details,
            **kwargs,
        )
    else:
        output_dict = run_over_dataset(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            decomp_method=decomp_method,
            claim_check_methods=claim_check_methods,
            claim_evaluator=claim_evaluator,
            previous_context=previous_context,
            user_prompt=user_prompt,
            seed=seed,
            return_method_details=return_method_details,
            return_calim_eval_details=return_calim_eval_details,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **kwargs,
        )

    sample_level_eval_list = get_metric_scores_sample_level(
        output_dict=output_dict, eval_metrics=sample_level_eval_metrics, seed=seed
    )
    dataset_level_eval_list = get_metric_scores_dataset_level(
        output_dict=output_dict, eval_metrics=dataset_level_eval_metrics, seed=seed
    )

    label_name_mapping = {}
    for i in range(len(claim_check_methods)):
        if "truth_methods" in output_dict[f"claim_check_methods_{i}"]:
            for j in range(
                len(output_dict[f"claim_check_methods_{i}"]["truth_methods"])
            ):
                label_name_mapping[f"claim_check_methods_{i}_truth_method_{j}"] = (
                    str(claim_check_methods[i].__class__.__name__)
                    + "-"
                    + str(claim_check_methods[i].truth_methods[j].__class__.__name__)
                )
        else:
            label_name_mapping[f"claim_check_methods_{i}"] = str(
                claim_check_methods[i].__class__.__name__
            )

    if wandb_run:
        sample_level_avg = 0
        dataset_level_avg = 0
        total_claims = 0
        for sample in output_dict["claim_correctness"]:
            sample = np.array(sample)
            values = sample[sample != -1]
            sample_level_avg += np.mean(values)
            dataset_level_avg += sum(values)
            total_claims += len(values)
        wandb_run.log(
            {
                "avg_model_accuracy_sample_level": sample_level_avg
                / len(output_dict["claim_correctness"]),
                "model_accuracy_dataset_level": dataset_level_avg / total_claims,
            }
        )

        # sample level results
        for key in sample_level_eval_metrics:
            data = []
            for method in sample_level_eval_list:
                data.append(
                    [
                        label_name_mapping[method],
                        sample_level_eval_list[method][key]["mean"],
                        sample_level_eval_list[method][key]["std"],
                        sample_level_eval_list[method][key]["min"],
                        sample_level_eval_list[method][key]["max"],
                    ]
                )
            table = wandb.Table(
                data=data, columns=["methods", "mean", "std", "min", "max"]
            )
            wandb.log(
                {
                    f"Sample Level {key}": wandb.plot.bar(
                        table,
                        "methods",
                        "scores",
                        title=f"Sample Level {key} Scores of Truth Methods",
                    )
                }
            )

        # dataset level results
        for key in dataset_level_eval_metrics:
            data = []
            for method in dataset_level_eval_list:
                data.append(
                    [label_name_mapping[method], dataset_level_eval_list[method][key]]
                )
            table = wandb.Table(data=data, columns=["methods", "scores"])
            wandb.log(
                {
                    f"Dataset Level {key}": wandb.plot.bar(
                        table,
                        "methods",
                        "scores",
                        title=f"Dataset Level {key} Scores of Truth Methods",
                    )
                }
            )

    return {
        "sample_level_eval_list": sample_level_eval_list,
        "dataset_level_eval_list": dataset_level_eval_list,
        "output_dict": output_dict,
    }


def get_metric_scores_sample_level(
    output_dict: dict, eval_metrics: list[str], seed: int = 0
):
    claim_check_methods = output_dict["claim_check_methods"]
    for k in range(len(output_dict["claim_correctness"])):
        output_dict["claim_correctness"][k] = np.array(
            output_dict["claim_correctness"][k]
        )
    eval_result = {}
    for i in range(len(claim_check_methods)):  # claim check method index
        if "truth_methods" in output_dict[f"claim_check_methods_{i}"]:
            for j in range(
                len(output_dict[f"claim_check_methods_{i}"]["truth_methods"])
            ):  # truth method index of the claim check method
                eval_result[f"claim_check_methods_{i}_truth_method_{j}"] = {}
                for metric in eval_metrics:
                    eval_result[f"claim_check_methods_{i}_truth_method_{j}"][metric] = {
                        "values": []
                    }
                # question index
                for k in range(len(output_dict["claim_correctness"])):
                    indices = np.where(
                        output_dict["claim_correctness"][k] != -1)[0]
                    predictions = [
                        output_dict[f"claim_check_methods_{i}"]["truth_values"][k][m][j]
                        for m in indices
                    ]  # m --> claim index within the answer to the question
                    predictions_normalized = [
                        output_dict[f"claim_check_methods_{i}"][
                            "normalized_truth_values"
                        ][k][m][j]
                        for m in indices
                    ]
                    eval_dicts = metric_score(
                        eval_metrics,
                        np.array(output_dict["claim_correctness"][k])[indices],
                        predictions,
                        predictions_normalized,
                        seed=seed,
                    )
                    for metric in eval_metrics:  # metric index
                        eval_result[f"claim_check_methods_{i}_truth_method_{j}"][
                            metric
                        ]["values"].append(eval_dicts[metric])
        else:
            eval_result[f"claim_check_methods_{i}"] = {}
            for metric in eval_metrics:
                eval_result[f"claim_check_methods_{i}"][metric] = {
                    "values": []}
            # question index
            for k in range(len(output_dict["claim_correctness"])):
                indices = np.where(
                    output_dict["claim_correctness"][k] != -1)[0]
                predictions = np.array(
                    output_dict[f"claim_check_methods_{i}"]["truth_values"][k]
                )[indices]
                predictions_normalized = np.array(
                    output_dict[f"claim_check_methods_{i}"]["normalized_truth_values"][
                        k
                    ]
                )[indices]
                eval_dicts = metric_score(
                    eval_metrics,
                    np.array(output_dict["claim_correctness"][k])[indices],
                    predictions,
                    predictions_normalized,
                    seed=seed,
                )
                for metric in eval_metrics:  # metric index
                    eval_result[f"claim_check_methods_{i}"][metric]["values"].append(
                        eval_dicts[metric]
                    )

    for method in eval_result:
        for metric in eval_metrics:
            values = eval_result[method][metric]["values"]
            eval_result[method][metric]["mean"] = np.mean(values)
            eval_result[method][metric]["max"] = np.max(values)
            eval_result[method][metric]["min"] = np.min(values)
            eval_result[method][metric]["std"] = np.std(values)
    return eval_result


def get_metric_scores_dataset_level(
    output_dict: dict, eval_metrics: list[str], seed: int = 0
):
    claim_check_methods = output_dict["claim_check_methods"]
    labels = []
    for k in range(len(output_dict["claim_correctness"])):
        output_dict["claim_correctness"][k] = np.array(
            output_dict["claim_correctness"][k]
        )
        labels.extend(
            np.array(output_dict["claim_correctness"][k])[
                np.where(output_dict["claim_correctness"][k] != -1)[0]
            ]
        )
    eval_result = {}
    for i in range(len(claim_check_methods)):  # claim check method index
        if "truth_methods" in output_dict[f"claim_check_methods_{i}"]:
            for j in range(
                len(output_dict[f"claim_check_methods_{i}"]["truth_methods"])
            ):  # truth method index of the claim check method
                predictions = []
                predictions_normalized = []
                # question index
                for k in range(len(output_dict["claim_correctness"])):
                    indices = np.where(
                        output_dict["claim_correctness"][k] != -1)[0]
                    predictions.extend(
                        [
                            output_dict[f"claim_check_methods_{i}"]["truth_values"][k][
                                m
                            ][j]
                            for m in indices
                        ]
                    )
                    predictions_normalized.extend(
                        [
                            output_dict[f"claim_check_methods_{i}"][
                                "normalized_truth_values"
                            ][k][m][j]
                            for m in indices
                        ]
                    )
                eval_result[f"claim_check_methods_{i}_truth_method_{j}"] = metric_score(
                    eval_metrics, labels, predictions, predictions_normalized, seed=seed
                )
        else:
            predictions = []
            predictions_normalized = []
            # question index
            for k in range(len(output_dict["claim_correctness"])):
                indices = np.where(
                    output_dict["claim_correctness"][k] != -1)[0]
                predictions.extend(
                    np.array(
                        output_dict[f"claim_check_methods_{i}"]["truth_values"][k]
                    )[indices]
                )
                predictions_normalized.extend(
                    np.array(
                        output_dict[f"claim_check_methods_{i}"][
                            "normalized_truth_values"
                        ][k]
                    )[indices]
                )
            eval_result[f"claim_check_methods_{i}"] = metric_score(
                eval_metrics, labels, predictions, predictions_normalized, seed=seed
            )

    return eval_result
