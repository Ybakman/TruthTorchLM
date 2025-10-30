from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.generation import generate_with_truth_value, generate_api_or_hf_local, run_truth_methods
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
import wandb
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import numpy as np
import warnings


def area_under_accuracy_coverage_curve(t_s, acc):
    """
    Calculates the area under the accuracy-coverage curve.

    The accuracy-coverage curve shows how model accuracy changes as we include more predictions,
    ordered by their truth scores. This function computes the area under this curve as a
    single metric for evaluating truth value estimation quality.

    Args:
        t_s (array-like): Array of truth scores for each prediction
        acc (array-like): Array of accuracy values (0 or 1) for each prediction

    Returns:
        float: Area under the accuracy-coverage curve. Higher values indicate better
              correlation between truth scores and actual accuracy.
    """
    # area under the rejection-VALUE curve, where VALUE could be accuracy, etc.
    df = pd.DataFrame({"t_s": t_s, "acc": acc}).sort_values(
        "t_s", ascending=False
    )  # that should be false in case of truth values
    df["acc_mean"] = df["acc"].expanding().mean()
    return auc(np.linspace(0, 1, len(df)), df["acc_mean"])


def normalize(target):
    """
    Normalizes an array of values to the range [0,1].

    Args:
        target (array-like): Array of values to normalize

    Returns:
        array: Normalized values between 0 and 1
    """
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1
        max_t += 1
    target = (np.array(target) - min_t) / (max_t - min_t)
    return target


def prediction_rejection_curve(estimator, target):
    """
    Calculates the prediction rejection curve score.

    The prediction rejection curve shows how model performance changes as we reject predictions
    based on their uncertainty estimates.

    Args:
        estimator (array-like): Array of uncertainty estimates for each prediction
        target (array-like): Array of true values/labels

    Returns:
        float: Prediction rejection curve score
    """
    target = normalize(target)  # higher is correct
    # estimator: lower is more uncertain
    ue = np.array(estimator)
    num_obs = len(ue)
    # Sort in descending order: the least uncertain come first
    ue_argsort = np.argsort(ue)[::-1]
    # want sorted_metrics to be increasing => smaller scores is better
    sorted_metrics = np.array(target)[ue_argsort]
    # Since we want all plots to coincide when all the data is discarded
    cumsum = np.cumsum(sorted_metrics)[-num_obs:]
    scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
    prr_score = np.sum(scores) / num_obs
    return prr_score


def get_random_scores(function, metrics, num_iter=1000, seed=42):
    """
    Calculates random baseline scores for a given metric function.

    Args:
        function (callable): Metric function to calculate scores
        metrics (array-like): Array of true metrics/labels
        num_iter (int, optional): Number of random iterations. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        float: Average random baseline score
    """
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    value = []
    for i in range(num_iter):
        np.random.shuffle(rand_scores)
        rand_val = function(rand_scores, metrics)
        value.append(rand_val)
    return np.mean(value)


def metric_score(
    metric_names: list[str],
    generation_correctness: list,
    truth_values: list[float],
    normalized_truth_values: list[float] = [],
    seed: int = 0,
) -> dict:
    """
    Calculates various evaluation metrics for truth value estimation.

    Args:
        metric_names (list[str]): List of metric names to calculate
        generation_correctness (list): Binary list indicating if each generation was correct
        truth_values (list[float]): Raw truth values from the model
        normalized_truth_values (list[float], optional): Normalized truth values. Defaults to [].
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        dict: Dictionary containing calculated metrics
    """
    eval_dict = {}
    # if generation_correctness is -1, it means that the model didn't attempt to generate an answer, remove those from the evaluation
    generation_correctness = np.array(generation_correctness)
    truth_values = np.array(truth_values)
    normalized_truth_values = np.array(normalized_truth_values)
    truth_values = truth_values[generation_correctness != -1]
    normalized_truth_values = normalized_truth_values[generation_correctness != -1]
    generation_correctness = generation_correctness[generation_correctness != -1]

    pos_inf_replacement = 1e10
    neg_inf_replacement = -1e10

    truth_values[np.isnan(truth_values)] = pos_inf_replacement
    normalized_truth_values[np.isnan(normalized_truth_values)] = 1.0

    # print(f'total inf values in truth values: {np.sum(np.isinf(truth_values))}')
    # print(f'total inf values in normalized truth values: {np.sum(np.isinf(normalized_truth_values))}')

    truth_values[truth_values == np.inf] = pos_inf_replacement
    truth_values[truth_values == -np.inf] = neg_inf_replacement

    truth_values = list(truth_values)  # convert to list
    normalized_truth_values = normalized_truth_values  # convert to list

    if "auroc" in metric_names:
        if len(np.unique(generation_correctness)) == 1:
            auroc = 0.5
            print(
                "Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc."
            )
        else:
            auroc = roc_auc_score(generation_correctness, truth_values)
        eval_dict["auroc"] = auroc

    if "auprc" in metric_names:
        precision, recall, thresholds = precision_recall_curve(
            generation_correctness, truth_values
        )
        auprc = auc(recall, precision)
        eval_dict["auprc"] = auprc

    if "auarc" in metric_names:
        # area under accuracy-coverage curve
        auarc = area_under_accuracy_coverage_curve(
            truth_values, generation_correctness)
        eval_dict["auarc"] = auarc

    if "accuracy" in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        accuracy = np.mean((normalized_truth_values > 0.5)
                           == generation_correctness)
        eval_dict["accuracy"] = accuracy

    if "f1" in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = normalized_truth_values > 0.5
        f1 = f1_score(generation_correctness, predictions, zero_division=1)
        eval_dict["f1"] = f1
    if "precision" in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = normalized_truth_values > 0.5
        precision = precision_score(
            generation_correctness, predictions, zero_division=1
        )
        eval_dict["precision"] = precision
    if "recall" in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = normalized_truth_values > 0.5
        recall = recall_score(generation_correctness,
                              predictions, zero_division=1)
        eval_dict["recall"] = recall

    if "prr" in metric_names:
        if len(np.unique(generation_correctness)) == 1:
            print(
                "PRR couldn't be calculated because there is only one class. Returning 0.0 as prr."
            )
            eval_dict["prr"] = 0.0
        else:
            ue_prr = prediction_rejection_curve(
                truth_values, generation_correctness)
            orc_prr = prediction_rejection_curve(
                generation_correctness, generation_correctness
            )
            rand_prr = get_random_scores(
                prediction_rejection_curve, generation_correctness, seed=seed
            )

            if not (orc_prr == rand_prr):
                ue_prr = (ue_prr - rand_prr) / (orc_prr - rand_prr)
            eval_dict["prr"] = ue_prr

    return eval_dict


def eval_model_over_dataset(dataset: Union[str, list],
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    correctness_evaluator=None,
    truth_methods: list = [],
    previous_context: list = [],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    add_generation_prompt=True,
    continue_final_message=False,
    **kwargs,
):
    """
    Evaluates a model over a dataset.
    """
    if dataset[0]["context"] != "" and user_prompt.find("context") == -1:
        user_prompt = "Context: {context}\n" + user_prompt 
        #show warning
        warnings.warn("Context is not in the user prompt but it is provided in the dataset. Adding context to the user prompt. Unexpecting behavior may occur.")
        
    output_dict = {}
    output_dict["previous_context"] = previous_context
    output_dict["user_prompt"] = user_prompt
    output_dict["generations"] = []
    output_dict["generations_correctness"] = []
    output_dict["question_text"] = []
    output_dict["ground_truths"] = []
    output_dict['contexts'] = []

    generation_dicts = []

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        if dataset[i]["context"] != "":
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(context=dataset[i]["context"], question=dataset[i]["question"]),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt.format(question=dataset[i]["question"]),
                }
            )

               
        generation_dict = generate_api_or_hf_local(
            model=model,
            messages=messages,
            truth_methods=truth_methods,
            question=dataset[i]["question"],
            tokenizer=tokenizer,
            generation_seed=seed,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **kwargs,
        )
        generation_dicts.append(generation_dict)

        is_correct = correctness_evaluator(
            dataset[i]["question"],
            generation_dict["generated_text"],
            dataset[i]["ground_truths"],
            dataset[i]["context"],
        )
        output_dict["generations_correctness"].append(is_correct)
        output_dict["generations"].append(generation_dict["generated_text"])
        output_dict["question_text"].append(dataset[i]["question"])
        output_dict["contexts"].append(dataset[i]["context"])
        output_dict["ground_truths"].append(dataset[i]["ground_truths"])
    
    output_dict['generation_dicts'] = generation_dicts

    return output_dict


def run_truth_methods_over_dataset( output_dict: dict,
    model: Union[str, PreTrainedModel],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    truth_methods: list = [],
    batch_generation=True,
    seed: int = 0,
    return_method_details: bool = False):

    output_dict["truth_methods"] = [] 
    for i in range(len(truth_methods)):
        output_dict["truth_methods"].append(
            f"{truth_methods[i].__class__.__name__}")
        output_dict[f"truth_method_{i}"] = {}
        output_dict[f"truth_method_{i}"]["name"] = str(truth_methods[i])
        output_dict[f"truth_method_{i}"]["truth_values"] = []
        output_dict[f"truth_method_{i}"]["normalized_truth_values"] = []
        if return_method_details:
            output_dict[f"truth_method_{i}"]["method_specific_details"] = []

    generation_dicts = output_dict['generation_dicts']
    #go over output_dict and generation_dicts
    for i in tqdm(range(len(generation_dicts))):
        question = output_dict["question_text"][i]
        context = output_dict["contexts"][i]
        messages = generation_dicts[i]["messages"]
        kwargs = generation_dicts[i]['kwargs']
        if type(model) == str:
            response = generation_dicts[i]['response']
            generated_text= generation_dicts[i]['generated_text']
            logprobs = generation_dicts[i]['logprobs']
            generated_tokens = generation_dicts[i]['generated_tokens']
            
            
            truth_dict = run_truth_methods(model = model,
                                            messages = messages,
                                            generated_text = generated_text,
                                            question=question,
                                            truth_methods = truth_methods,
                                            generation_seed = seed,
                                            context = context,
                                            logprobs=logprobs,
                                            generated_tokens=generated_tokens,
                                            **kwargs)
        else:
            model_output = generation_dicts[i]['model_output']
            generated_text= generation_dicts[i]['generated_text']
            text = generation_dicts[i]['text']
            truth_dict = run_truth_methods(model = model,
                                   messages = messages,
                                   question=question,
                                   truth_methods = truth_methods,
                                   tokenizer = tokenizer,
                                   generation_seed = seed,   
                                   context = context,
                                   text = text,
                                   generated_text = generated_text,
                                   model_output = model_output,
                                   batch_generation = batch_generation,
                                   **kwargs)
            
        for j in range(len(truth_methods)):
            output_dict[f"truth_method_{j}"]["truth_values"].append(
                truth_dict["unnormalized_truth_values"][j]
            )
            output_dict[f"truth_method_{j}"]["normalized_truth_values"].append(
                truth_dict["normalized_truth_values"][j]
            )
            if return_method_details:
                output_dict[f"truth_method_{j}"]["method_specific_details"].append(
                    truth_dict["method_specific_outputs"][j]
                )
    return output_dict
    


def run_over_dataset(
    dataset: Union[str, list],
    model: Union[str, PreTrainedModel],
    truth_methods: list,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    correctness_evaluator=None,
    previous_context: list = [
        {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
    ],
    user_prompt: str = DEFAULT_USER_PROMPT,
    seed: int = 0,
    return_method_details: bool = False,
    batch_generation=True,
    add_generation_prompt=True,
    continue_final_message=False,
    **kwargs,
):
    """
    Runs truth value estimation over a dataset and collects results.

    Args:
        dataset (Union[str, list]): Dataset to evaluate on
        model (Union[str,PreTrainedModel]): Model to use for generation
        truth_methods (list): List of truth value estimation methods to evaluate
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): Tokenizer for the model. Defaults to None.
        correctness_evaluator (callable, optional): Function to evaluate correctness of generations. Defaults to None.
        previous_context (list, optional): Previous conversation context. Defaults to system prompt.
        user_prompt (str, optional): Template for user prompts. Defaults to DEFAULT_USER_PROMPT.
        seed (int, optional): Random seed. Defaults to 0.
        return_method_details (bool, optional): Whether to return detailed method outputs. Defaults to False.
        batch_generation (bool, optional): Whether to use batch generation. Defaults to True.
        add_generation_prompt (bool, optional): Whether to add generation prompt. Defaults to True.
        continue_final_message (bool, optional): Whether to continue from final message. Defaults to False.

    Returns:
        dict: Dictionary containing all evaluation results and generations
    """

    print(f'Running Model over dataset with {len(dataset)} examples')
    output_dict = eval_model_over_dataset(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        correctness_evaluator=correctness_evaluator,
        truth_methods=truth_methods,
        previous_context=previous_context,
        user_prompt=user_prompt,
        seed=seed,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        **kwargs,
    )
    
    output_dict = run_truth_methods_over_dataset(output_dict, 
                                                 model = model, 
                                                 tokenizer = tokenizer, 
                                                 truth_methods = truth_methods, 
                                                 batch_generation = batch_generation, 
                                                 seed = seed, 
                                                 return_method_details = return_method_details)

    # Write results to a text file
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"Evaluation Results\n{'='*20}\n\n")
        
        # Write dataset info
        f.write(f"Dataset size: {len(dataset)} examples\n\n")
        
        # Write truth method results
        f.write("Truth Method Results\n{'-'*20}\n")
        for i, method in enumerate(truth_methods):
            method_name = output_dict[f"truth_method_{i}"]["name"]
            truth_values = output_dict[f"truth_method_{i}"]["truth_values"]
            normalized_values = output_dict[f"truth_method_{i}"]["normalized_truth_values"]
            
            f.write(f"\nMethod {i+1}: {method_name}\n")
            f.write(f"Average truth value: {np.mean(truth_values):.4f}\n")
            f.write(f"Average normalized truth value: {np.mean(normalized_values):.4f}\n")
        
        # Write generation correctness stats
        correct_count = sum(1 for x in output_dict["generations_correctness"] if x == 1)
        total_count = len(output_dict["generations_correctness"])
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        f.write(f"\nGeneration Statistics\n{'-'*20}\n")
        f.write(f"Total generations: {total_count}\n")
        f.write(f"Correct generations: {correct_count}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    return output_dict
