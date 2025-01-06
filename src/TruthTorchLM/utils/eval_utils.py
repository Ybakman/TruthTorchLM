from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.generation import generate_with_truth_value
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np




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
    df = pd.DataFrame({"t_s": t_s, 'acc': acc}).sort_values('t_s', ascending=False)#that should be false in case of truth values
    df['acc_mean'] = df['acc'].expanding().mean()
    return auc(np.linspace(0,1,len(df)), df['acc_mean'])

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
    target = normalize(target) #higher is correct
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

def metric_score(metric_names:list[str], generation_correctness:list, truth_values:list[float], normalized_truth_values:list[float] = [],  seed:int = 0) -> dict:
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
    #if generation_correctness is -1, it means that the model didn't attempt to generate an answer, remove those from the evaluation
    generation_correctness = np.array(generation_correctness)
    truth_values = np.array(truth_values)
    normalized_truth_values = np.array(normalized_truth_values)
    truth_values = truth_values[generation_correctness != -1]
    normalized_truth_values = normalized_truth_values[generation_correctness != -1]
    generation_correctness = generation_correctness[generation_correctness != -1]

    # print(f'total nan values in truth values: {np.sum(np.isnan(truth_values))}')
    # print(f'total nan values in normalized truth values: {np.sum(np.isnan(normalized_truth_values))}')
    #replace NaN values with 0
    truth_values[np.isnan(truth_values)] = 0
    normalized_truth_values[np.isnan(normalized_truth_values)] = 0

    # print(f'total inf values in truth values: {np.sum(np.isinf(truth_values))}')
    # print(f'total inf values in normalized truth values: {np.sum(np.isinf(normalized_truth_values))}')
    truth_values[np.isinf(truth_values)] = 0
    normalized_truth_values[np.isinf(normalized_truth_values)] = 0

    truth_values = list(truth_values) #convert to list
    normalized_truth_values = (normalized_truth_values) #convert to list

    if "auroc" in metric_names:
        try:
            auroc = roc_auc_score(generation_correctness, truth_values) 
        except:
            print("Auroc couldn't be calculated because there is only one class. Returning 0.5 as auroc.")
            auroc = 0.5
        eval_dict['auroc'] = auroc

    if 'auprc' in metric_names:
        precision, recall, thresholds = precision_recall_curve(generation_correctness, truth_values)
        auprc = auc(recall, precision)
        eval_dict['auprc'] = auprc

    if 'auarc' in metric_names:
        #area under accuracy-coverage curve
        auarc = area_under_accuracy_coverage_curve(truth_values, generation_correctness)
        eval_dict['auarc'] = auarc

    if 'accuracy' in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        accuracy = np.mean((normalized_truth_values > 0.5) == generation_correctness)
        eval_dict['accuracy'] = accuracy

    if 'f1' in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = (normalized_truth_values > 0.5)
        f1 = f1_score(generation_correctness, predictions, zero_division=1)
        eval_dict['f1'] = f1
    if 'precision' in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = (normalized_truth_values > 0.5)
        precision = precision_score(generation_correctness, predictions, zero_division=1)
        eval_dict['precision'] = precision
    if 'recall' in metric_names:
        normalized_truth_values = np.array(normalized_truth_values)
        predictions = (normalized_truth_values > 0.5)
        recall = recall_score(generation_correctness, predictions, zero_division=1)
        eval_dict['recall'] = recall

    if 'prr' in metric_names:
        ue_prr = prediction_rejection_curve(truth_values, generation_correctness)
        orc_prr = prediction_rejection_curve(generation_correctness, generation_correctness)
        rand_prr = get_random_scores(prediction_rejection_curve, generation_correctness, seed = seed)

        if not (orc_prr == rand_prr):
            ue_prr = (ue_prr - rand_prr) / (orc_prr - rand_prr)
        eval_dict['prr'] = ue_prr

    return eval_dict


def run_over_dataset(dataset: Union[str, list], model:Union[str,PreTrainedModel],  truth_methods: list, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                          correctness_evaluator = None, previous_context:list =[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], user_prompt:str = DEFAULT_USER_PROMPT, seed:int = 0, return_method_details:bool = False, wandb_run = None, 
                          wandb_push_method_details:bool = False, batch_generation=True,  add_generation_prompt = True, continue_final_message = False, **kwargs):
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
        wandb_run (optional): Weights & Biases run for logging. Defaults to None.
        wandb_push_method_details (bool, optional): Whether to log detailed method outputs to W&B. Defaults to False.
        batch_generation (bool, optional): Whether to use batch generation. Defaults to True.
        add_generation_prompt (bool, optional): Whether to add generation prompt. Defaults to True.
        continue_final_message (bool, optional): Whether to continue from final message. Defaults to False.

    Returns:
        dict: Dictionary containing all evaluation results and generations
    """
    output_dict = {}
    output_dict['previous_context'] = previous_context
    output_dict['user_prompt'] = user_prompt
    output_dict['generation'] = []
    output_dict['generation_correctness'] = []
    output_dict['question_text'] = []
    output_dict['ground_truths'] = []
    
    output_dict['truth_methods'] = []#save the truth methods

    
    for i in range(len(truth_methods)):
        output_dict['truth_methods'].append(f'{truth_methods[i].__class__.__name__}')
        output_dict[f'truth_method_{i}'] = {}
        output_dict[f'truth_method_{i}']['name'] = str(truth_methods[i])
        output_dict[f'truth_method_{i}']['truth_values'] = []
        output_dict[f'truth_method_{i}']['normalized_truth_values'] = []  
        if return_method_details:
            output_dict[f'truth_method_{i}']['method_specific_details'] = []

    if wandb_run is not None:
        logged_data = []
        #add method names to the columns
        names = []
        columns = []
        for i in range(len(truth_methods)):
            names.append(str(truth_methods[i]))
            columns.append(f'truth_method_{i}')
        
        names_table = wandb.Table(data = [names], columns=columns)
        wandb_run.log({"method_names": names_table})
        

    for i in tqdm(range(len(dataset))):
        messages = previous_context.copy()
        messages.append({'role': 'user', 'content': user_prompt.format(question_context = dataset[i]['question'])})

        truth_dict = generate_with_truth_value(model = model, messages = messages, question_context = dataset[i]['question'], truth_methods = truth_methods, tokenizer=tokenizer,
        generation_seed = seed, batch_generation=batch_generation, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message, **kwargs)
        
        is_correct = correctness_evaluator(dataset[i]['question'], truth_dict['generated_text'], dataset[i]['ground_truths'])
        output_dict['generation_correctness'].append(is_correct)
        output_dict['generation'].append(truth_dict['generated_text'])
        output_dict['question_text'].append(dataset[i]['question'])
        output_dict['ground_truths'].append(dataset[i]['ground_truths'])
        
        for j in range(len(truth_methods)):
            output_dict[f'truth_method_{j}']['truth_values'].append(truth_dict['unnormalized_truth_values'][j])
            output_dict[f'truth_method_{j}']['normalized_truth_values'].append(truth_dict['normalized_truth_values'][j])
            if return_method_details:
                output_dict[f'truth_method_{j}']['method_specific_details'].append(truth_dict['method_specific_outputs'][j])
        
        if wandb_push_method_details and wandb_run is not None:
            columns=['truth_values','normalized_truth_values', 'generation_correctness', 'question_text', 'ground_truths', 'generated_text', 'index', 'method_specific_details']
            data = [str(truth_dict['unnormalized_truth_values']), str(truth_dict['normalized_truth_values']), is_correct, dataset[i]['question'], 
                (', ').join(dataset[i]['ground_truths']) , truth_dict['generated_text'], i, str(truth_dict['method_specific_outputs'])]
            logged_data.extend([data])
            summary_table = wandb.Table(data = logged_data, columns=columns)
            wandb_run.log({
                'accuracy': is_correct,
                'index': i,
            })
            wandb.log({"run_summary" : summary_table})

    return output_dict
