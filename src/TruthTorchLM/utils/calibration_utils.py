import numpy as np
from sklearn.metrics import precision_recall_curve


def f1_picker(correctness: list, truth_values: list):
    truth_values = np.array(truth_values)
    truth_values = np.nan_to_num(truth_values)  # Replace NaN with 0
    precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    # Remove NaN values and find the index of the highest F1 score
    f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
    max_index = np.argmax(f1_scores)
    
    # The thresholds array is of length len(precisions) - 1, so we use max_index-1 to get the corresponding threshold
    threshold = thresholds[max_index - 1] if max_index > 0 else thresholds[0]
    return threshold

def precision_picker(correctness: list, truth_values: list, precision: float):
    precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
    # Find the index of the smallest precision that is greater than or equal to the target precision
    index = np.where(precisions >= precision)[0][0]
    # Since precisions is always one element longer than thresholds, we need to adjust the index
    threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    return threshold

def recall_picker(correctness: list, truth_values: list, recall: float):
    precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
    # Find the index of the smallest recall that is greater than or equal to the target recall
    index = np.where(recalls >= recall)[0][0]
    # Since recalls is always one element longer than thresholds, we need to adjust the index
    threshold = thresholds[index - 1] if index > 0 else thresholds[0]
    return threshold

def accuracy_picker(correctness: list, truth_values: list):
    accuracies = [np.mean((truth_values > threshold) == correctness) for threshold in np.unique(truth_values)]
    threshold = np.unique(truth_values)[np.argmax(accuracies)]
    return threshold


# def find_threshold_std(correctness: list, truth_values: list, precision: float = -1, recall: float = -1):
#     std = np.std(truth_values)
#     precisions, recalls, thresholds = precision_recall_curve(correctness, truth_values)
#     if precision != -1:
#         # Find the index of the smallest precision that is greater than or equal to the target precision
#         index = np.where(precisions >= precision)[0][0]
#         # Since precisions is always one element longer than thresholds, we need to adjust the index
#         threshold = thresholds[index - 1] if index > 0 else thresholds[0]
#     elif recall != -1:
#         # Find the index of the smallest recall that is greater than or equal to the target recall
#         index = np.where(recalls >= recall)[0][0]
#         # Since recalls is always one element longer than thresholds, we need to adjust the index
#         threshold = thresholds[index - 1] if index > 0 else thresholds[0]
#     else:
#         # Calculate F1 scores for each threshold
#         f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        
#         # Remove NaN values and find the index of the highest F1 score
#         f1_scores = np.nan_to_num(f1_scores)  # Replace NaN with 0
#         max_index = np.argmax(f1_scores)
        
#         # The thresholds array is of length len(precisions) - 1, so we use max_index-1 to get the corresponding threshold
#         threshold = thresholds[max_index - 1] if max_index > 0 else thresholds[0]

#     return threshold, std
