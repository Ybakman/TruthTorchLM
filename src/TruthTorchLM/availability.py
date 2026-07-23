# AVAILABLE_API_MODELS = ['gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'chatgpt-4o-latest', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
#                         'gpt-4-turbo','gpt-4-turbo-2024-04-09', 'gpt-4-turbo-preview', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4',
#                         'gpt-4-0613', 'gpt-4-0314', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-instruct',
#                         'claude-3-5-sonnet-20240620','claude-3-haiku-20240307', 'claude-3-opus-20240229', 'claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229',]

# PROB_AVAILABLE_API_MODELS = ['gpt-4o', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06', 'chatgpt-4o-latest', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
#                         'gpt-4-turbo','gpt-4-turbo-2024-04-09', 'gpt-4-turbo-preview', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-4',
#                         'gpt-4-0613', 'gpt-4-0314', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-instruct']

# ACTIVATION_AVAILABLE_API_MODELS = []

UPSTREAM_DATASETS = ["trivia_qa", "gsm8k",
                     "natural_qa", "pop_qa", "simple_qa", "wikipedia_factual", "narrative_qa", "web_questions"]

# --- D axis: datasets added by the HC fork (benchmark protocol) -------------------
# Health is the gap upstream leaves open across every format; MCQ, extractive and
# multi-hop-general are the other three. Loaders live in utils/hc_datasets.py.
HEALTH_DATASETS = ["bioasq", "medqa", "mmlu_med", "kqa", "medlfqa"]
HC_GENERAL_DATASETS = ["hotpot_qa", "truthful_qa", "squad_v2"]

# Formats whose correctness evaluation differs from free-form QA. MCQ sets need
# evaluators.MCQMatch -- ExactMatch scores "B)" and "The answer is B" as wrong, which
# corrupts the labels the UQ score is graded against, not merely the reported accuracy.
MCQ_DATASETS = ["medqa", "mmlu_med"]
LONG_FORM_HEALTH_DATASETS = ["kqa", "medlfqa"]

AVAILABLE_DATASETS = UPSTREAM_DATASETS + HEALTH_DATASETS + HC_GENERAL_DATASETS
LONG_FORM_AVAILABLE_DATASETS = ["longfact_concepts", "longfact_objects"]

# --- V axis: evaluation metrics (benchmark protocol §4) --------------------------
# A. Discrimination -- rank-based, need no score->probability mapping. Primary comparison.
DISCRIMINATION_METRICS = ["auroc", "auprc", "auarc", "prr"]

# Threshold-based, computed at the 0.5 cut of the normalized truth value.
THRESHOLD_METRICS = ["accuracy", "f1", "precision", "recall"]

# B. Calibration error (§4B). These read normalized truth values as probabilities and
# therefore require a *fitted* normalizer -- metric_score raises otherwise.
CALIBRATION_METRICS = ["ece", "ace", "mce", "brier", "kde_ece", "classwise_ece"]

# C. Safety-weighted / selective prediction (§4C). "harm_recall" additionally requires
# per-item harm_labels, which are distinct from correctness labels by design (Q5).
SAFETY_METRICS = ["risk_at_coverage", "coverage_at_risk", "harm_recall"]

AVAILABLE_EVALUATION_METRICS = (
    DISCRIMINATION_METRICS + THRESHOLD_METRICS + CALIBRATION_METRICS + SAFETY_METRICS
)
