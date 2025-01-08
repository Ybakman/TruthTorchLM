# TruthTorchLM: A Comprehensive Library for Hallucination Detection in LLMs  

**TruthTorchLM** is an open-source library designed to detect and mitigate hallucinations in text generation models. The library integrates state-of-the-art methods, offers comprehensive benchmarking tools across various tasks, and enables seamless integration with popular frameworks like Huggingface and LiteLLM.

---

## Features  

- **State-of-the-Art Methods**: Implementations of advanced hallucination detection techniques.
- **Integration**: Seamlessly works with Huggingface and LiteLLM. 
- **Evaluation Tools**: Benchmark hallucination detection methods using various metrics like AUROC, PRR, and Accuracy.  
- **Calibration**: Normalize and calibrate truth values for interpretable and comparable outputs.  
- **Long-Form Generation**: Adapts detection methods to detect hallucinations in long-form text generations effectively.  
- **Extendability**: Provides an intuitive interface for implementing new hallucination detection methods.  

---

## Installation  

Install TruthTorchLM using pip:  

```bash
pip install TruthTorchLM
```

---

## Quick Start  

### Setting Up a Model  

You can define your model and tokenizer using Huggingface or specify an API-based model:  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import TruthTorchLM as ttlm
import torch

# Huggingface model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    torch_dtype=torch.bfloat16
).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)

# API model
api_model = "gpt-4o"
```

### Generating Text with Truth Values  

TruthTorchLM generates messages with a truth value, indicating whether the model output is hallucinating or not. Various methods (called **truth methods**) can be used to detect hallucinations. Each method can have different algorithms and output ranges. Lower truth values generally suggest hallucinations. This functionality is mostly useful for short-form QA:
```python
# Define truth methods
lars = ttlm.truth_methods.LARS()
confidence = ttlm.truth_methods.Confidence()
self_detection = ttlm.truth_methods.SelfDetection(number_of_questions=5)
truth_methods = [lars, confidence, self_detection]
```
```python
# Define a chat history
chat = [{"role": "system", "content": "You are a helpful assistant. Give short and precise answers."},
        {"role": "user", "content": "What is the capital city of France?"}]
```
```python
# Generate text with truth values (Huggingface model)
output_hf_model = ttlm.generate_with_truth_value(
    model=model,
    tokenizer=tokenizer,
    messages=chat,
    truth_methods=truth_methods,
    max_new_tokens=100,
    temperature=0.7
)
# Generate text with truth values (API model)
output_api_model = ttlm.generate_with_truth_value(
    model=api_model,
    messages=chat,
    truth_methods=truth_methods
)
```

### Calibrating Truth Methods  
Truth values for different methods may not be directly comparable. Use the `calibrate_truth_method` function to normalize truth values to a common range for better interpretability

```python
model_judge = ttlm.evaluators.ModelJudge('gpt-4o-mini')
calibration_results = ttlm.calibrate_truth_method(
    dataset='trivia_qa',
    model=model,
    truth_methods=truth_methods,
    tokenizer=tokenizer,
    correctness_evaluator=model_judge,
    size_of_data=1000,
    max_new_tokens=64
)
```

### Evaluating Truth Methods  

We can evaluate the truth methods with the `evaluate_truth_method` function. We can define different evaluation metrics including AUROC, AUPRC, AUARC, Accuracy, F1, Precision, Recall, PRR:  

```python
results = ttlm.evaluate_truth_method(
    dataset='trivia_qa',
    model=model,
    truth_methods=truth_methods,
    eval_metrics=['auroc', 'prr'],
    tokenizer=tokenizer,
    size_of_data=1000,
    correctness_evaluator=model_judge,
    max_new_tokens=64
)
```

### Hallucination Detection in Long-Form Generation 

Assigning a single score for long text is neither practical nor useful. TruthTorchLM first decomposes the generated text into short, single-sentence statements and assigns truth values to these statements using statement check methods. The `long_form_generation_with_truth_value` function returns the generated text, decomposed statements, and their truth values.


```python
import TruthTorchLM.long_form_generation as LFG
from transformers import DebertaForSequenceClassification, DebertaTokenizer

#define a decomposition method that breaks the the long text into statements
decomposition_method = LFG.decomposition_methods.StructuredDecompositionAPI(model="gpt-4o-mini", decomposition_depth=1, instruction=ttlm.LFG_DECOMPOSITION_PROMPT) #Utilize API models to decompose text
# decomposition_method = LFG.decomposition_methods.StructuredDecompositionLocal(model, tokenizer, decomposition_depth=1, chat_template=DECOMPOSITION_PROMT) #Utilize HF models to decompose text

#entailment model is used by some truth methods and statement check methods
model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to('cuda:0')
tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')
```

```python
#define truth methods 
confidence = ttlm.truth_methods.Confidence()
lars = ttlm.truth_methods.LARS()

#define the statement check methods that applies truth methods
qa_generation = LFG.statement_check_methods.QuestionAnswerGeneration(model="gpt-4o-mini", tokenizer=None, num_questions=2, max_answer_trials=2,
                                                                     truth_methods=[confidence, lars], seed=0,
                                                                     instruction=ttlm.LFG_QUESTION_GENERATION_PROMPT, 
                                                                     first_statement_instruction=ttlm.LFG_QUESTION_GENERATION_PROMPT,
                                                                     entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment) #HF model and tokenizer can also be used, LM is used to generate question
#there are some statement check methods that are directly designed for this purpose, not utilizing truth methods
as_entailment = LFG.statement_check_methods.AnswerStatementEntailment( model="gpt-4o-mini", tokenizer=None, 
                                                                      num_questions=3, num_answers_per_question=2, 
                                                                      instruction=ttlm.LFG_QUESTION_GENERATION_PROMPT, 
                                                                      first_statement_instruction=ttlm.LFG_QUESTION_GENERATION_PROMPT,
                                                                      entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment) #HF model and tokenizer can also be used, LM is used to generate question
```

```python
#define a chat history
chat = [{"role": "system", "content": 'You are a helpful assistant. Give brief and precise answers.'},
        {"role": "user", "content": f'Who is Ryan Reynolds?'}]

#generate a message with a truth value, it's a wrapper fucntion for model.generate in Huggingface
output_hf_model = LFG.long_form_generation_with_truth_value(model=model, tokenizer=tokenizer, messages=chat, fact_decomp_method=decomposition_method, 
                                          stmt_check_methods=[qa_generation, as_entailment], generation_seed=0)

#generate a message with a truth value, it's a wrapper fucntion for litellm.completion in litellm
output_api_model = LFG.long_form_generation_with_truth_value(model="gpt-4o-mini", messages=chat, fact_decomp_method=decomposition_method, 
                                          stmt_check_methods=[qa_generation, as_entailment], generation_seed=0, seed=0)
```

### Evaluation of Truth Methods in Long-Form Generation

We can evaluate truth methods on long form generation by using `evaluate_truth_method_long_form` function. To obtain the correctness labels of the statements we follow [SAFE paper](https://arxiv.org/pdf/2403.18802). SAFE performs google search for each statement and assigns labels as supported, unsupported or irrelevant. We can define different evaluation metrics including AUROC, AUPRC, AUARC, Accuracy, F1, Precision, Recall, PRR. 

Note: Calibrating truth methods before running evaluation is recommended.

```python
#create safe object that assigns labels to the statements
safe = LFG.ClaimEvaluator(rater='gpt-4o-mini', tokenizer = None, max_steps = 2, max_retries = 2, num_searches = 2)

#Define metrics
sample_level_eval_metrics = ['f1'] #calculate metric over the statements of a question, then average across all the questions
dataset_level_eval_metrics = ['auroc', 'prr'] #calculate the metric across all statements 
```
```python
results = LFG.evaluate_truth_method_long_form(dataset='longfact_objects', model='gpt-4o-mini', tokenizer=None,
                                sample_level_eval_metrics=sample_level_eval_metrics, dataset_level_eval_metrics=dataset_level_eval_metrics,
                                fact_decomp_method=decomposition_method, stmt_check_methods=[qa_generation],
                                claim_evaluator = safe, size_of_data=3,  previous_context=[{'role': 'system', 'content': 'You are a helpful assistant. Give precise answers.'}], 
                                user_prompt="Question: {question_context}", seed=41,  return_method_details = False, return_calim_eval_details=False, wandb_run = None,  
                                add_generation_prompt = True, continue_final_message = False)
```
---

## Available Hallucination Detection Methods  

- **LARS**: [Do Not Design, Learn: A Trainable Scoring Function for Uncertainty Estimation in Generative LLMs](https://arxiv.org/pdf/2406.11278).  
- **Confidence**: [Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/pdf?id=jN5y-zb5Q7m). 
- **Entropy**:[Uncertainty Estimation in Autoregressive Structured Prediction](https://openreview.net/pdf?id=jN5y-zb5Q7m).  
- **SelfDetection**: [Knowing What LLMs DO NOT Know: A Simple Yet Effective Self-Detection Method](https://arxiv.org/pdf/2310.17918). 
- **AttentionScore**: [LLM-Check: Investigating Detection of Hallucinations in Large Language Models](https://openreview.net/pdf?id=LYx4w3CAgy). 
- **CrossExamination**: [LM vs LM: Detecting Factual Errors via Cross Examination](https://arxiv.org/pdf/2305.13281). 
- **EccentricityConfidence**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187). 
- **EccentricityUncertainty**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187).
- **GoogleSearchCheck**: [FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios](https://arxiv.org/pdf/2307.13528). 
- **Inside**: [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://openreview.net/pdf?id=Zj12nzlQbz). 
- **KernelLanguageEntropy**: [Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities](https://arxiv.org/pdf/2405.20003). 
- **MARS**: [MARS: Meaning-Aware Response Scoring for Uncertainty Estimation in Generative LLMs](https://aclanthology.org/2024.acl-long.419.pdf). 
- **MatrixDegreeConfidence**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187). 
- **MatrixDegreeUncertainty**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187).
- **MultiLLMCollab**: [Don’t Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration](https://arxiv.org/pdf/2402.00367). 
- **NumSemanticSetUncertainty**: [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/pdf/2302.09664). 
- **PTrue**: [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221). 
- **Saplma**: [The Internal State of an LLM Knows When It’s Lying](https://aclanthology.org/2023.findings-emnlp.68.pdf). 
- **SemanticEntropy**: [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/pdf/2302.09664).
- **sentSAR**: [Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models](https://aclanthology.org/2024.acl-long.276.pdf). 
- **SumEigenUncertainty**: [Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models](https://arxiv.org/pdf/2305.19187).
- **tokenSAR**: [Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models](https://aclanthology.org/2024.acl-long.276.pdf). 
- **VerbalizedConfidence**: [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://openreview.net/pdf?id=g3faCfrwm7). 

---

## Contributors  

- **Yavuz Faruk Bakman** (ybakman@usc.edu)  
- **Duygu Nur Yaldiz** (yaldiz@usc.edu)  
- **Sungmin Kang** (kangsung@usc.edu)  
- **Hayrettin Eren Yildiz** (hayereyil@gmail.com)  
- **Alperen Ozis** (alperenozis@gmail.com)  

---

## Citation  

If you use TruthTorchLM in your research, please cite:  

```bibtex
@misc{truthtorchlm2025,
  title={TruthTorchLM: A Comprehensive Library for Hallucination Detection in Large Language Models},
  author={Yavuz Faruk Bakman, Duygu Nur Yaldiz,Sungmin Kang, Hayrettin Eren Yildiz, Alperen Ozis},
  year={2025},
  howpublished={GitHub},
  url={https://github.com/Ybakman/TruthTorchLM}
}
```

---

## License  

TruthTorchLM is released under the [MIT License](LICENSE).  

For inquiries or support, feel free to contact the maintainers.

