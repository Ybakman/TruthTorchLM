# TruthTorchLM: A Comprehensive Library for Hallucination Detection in LLMs 


**TruthTorchLM** is an open-source library for hallucination detection in text generation models. It integrates state-of-the-art methods, provides benchmarking tools across various tasks, and facilitates seamless integration with popular frameworks like Huggingface and LiteLLM.

---

## Features

- **State-of-the-Art Methods**: Implementations of cutting-edge hallucination detection techniques.
- **Evaluation Tools**: Benchmark hallucination detection methods with various metrics.
- **Calibration**: Normalize and calibrate truth values for interpretable and comparable hallucination scores.
- **Integration**: Works seamlessly with Huggingface and LiteLLM for model interfacing.
- **Long Form Generation**: Adapts hallucination detection methods to long-form generations.
- **New Hallucination Detection Methods**: It provides an easy-to-use interface for the implementation of new hallucination detection methods.

---

## Installation

Install TruthTorchLM using pip:

```bash
pip install TruthTorchLM
```

---

## Simple Usage

### Setting Up a Model

Define your model and tokenizer using Huggingface or specify an API-based model:

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
api_model = "gpt-4o-mini"
```

### Generating Text with Truth Values

Generate text from a model with truth values using built-in detection methods:

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

Normalize truth values across different methods for consistent evaluation:

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

Evaluate truth methods with metrics like AUROC, PRR:

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

---

## Implemented Hallucination Detection Methods

- **LARS**: Learnable Response Scoring ([paper](https://arxiv.org/pdf/2406.11278)).
- **Confidence**: Average log probability of generated text.
- **SelfDetection**: Measures consistency by generating rephrased questions.

---

## Main Contributors 

Yavuz Faruk Bakman (ybakman@usc.edu)
Duygu Nur Yaldiz (yaldiz@usc.edu)
Sungmin Kang (kangsung@usc.edu)
Hayrettin Eren Yildiz (hayereyil@gmail.com)
Alperen Ozis (alperenozis@gmail.com)


---

## License

TruthTorchLM is released under the [MIT License](LICENSE).

For inquiries or support, reach out to the maintainers.
