# TruthTorchLM: A Comprehensive Library for Hallucination Detection in LLMs  

**TruthTorchLM** is an open-source library designed to detect and mitigate hallucinations in text generation models. The library integrates state-of-the-art methods, offers comprehensive benchmarking tools across various tasks, and enables seamless integration with popular frameworks like Huggingface and LiteLLM.

---

## Features  

- **State-of-the-Art Methods**: Implementations of advanced hallucination detection techniques.  
- **Evaluation Tools**: Benchmark hallucination detection methods using various metrics like AUROC, PRR, and Accuracy.  
- **Calibration**: Normalize and calibrate truth values for interpretable and comparable hallucination scores.  
- **Integration**: Seamlessly works with Huggingface and LiteLLM.  
- **Long-Form Generation**: Adapts detection methods to handle long-form text generations effectively.  
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

Normalize truth values for consistent evaluation across methods:  

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

Evaluate the performance of hallucination detection methods with metrics like AUROC and PRR:  

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

# Display evaluation results
for i, method in enumerate(results['output_dict']['truth_methods']):
    print(f"{method}: {results['eval_list'][i]}")
```

---

## Implemented Hallucination Detection Methods  

- **LARS**: Learnable Response Scoring ([paper](https://arxiv.org/pdf/2406.11278)).  
- **Confidence**: Measures the average log probability of the generated text.  
- **SelfDetection**: Assesses consistency by generating rephrased questions.  

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

