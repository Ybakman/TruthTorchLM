
import torch
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, DebertaForSequenceClassification, DebertaTokenizer
from TruthTorchLM.utils import calculate_affinity_matrix, calculate_laplacian, create_kernel, calculate_VNE
from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
from .truth_method import TruthMethod
from ..generation import sample_generations_hf_local, sample_generations_api

class KernelLanguageEntropy(TruthMethod):
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = False
    
    def __init__(self, number_of_generations=5,
                 model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda',
                 kernel_type:str='heat', normalize_laplacian:bool=False, temperature=0.3, smoothness=1., scale=1.):
        super().__init__()
    
        if model_for_entailment is None or tokenizer_for_entailment is None:
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.number_of_generations = number_of_generations
        self.kernel_type = kernel_type
        self.normalize_laplacian = normalize_laplacian
        self.temperature = temperature
        self.smoothness = smoothness
        self.scale = scale
        
    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=False, **kwargs)

        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        
        # KLE part
        # semantic_graph = calculate_affinity_matrix(texts = generated_texts, context = question_context, method_for_similarity = 'kernel',
        #                                            model_for_entailment = self.model_for_entailment, tokenizer_for_entailment = self.tokenizer_for_entailment)
        # graph_laplacian = calculate_laplacian(graph = semantic_graph, normalize = self.normalize_laplacian)
        # kernel = create_kernel(laplacian = graph_laplacian, kernel_type = self.kernel_type, temperature=self.temperature, smoothness=self.smoothness, scale=self.scale)
        # kernel_entropy = calculate_VNE(kernel)
        semantic_graph, kernel, kernel_entropy = self.calculate_kernel_language_entropy(texts=generated_texts, context=question_context, method_for_similarity='kernel',
                                                    model_for_entailment = self.model_for_entailment, tokenizer_for_entailment = self.tokenizer_for_entailment,
                                                    normalize_laplacian=self.normalize_laplacian, kernel_type=self.kernel_type, 
                                                    temperature=self.temperature, smoothness=self.smoothness, scale=self.scale)
        
        return {'truth_value': -kernel_entropy, 'generated_texts': generated_texts, 'kernel': kernel, 'semantic_graph':semantic_graph}


    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):
        
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
            number_of_generations=self.number_of_generations, return_text = True, return_logprobs=False, **kwargs)
            
        generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]

        # # KLE part
        semantic_graph, kernel, kernel_entropy = self.calculate_kernel_language_entropy(texts=generated_texts, context=question_context, method_for_similarity='kernel',
                                                           model_for_entailment = self.model_for_entailment, tokenizer_for_entailment = self.tokenizer_for_entailment,
                                                           normalize_laplacian=self.normalize_laplacian, kernel_type=self.kernel_type, 
                                                           temperature=self.temperature, smoothness=self.smoothness, scale=self.scale)

        return {'truth_value': -kernel_entropy, 'generated_texts': generated_texts, 'kernel': kernel, 'semantic_graph':semantic_graph}

    def calculate_kernel_language_entropy(self, texts: list[str], context:str, method_for_similarity:str, model_for_entailment: PreTrainedModel,
                                        tokenizer_for_entailment: PreTrainedTokenizer, normalize_laplacian: bool, kernel_type: str,
                                        temperature:float, smoothness:float, scale:float):
        semantic_graph = calculate_affinity_matrix(texts = texts, context = context, method_for_similarity = method_for_similarity,
                                                model_for_entailment = model_for_entailment, tokenizer_for_entailment = tokenizer_for_entailment)
        graph_laplacian = calculate_laplacian(graph = semantic_graph, normalize = normalize_laplacian)
        kernel = create_kernel(laplacian = graph_laplacian, kernel_type = kernel_type, temperature=temperature, smoothness=smoothness, scale=scale)
        kernel_entropy = calculate_VNE(kernel)
        return semantic_graph, kernel, kernel_entropy