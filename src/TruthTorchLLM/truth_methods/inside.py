import copy
import torch
from typing import Union

from .truth_method import TruthMethod
from TruthTorchLLM.utils import sigmoid_normalization
from TruthTorchLLM.availability import ACTIVATION_AVAILABLE_API_MODELS 

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

class Inside(TruthMethod):
    def __init__(self, threshold:float=0.0, std:float=1.0, number_of_generations: int = 10, alpha:float = 0.001): 
        super().__init__(threshold = threshold, std = std)
        self.number_of_generations = number_of_generations
        self.alpha = alpha

    def generate_forward(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, **kwargs):
        super().generate_forward(model, input_text, generated_text, question_context, all_ids, generation_seed=generation_seed)
        kwargs = copy.deepcopy(kwargs)
        generated_texts = []
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        sentence_embeddings = torch.Tensor().to(model.device)
        for i in range(self.number_of_generations):
            
            model_output = model.generate(input_ids, num_return_sequences = 1, do_sample = True, **kwargs)
            
            tokens = model_output[0][len(input_ids[0]):]
            generated_texts.append(tokenizer.decode(tokens))

            with torch.no_grad():
                outputs = model(model_output, output_hidden_states=True)
                layer_activations = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
                sentence_embedding = layer_activations[0, -2:-1, :]

            sentence_embeddings = torch.cat((sentence_embeddings, sentence_embedding), dim=0)

        hidden_dim = sentence_embeddings.shape[-1]
        centering_matrix = torch.eye(hidden_dim) - (torch.ones((hidden_dim, hidden_dim)) / hidden_dim)
        centering_matrix = centering_matrix.to(model.device)

        covariance = sentence_embeddings @ centering_matrix @ sentence_embeddings.T
        regularized_covarience = covariance + torch.eye(self.number_of_generations).to(model.device) * self.alpha

        eigenvalues, _ = torch.linalg.eig(regularized_covarience)
        eigenvalues = eigenvalues.real

        eigen_score = torch.mean(torch.log(eigenvalues)).cpu().item()


        normalized_truth_value = sigmoid_normalization(-eigen_score, self.threshold, self.std)
        return {"truth_value": -eigen_score, 'normalized_truth_value':normalized_truth_value,  'generated_texts_for_inside': generated_texts}#this output format should be same for all truth methods

    def completion_forward(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, **kwargs):
        super().completion_forward(model, messages, generated_text, question_context, generation_seed=generation_seed)

        if not model in ACTIVATION_AVAILABLE_API_MODELS:
            raise ValueError("Inside method cannot be used with black-box API models since it requires access to activations.")

        return {"truth_value": 0, 'normalized_truth_value':0,  'generated_texts_for_inside': []}#this output format should be same for all truth methods

    def __str__(self):
        return "Inside Truth Method with " + str(self.number_of_generations) + " generations. Threshold: " + str(self.threshold) + ". Standard Deviation: " + str(self.std)

    