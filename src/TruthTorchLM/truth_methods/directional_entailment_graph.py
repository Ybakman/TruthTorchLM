import torch
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from ..generation import sample_generations_hf_local, sample_generations_api
from .truth_method import TruthMethod

import numpy as np
import re
from scipy.linalg import eigvals


class DirectionalEntailmentGraph(TruthMethod):
    """
    Implements the Directed Entailment Graph approach for measuring uncertainty
    based on random walk Laplacian eigenvalues.
    """

    REQUIRES_SAMPLED_TEXT = True

    def __init__(
        self,
        number_of_generations=5,
        method_for_similarity: str = "jaccard",  # or "semantic"
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device: str = "cuda",
        temperature=0.7,
        batch_generation=True,
    ):
        """
        Arguments:
            number_of_generations (int): How many responses to generate.
            method_for_similarity (str): 'jaccard' or 'semantic'.
            model_for_entailment (PreTrainedModel): Any NLI model (DeBERTa, etc.).
            tokenizer_for_entailment (PreTrainedTokenizer): Tokenizer for the NLI model.
            entailment_model_device (str): GPU or CPU for the NLI model.
            temperature (float): Generation temperature for local HF sampling.
            batch_generation (bool): If True, generate in batch mode.
        """
        super().__init__()
        self.number_of_generations = number_of_generations
        self.method_for_similarity = method_for_similarity
        self.temperature = temperature
        self.batch_generation = batch_generation

        # Load default DeBERTa MNLI if user didn't specify
        if model_for_entailment is None or tokenizer_for_entailment is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            print(
                "No entailment model provided. Loading microsoft/deberta-large-mnli by default."
            )
            nli_model_name = "microsoft/deberta-large-mnli"
            tokenizer_for_entailment = AutoTokenizer.from_pretrained(
                nli_model_name)
            model_for_entailment = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name
            ).to(entailment_model_device)
            model_for_entailment.eval()

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment
        self.entailment_model_device = entailment_model_device

    def _compute_directional_entailment_uncertainty(
        self, sampled_generations_dict, question
    ):
        """
        Core logic that:
          1. Extracts the top N generated texts
          2. Builds an entailment matrix + text similarity matrix
          3. Constructs the adjacency matrix
          4. Computes the random walk Laplacian
          5. Calculates the final uncertainty measure
        Returns a dictionary with the final uncertainty and any additional info.
        """
        generated_texts = sampled_generations_dict["generated_texts"][
            : self.number_of_generations
        ]

        # STEP 1: Build Entailment Matrix
        entailment_matrix = self._build_entailment_matrix(generated_texts)

        # STEP 2: Build Similarity Matrix (Jaccard or semantic)
        similarity_matrix = self._build_similarity_matrix(generated_texts)

        # STEP 3: Combine -> Adjacency Matrix
        adjacency_matrix = entailment_matrix + similarity_matrix

        # STEP 4: Random Walk Laplacian
        laplacian = self._compute_random_walk_laplacian(adjacency_matrix)

        # STEP 5: Eigenvalues + Final Uncertainty
        uncertainty_value = self._compute_uncertainty(laplacian)

        output_dict = {
            "generated_texts": generated_texts,
            "directional_entailment_uncertainty": uncertainty_value,
            "truth_value": -uncertainty_value,
        }
        return output_dict

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        messages: list = [],
        context: str = "",
        **kwargs
    ):
        """
        Called by the pipeline for local Hugging Face generation usage.
        If `sampled_generations_dict` is not provided, generate text here.
        """
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_hf_local(
                model=model,
                input_text=input_text,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_generations,
                return_text=True,
                batch_generation=self.batch_generation,
                temperature=self.temperature,
                **kwargs
            )

        return self._compute_directional_entailment_uncertainty(
            sampled_generations_dict, question
        )

    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs
    ):
        """
        Called by the pipeline for API-based generation usage (e.g., OpenAI API).
        If `sampled_generations_dict` is not provided, generate text via sample_generations_api().
        """
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(
                model=model,
                messages=messages,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_generations,
                return_text=True,
                temperature=self.temperature,
                **kwargs
            )

        return self._compute_directional_entailment_uncertainty(
            sampled_generations_dict, question
        )

    # "Private" utility methods. We can put them in `utils` if they are going to be used somewhere else

    def _build_entailment_matrix(self, responses):
        """
        Build an n x n matrix of entailment probabilities
        using the loaded NLI model (self.model_for_entailment).
        """
        n = len(responses)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = self._compute_entailment_prob(
                        responses[i], responses[j]
                    )
        return matrix

    def _compute_entailment_prob(self, r1, r2):
        """
        Use the loaded NLI model & tokenizer to get entailment probability.
        Expects roberta/deberta style [contradiction, neutral, entailment].
        """
        inputs = self.tokenizer_for_entailment.encode_plus(
            r1, r2, return_tensors="pt", truncation=True
        ).to(self.entailment_model_device)
        with torch.no_grad():
            logits = self.model_for_entailment(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        entailment_prob = probs[0, 2].item()
        return entailment_prob

    def _build_similarity_matrix(self, responses):
        """
        Build an n x n matrix of text similarity.
        If method_for_similarity == 'jaccard', compute Jaccard.
        Otherwise, you can implement a semantic approach.
        """
        n = len(responses)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # if i == j:
                #     matrix[i, j] = 1.0
                # else:
                if self.method_for_similarity.lower() == "jaccard":
                    matrix[i, j] = self._jaccard_similarity(
                        responses[i], responses[j])
                # else:
                #     # Fall back to a placeholder if we want to implement semantic or something
                #     matrix[i, j] = 0.0
        return matrix

    def _jaccard_similarity(self, r1, r2):
        words_r1 = set(self._tokenize(r1))
        words_r2 = set(self._tokenize(r2))
        intersection = words_r1.intersection(words_r2)
        union = words_r1.union(words_r2)
        return float(len(intersection)) / float(len(union)) if union else 0.0

    def _tokenize(self, text):
        cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
        return cleaned_text.split()

    def _compute_random_walk_laplacian(self, adjacency_matrix):
        out_degree = np.sum(adjacency_matrix, axis=1)
        epsilon = 1e-6
        D_inv = np.diag(1.0 / (out_degree + epsilon))
        laplacian = np.eye(len(adjacency_matrix)) - D_inv @ adjacency_matrix
        return laplacian

    def _compute_uncertainty(self, laplacian):
        eigenvalues = eigvals(laplacian)
        real_eigenvalues = [abs(ev.real) for ev in eigenvalues]
        # Summation of max(0, 1 - λ_k)
        uncertainty = sum(max(0, 1 - ev) for ev in real_eigenvalues)
        return uncertainty
