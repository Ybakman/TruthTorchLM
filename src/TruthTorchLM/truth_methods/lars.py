import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Union
from datasets import  Dataset
from sklearn.model_selection import train_test_split

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import DebertaForSequenceClassification, DebertaTokenizer

from .truth_method import TruthMethod
from TruthTorchLM.utils import bidirectional_entailment_clustering
from TruthTorchLM.availability import PROB_AVAILABLE_API_MODELS
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from .semantic_entropy import calculate_total_log

from ..evaluators.correctness_evaluator import CorrectnessEvaluator 
from TruthTorchLM.utils.dataset_utils import get_dataset
from ..generation import sample_generations_hf_local, sample_generations_api, sample_generations_batch_hf_local, sample_generations_sequential_hf_local
from TruthTorchLM.utils.eval_utils import metric_score
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat

class LARS(TruthMethod):

    REQUIRES_LOGPROBS = True

    def __init__(self, 
                        device="cuda", lars_model:PreTrainedModel=None, lars_tokenizer:PreTrainedTokenizer=None, 
                        ue_type:str="confidence", number_of_generations:int=0,
                        model_for_entailment: PreTrainedModel = None, tokenizer_for_entailment: PreTrainedTokenizer = None, entailment_model_device = 'cuda'):
        super().__init__()

        assert ue_type in ["confidence", "semantic_entropy", "se", "entropy"], f"ue_type must be one of ['confidence', 'semantic_entropy', 'se', 'entropy'] but it is {ue_type}."
        self.ue_type = ue_type
        self.number_of_generations = number_of_generations #number of generations for semantic entropy and entropy
        
        #lars model
        if lars_model is None or lars_tokenizer is None:
            lars_model = AutoModelForSequenceClassification.from_pretrained("duygunuryldz/LARS").to(device) #TODO
            lars_tokenizer = AutoTokenizer.from_pretrained("duygunuryldz/LARS") #TODO 
        self.lars_model = lars_model
        self.lars_tokenizer = lars_tokenizer
        self.device = device

        #lars params
        self.number_of_bins = lars_model.config.number_of_bins #number of bins for discretization of the probability space 
        self.edges = lars_model.config.edges #edges of bins, discretization of the probability space 

        #params for semantic entropy 
        if (ue_type == "se" or ue_type == "semantic_entropy") and (model_for_entailment is None or tokenizer_for_entailment is None):
            model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment


    @staticmethod
    def _find_bin(value, edges, number_of_bins):
        if edges is not None:
            bin_index = np.digitize(value, edges, right=False)
        else:
            bin_index = int(value*number_of_bins) #discretize the probability space equally
        return min(bin_index, (number_of_bins-1))
    
    @staticmethod
    def prepare_answer_text(probs, answer_tokens, edges, number_of_bins):
        a_text = ""
        assert len(probs) == len(answer_tokens)
        for i, tkn_text in enumerate(answer_tokens):
            bin_id = LARS._find_bin(probs[i], edges, number_of_bins)
            a_text += tkn_text+f"[prob_token_{bin_id}]"
        return a_text

    @staticmethod
    def tokenize_input(tokenizer, question, answer_text): 

        tokenized_input = tokenizer(
            question, answer_text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            is_split_into_words=False,  #???
            truncation=True,
            max_length = None, 
            padding='max_length'
        )
        return tokenized_input

    def _lars(self, question, generation_token_texts, probs):

        a_text = LARS.prepare_answer_text(probs, generation_token_texts, self.edges, self.number_of_bins)
        tokenized_input = LARS.tokenize_input(self.lars_tokenizer, question, a_text)

        input_ids = torch.tensor(tokenized_input['input_ids']).reshape(1,-1).to(self.device)
        attention_mask = torch.tensor(tokenized_input['attention_mask']).reshape(1,-1).to(self.device)
        token_type_ids = torch.tensor(tokenized_input['token_type_ids']).reshape(1,-1).to(self.device)

        self.lars_model.eval()
        logits = self.lars_model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits.detach()
            
        return torch.nn.functional.sigmoid(logits[:,  0]).item()


    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs):

        if self.ue_type == "confidence":
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
            model_output = all_ids.to(model.device)
            tokens = model_output[0][len(input_ids[0]):]
            tokens_text = [tokenizer.decode([token]) for token in tokens]

            # tokens_text = tokenizer.convert_ids_to_tokens(tokens)
            # tokens_text = [s.replace(space_char, " ") for s in tokens_text]  #requires space_char for each model

            with torch.no_grad():
                outputs = model(model_output)
                logits = outputs.logits  # Logits for each token in the input

                # Calculate probabilities from logits
                probs = torch.nn.functional.softmax(logits, dim=-1)#probs for each token
                probs = probs[0, len(input_ids[0])-1:-1, :]#probs for each token in the generated text
                probs = torch.gather(probs, dim=1, index = model_output[0][len(input_ids[0]):].view(-1, 1))#probs for each token in the generated text
                probs = probs.view(-1).tolist()#convert to list

                lars_score = self._lars(question_context, tokens_text, probs)

        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            if sampled_generations_dict is None:
                sampled_generations_dict = sample_generations_hf_local(model = model, input_text = input_text, tokenizer = tokenizer, generation_seed=generation_seed, 
                                                                        number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, batch_generation=self.batch_generation, **kwargs)
            scores = []
            generated_outputs = []
            generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        
            for i in range(self.number_of_generations):
                tokens_text = [tokenizer.decode([token]) for token in sampled_generations_dict["tokens"][i]]
                score = torch.log(self._lars(question_context, tokens_text, torch.exp(sampled_generations_dict["logprobs"][i])) )
                scores.append(score) #scores are in log scale
                generated_outputs.append((generated_texts[i],score))

            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, sampled_generations_dict["generated_texts"])
                lars_score = -calculate_total_log(generated_outputs,clusters)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}
            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts}

        return {"truth_value": lars_score,  "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format
    

    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, logprobs:list=None, generated_tokens:list=None, **kwargs):

        if not model in PROB_AVAILABLE_API_MODELS:
            raise ValueError("LARS method is not applicable to given model")

        if self.ue_type == "confidence":
            lars_score = self._lars(question_context, generated_tokens, torch.exp(torch.tensor(logprobs)))

        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            if sampled_generations_dict is None:
                sampled_generations_dict = sample_generations_api(model = model, messages = messages, generation_seed = generation_seed, 
                                                                    number_of_generations=self.number_of_generations, return_text = True, return_logprobs=True, **kwargs)
            scores = []
            generated_outputs = []
            generated_texts = sampled_generations_dict["generated_texts"][:self.number_of_generations]
        
            for i in range(self.number_of_generations):
                score = torch.log(self._lars(question_context, sampled_generations_dict["tokens"][i], torch.exp(sampled_generations_dict["logprobs"][i])) )
                scores.append(score) #scores are in log scale
                generated_outputs.append((generated_texts[i],score))

            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(self.model_for_entailment, self.tokenizer_for_entailment, question_context, sampled_generations_dict["generated_texts"])
                lars_score = -calculate_total_log(generated_outputs,clusters)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts, "clusters": clusters}
            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)
                return {"truth_value": lars_score,  "score_for_each_generation": scores, 'generated_texts': generated_texts}

        return {"truth_value": lars_score, "generated_text": generated_text}# we shouldn't return generated text. remove it from the output format

    @staticmethod
    def _get_datasets(datasets:list, size_for_each_dataset:list, val_ratio:float, seed:int):
        print("Creating train and validation datasets...")
        all_data = []
        for i, dataset in enumerate(datasets):
            all_data.append(get_dataset(dataset, size_of_data=size_for_each_dataset[i], seed=seed, split = "train"))

        all_data = sum(all_data,[]) #list of dict, each dict contains "question" and "ground_truths"
        train_data, val_data = train_test_split(all_data, test_size=val_ratio, random_state=seed)
        return train_data, val_data
    
    @staticmethod
    def _generate_answers_and_label(train_data:list, val_data:list, model:PreTrainedModel, tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                               correctness_evaluator:CorrectnessEvaluator, previous_context:list=[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}], 
                               user_prompt:str = DEFAULT_USER_PROMPT, num_gen_per_question:int=5, **kwargs):
        print("Generating answers and labels for training data...")
        for i in tqdm(range(len(train_data))):
            messages = previous_context.copy()
            messages.append({'role': 'user', 'content': user_prompt.format(question_context = train_data[i]['question'])})
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, continue_final_message=False)
            
            sampled = sample_generations_batch_hf_local(model=model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=num_gen_per_question-1, return_text=True, return_logprobs=True, return_model_output=False, **kwargs)

            most_likely = sample_generations_sequential_hf_local(model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=1, do_sample=False, return_text=True, return_logprobs=True, return_model_output=False, top_p=1, temperature=None, **kwargs)
            
            train_data[i]['generated_texts'] = most_likely['generated_texts'] 
            train_data[i]["probs"] = [np.exp(most_likely['logprobs'][0]).tolist()]
            train_data[i]["token_texts"] = [[tokenizer.decode(token) for token in most_likely['tokens'][0]]]
            for j in range(len(sampled['generated_texts'])):
                if sampled['generated_texts'][j] in train_data[i]['generated_texts']:
                    continue
                train_data[i]['generated_texts'].append(sampled['generated_texts'][j])
                train_data[i]["probs"].append(np.exp(sampled['logprobs'][j]).tolist())
                train_data[i]["token_texts"].append([tokenizer.decode([token]) for token in sampled['tokens'][j]])
            train_data[i]["labels"] = [correctness_evaluator(train_data[i]['question'], answer, train_data[i]['ground_truths']) for answer in train_data[i]['generated_texts']]

        print("Generating answers and labels for validation data...")
        for i in tqdm(range(len(val_data))):
            messages = previous_context.copy()
            messages.append({'role': 'user', 'content': user_prompt.format(question_context = val_data[i]['question'])})
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, continue_final_message=False)

            most_likely = sample_generations_sequential_hf_local(model, input_text=text, tokenizer=tokenizer,
                        number_of_generations=1, do_sample=False, return_text=True, return_logprobs=True, return_model_output=False, top_p=1, temperature=None, **kwargs)
            
            val_data[i]['generated_text'] = most_likely['generated_texts'][0]
            val_data[i]["probs"] = np.exp(most_likely['logprobs'][0]).tolist() 
            val_data[i]["token_texts"] = [tokenizer.decode([token]) for token in most_likely['tokens'][0]]
            val_data[i]["label"] = correctness_evaluator(val_data[i]['question'], most_likely['generated_texts'][0], val_data[i]['ground_truths'])

        return train_data, val_data
    
    @staticmethod
    def _prepare_model_and_tokenizer(model, tokenizer, number_of_bins, edges):

        print("Preparing LARS model and tokenizer...")
        #add new prob tokens
        prob_tokens = [f"[prob_token_{i}]" for i in range(number_of_bins)]
        num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': prob_tokens})
        print("Number of tokens added:", num_added_toks)
        model.resize_token_embeddings(len(tokenizer))

        #initialize embeddings of prob tokens
        embeddings = model.get_input_embeddings()  #
        num_ones = int(embeddings.weight.data.shape[1]/number_of_bins)
        scale = torch.sum(torch.abs(embeddings.weight.data))/embeddings.weight.data.shape[1]/embeddings.weight.data.shape[0]
        with torch.no_grad():
            for i in range(number_of_bins):
                idx = number_of_bins - i - 1
                vector = torch.zeros(embeddings.weight.data[0].shape)
                vector[num_ones*idx:num_ones*(idx+1)] = 1.0 * scale * number_of_bins
                embeddings.weight.data[-(i+1)] = vector

        model.config.edges = list(edges)
        model.config.number_of_bins = number_of_bins

    @staticmethod
    def _prepare_data(tokenizer, train_data:list, val_data:list, number_of_bins:int, edges:list):
    
        print("Preparing train data for LARS training...")
        all_data = []
        for d in tqdm(train_data):
            question = d["question"]
            for i in range(len(d["probs"])):
                if d["labels"][i] != -1:
                    ans_text = LARS.prepare_answer_text(d["probs"][i], d["token_texts"][i], edges, number_of_bins)
                    tokenized_input = LARS.tokenize_input(tokenizer, question, ans_text)
                    all_data.append(
                        { "label": d["labels"][i],
                        'input_ids': tokenized_input["input_ids"], 
                        'token_type_ids': tokenized_input["token_type_ids"], 
                        'attention_mask': tokenized_input["attention_mask"],
                        } 
                    )
        print("Preparing validation data for LARS training...")
        all_test_data = []
        for d in tqdm(val_data):
            ans_text = LARS.prepare_answer_text(d["probs"], d["token_texts"], edges, number_of_bins)
            tokenized_input = LARS.tokenize_input(tokenizer, d["question"], ans_text)
            if d["label"] != -1:
                all_test_data.append(
                        { "label": d["label"],
                        'input_ids': tokenized_input["input_ids"], 
                        'token_type_ids': tokenized_input["token_type_ids"], 
                        'attention_mask': tokenized_input["attention_mask"],
                        } 
                )
        return Dataset.from_list(all_data), Dataset.from_list(all_test_data)

    @staticmethod
    def _test_loss(test_data, model, metrics, device):
        model.eval()    
        losses = []
        scores = []
        labels = []
        cross_loss = torch.nn.BCEWithLogitsLoss()
        for i in range(len(test_data)):
            #test loss code
            label = torch.tensor(test_data[i]['label']).reshape(1,-1).to(device)
            input_ids = torch.tensor(test_data[i]['input_ids']).reshape(1,-1).to(device)
            attention_mask = torch.tensor(test_data[i]['attention_mask']).reshape(1,-1).to(device)
            token_type_ids = torch.tensor(test_data[i]['token_type_ids']).reshape(1,-1).to(device)

            logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits.detach()
            loss = cross_loss(logits[:,  :], label.reshape(-1, 1).float())
            scores.append(torch.nn.functional.sigmoid(logits[:,  0]).item())
            labels.append(label.item())
            losses.append(loss.item())

        losses = np.array(losses)
        metric_scores = metric_score(metrics, labels, scores, scores)

        return np.mean(losses), metric_scores
    
    @staticmethod
    def _train(train_dataset:Dataset, val_dataset:Dataset, model, tokenizer, save_path:str=None, device="cuda", 
            test_metrics:list[str]=['auroc'], main_metric:str='auroc', wandb_run=None,
            number_of_bins:int=8, epochs:int=3, lr:float=5e-6, batch_size:int=8, test_freq:int=100):
        
        expected_features = {'label', 'input_ids', 'token_type_ids', 'attention_mask'}
        assert expected_features.issubset(set(train_dataset.features.keys()))
        assert expected_features.issubset(set(val_dataset.features.keys()))
        assert main_metric in test_metrics

        print("LARS training started...")

        def custom_collate_fn(batch):
            # Convert lists to tensors and stack them
            labels = torch.stack([torch.tensor(item["label"]) for item in batch])
            inps = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
            types = torch.stack([torch.tensor(item["token_type_ids"]) for item in batch])
            masks = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])

            return labels, inps, types, masks

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                                collate_fn=custom_collate_fn, pin_memory=True, pin_memory_device=device)

        #Set loss and optimizer
        cross_loss = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.to(device)

        best_score = 0.0
        best_model = None

        with torch.no_grad():
            model.eval()
            tloss, metric_scores = LARS._test_loss(val_dataset, model, test_metrics, device)
            log = f"Test loss: {tloss:.2f}"
            for key, val in metric_scores.items():
                log += f"  | Test {key}: {val:.2f}"
            print(log)
            if wandb_run:
                wandb_run.log({"iter": 0, f"test_loss": tloss}.update(metric_scores))
            model.train()

        for epoch in range(epochs):
            model.train()
            train_loss, total_sample = 0, 0
            for iteration, (labels, input_ids, token_type_ids, attention_mask) in enumerate(train_loader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids).logits
                loss = cross_loss(logits, labels.reshape(-1, 1).float())

                loss.backward()

                #mask the embeddings of the prob tokens -- only works for roberta 
                grad_mask = torch.ones_like(model.roberta.embeddings.word_embeddings.weight.grad)
                grad_mask[-number_of_bins:] = 0.0
                model.roberta.embeddings.word_embeddings.weight.grad *= grad_mask
                    
                optimizer.step()
                train_loss += loss.item() * len(labels)
                total_sample += len(labels)

                if (iteration) % test_freq == 0:
                    model.eval()

                    with torch.no_grad():
                        tloss, metric_scores = LARS._test_loss(val_dataset, model, test_metrics, device)
                        log = f"epoch {epoch} | Train loss: {train_loss/total_sample:.2f}  | Test loss: {tloss:.2f}"
                        for key, val in metric_scores.items():
                            log += f"  | Test {key}: {val:.2f}"
                        print(log)
                    if wandb_run:
                        wandb_run.log({"iter":iteration,"train_loss": train_loss/total_sample, f"test_loss": tloss}.update(metric_scores))
                    
                    if  metric_scores[main_metric] > best_score:
                        best_model = copy.deepcopy(model.cpu())
                        best_score = metric_scores[main_metric]
                        model.to(device)
                        if save_path:
                            checkpoint = {
                                'model': best_model,
                                'tokenizer': tokenizer,
                                'epoch' : epoch,
                                'iter': iteration,
                            }
                            # Save the checkpoint
                            torch.save(checkpoint, save_path)

                    train_loss, total_sample = 0, 0
                    model.train()
        return best_model
    
    @staticmethod
    def train_lars_model(datasets:list, size_for_each_dataset:list, val_ratio:float, seed:int, chat_model_name:str, 
                        correctness_evaluator, save_path:str=None, wandb_run=None,
                        previous_context:list=[{'role': 'system', 'content': DEFAULT_SYSTEM_BENCHMARK_PROMPT}],
                        user_prompt:str = DEFAULT_USER_PROMPT, num_gen_per_question:int=5, number_of_bins:int=8,
                        lars_model_name:str = "roberta-base", test_metrics:list[str]=['auroc'], main_metric:str='auroc',
                        epochs:int=3, lr:float=5e-6, batch_size:int=8, test_freq:int=100,
                        device='cuda', **kwargs):
        
        assert val_ratio > 0

        #get all datasets as a whole and split to train and val
        train_data, val_data = LARS._get_datasets(datasets, size_for_each_dataset, val_ratio, seed)

        #crete chat model to get answers
        chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype=torch.float16).to(device)
        chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
        pad_token_id = chat_tokenizer.pad_token_id
        if pad_token_id == None:
            pad_token_id = chat_model.config.eos_token_id
            chat_tokenizer.pad_token_id = pad_token_id

        #generate answers to the questions and get their labels
        train_data, val_data = LARS._generate_answers_and_label(train_data=train_data, val_data=val_data, model=chat_model, tokenizer=chat_tokenizer, 
                                correctness_evaluator=correctness_evaluator, previous_context=previous_context,
                                user_prompt=user_prompt, num_gen_per_question=num_gen_per_question, pad_token_id=pad_token_id, **kwargs)

        del chat_model
        del chat_tokenizer
        
        #Find edges
        all_probs = []
        for d in tqdm(train_data):
            for i in range(len(d["probs"])):
                all_probs += d["probs"][i]
        edges = np.quantile(all_probs, np.linspace(0, 1, number_of_bins))
        all_probs = None

        #create LARS model
        model = AutoModelForSequenceClassification.from_pretrained(lars_model_name, num_labels=1).to(device)
        tokenizer = AutoTokenizer.from_pretrained(lars_model_name) 

        LARS._prepare_model_and_tokenizer(model=model, tokenizer=tokenizer, number_of_bins=number_of_bins, edges=edges)

        #prepare data for LARS training, return train and val datasets
        train_dataset, val_dataset = LARS._prepare_data(tokenizer=tokenizer, train_data=train_data, val_data=val_data, 
                                            number_of_bins=number_of_bins, edges=edges)
        

        model = LARS._train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, tokenizer=tokenizer, 
                    save_path=save_path, device=device, test_metrics=test_metrics, main_metric=main_metric, wandb_run=wandb_run,
                    number_of_bins=number_of_bins, epochs=epochs, lr=lr, batch_size=batch_size, test_freq=test_freq)
        

        return model, tokenizer, train_data, val_data