from .truth_method import TruthMethod
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.availability import AVAILABLE_API_MODELS
from litellm import completion
import copy
from typing import Union
import torch


#paper link: https://arxiv.org/abs/2402.00367

EXPERT_SYSTEM_PROMPT = "You are an expert of the given domain. Give helpful information about the question, based on the domain."
JUDGE_SYSTEM_PROMPT = "You are a judge assistant. Decide whether the proposed answer is true or false, given the question, answer, and feedbacks."
ALTERNATIVE_ANSWER_SYSTEM_PROMPT = "You are a helpful assistant. Give a possible alternative answer, given a question and an answer."

QUESTION = "Question: <question>"
ANSWER = "Answer: <answer>"
KNOWLEDGE = "Knowledge: <generated domain knowledge>"
DOMAINS = ['factual information', 'commonsense knowledge', 'mathematical knowledge']
KNOWLEDGE_COOP_SELF = "for domain in <domains>: Generate some knowledge about the question, focusing on <domain>:"
FEEDBACK_COOP_SELF = "Please review the proposed answer and provide feedback on its correctness."
JUDGE_COOP = "Answer only 'True' or 'False'. Based on the feedback, the proposed answer is: "
FEEDBACK_COOP_OTHERS = "Please review the proposed answer and provide feedback on its correctness."
ALTERNATIVE_COMPETE = "Please propose an alternative answer:"
KNOWLEDGE_COMPETE = "Generate a knowledge paragraph about <alternative answer>:"
DECISION_COMPETE = "Answer the question with the following knowledge: feel free to ignore irrelevant or wrong information."
JUDGE_COMPETE = "Answer only 'Same' or 'Different'. The two given answers are: "


class MultiLLMCollab(TruthMethod):
    REQUIRES_NORMALIZATION = False

    def __init__(self, collaborate_mode:str, feedback_models:list[Union[str, PreTrainedModel]],
                feedback_tokenizers:list[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]=None,
                 question_form: str='free_form', max_new_tokens=1024, temperature=1.0, top_k=50, num_beams=1, **generation_kwargs):
        super().__init__()
        
        if collaborate_mode not in ['coop_self', 'coop_others', 'compete']:
            raise ValueError("Collaboration type should be one of 'coop_self', 'coop_others', 'compete'")
        else:
            self.collaborate_mode = collaborate_mode
            
        if question_form not in ['multiple_choice', 'free_form']:
            raise ValueError("Question form should be one of 'multiple_choice', 'free_form'")
        else:
            self.question_form = question_form

        # Validate feedback models and tokenizers
        if collaborate_mode != 'coop_self':
            if not feedback_models:
                raise ValueError("Feedback models and tokenizers are required for 'coop_others' and 'compete' modes.")
        self.feedback_models = feedback_models
        self.feedback_tokenizers = feedback_tokenizers

        # Additional debug output for clarity
        print(f"Collaboration mode: {self.collaborate_mode}")
        
        self.feedback_models = feedback_models
        self.feedback_tokenizers = feedback_tokenizers 
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.num_beams = num_beams
        self.generation_kwargs = generation_kwargs
        

    def forward_hf_local(self, model:PreTrainedModel, input_text:str, generated_text:str, question_context:str, all_ids:Union[list, torch.Tensor], 
                         tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, generation_seed = None, sampled_generations_dict:dict = None, messages:list = [], **kwargs): 
        kwargs = copy.deepcopy(kwargs)
        
        # Get Decision - abstain or not
        if self.collaborate_mode == 'coop_self':
            generated_answer, abstain, _, feedbacks = self.coop_self_hf_local(question_context=question_context, generated_answer=generated_text,
                                                                      model=model, tokenizer=tokenizer,
                                                                      max_new_tokens=self.max_new_tokens, temperature=self.temperature, 
                                                                      top_k=self.top_k, num_beams=self.num_beams, **self.generation_kwargs)
        elif self.collaborate_mode == 'coop_others':
            generated_answer, abstain, feedbacks = self.coop_others_hf_local(question_context=question_context, generated_answer=generated_text,
                                                                     qa_model=model, qa_tokenizer=tokenizer, feedback_models=self.feedback_models,
                                                                     feedback_tokenizers=self.feedback_tokenizers,
                                                                     max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                                                                     top_k=self.top_k, num_beams=self.num_beams, **self.generation_kwargs)
        elif self.collaborate_mode == 'compete':
            generated_answer, abstain, _, feedbacks = self.compete_hf_local(question_context=question_context, generated_answer=generated_text,
                                                                    feedback_models=self.feedback_models, feedback_tokenizers=self.feedback_tokenizers,
                                                                    question_form=self.question_form, max_new_tokens=self.max_new_tokens, temperature=self.temperature, top_k=self.top_k, num_beams=self.num_beams, **self.generation_kwargs)
        return {"truth_value": 1.-float(abstain), "generated_text": generated_answer, "feedbacks": feedbacks}


    def forward_api(self, model:str, messages:list, generated_text:str, question_context:str, generation_seed = None, sampled_generations_dict:dict = None, **kwargs):
        if model not in AVAILABLE_API_MODELS:
            raise ValueError("This method is not applicable to given model")
        kwargs = copy.deepcopy(kwargs)
        
        # Get Decision - abstain or not
        if self.collaborate_mode == 'coop_self':
            generated_answer, abstain, _, feedbacks = self.coop_self_api(question_context=question_context, generated_answer=generated_text,
                                                                      model=model, temperature=self.temperature)
        elif self.collaborate_mode == 'coop_others':
            generated_answer, abstain, feedbacks = self.coop_others_api(question_context=question_context, generated_answer=generated_text,
                                                                     qa_model=model, feedback_models=self.feedback_models,
                                                                     temperature=self.temperature)
        elif self.collaborate_mode == 'compete':
            generated_answer, abstain, _, feedbacks = self.compete_api(question_context=question_context, generated_answer=generated_text,
                                                                    feedback_models=self.feedback_models,
                                                                    question_form=self.question_form, temperature=self.temperature)
        return {"truth_value": 1.-float(abstain), "generated_text": generated_answer, "feedbacks": feedbacks}

    def __str__(self):
        return f"Multi-LLM Collaboration Truth Method by {self.collaborate_mode}"
    

    # Coop-self method
    def coop_self_hf_local(self, question_context, generated_answer, model, tokenizer, max_new_tokens, temperature, top_k, num_beams, **kwargs):
        knolwedge_passages = []
        feedbacks = []
        abstain = True
        tokenizer.pad_token = tokenizer.eos_token

        # Generate knowledge passage and Feedback
        for domain in DOMAINS:
            expert_content = EXPERT_SYSTEM_PROMPT
            expert_content += KNOWLEDGE_COOP_SELF.replace("<domains>", str(DOMAINS)).replace("<domain>", domain) + "\n"
            expert_prompt = [{"role":"user", "content": expert_content}]

            # Generate knowledge passage
            text = tokenizer.apply_chat_template(expert_prompt, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            generated_knowledge_passage = tokenizer.decode(tokens, skip_special_tokens=False)
            knolwedge_passages.append(generated_knowledge_passage)
            
            expert_content = KNOWLEDGE.replace("<generated domain knowledge>", generated_knowledge_passage) + "\n"
            expert_content += QUESTION.replace('<question>', question_context) + "\n"
            expert_content += ANSWER.replace('<answer>', generated_answer) + "\n"
            expert_content += FEEDBACK_COOP_SELF

            expert_prompt = [{"role": "user", "content": expert_content}]
            text = tokenizer.apply_chat_template(expert_prompt, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            generated_feedback = tokenizer.decode(tokens, skip_special_tokens=False)
            feedbacks.append(generated_feedback)

        # Judging process
        judge_content = JUDGE_SYSTEM_PROMPT
        judge_content += QUESTION.replace('<question>', question_context) + "\n"
        judge_content += "Proposed " + ANSWER.replace('<answer>', generated_answer) + "\n"
        for i, feedback in enumerate(feedbacks):
            judge_content += f"Feedback {i+1}: {feedback}\n"
        judge_content += JUDGE_COOP
        
        # Judge prompt
        judge_prompt = [{"role": "user", "content": judge_content}]
        text = tokenizer.apply_chat_template(judge_prompt, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )
        tokens = model_output[0][len(input_ids[0]):]
        generated_final_answer = tokenizer.decode(tokens, skip_special_tokens=False)
        if "True" in generated_final_answer:
            abstain = False
        elif "False" in generated_final_answer:
            abstain = True
        return generated_answer, abstain, knolwedge_passages, feedbacks

    def coop_self_api(self, question_context, generated_answer, model, temperature):
        knolwedge_passages = []
        feedbacks = []
        abstain = True

        # Generate knowledge passage and Feedback
        for domain in DOMAINS:
            expert_content = EXPERT_SYSTEM_PROMPT
            expert_content += KNOWLEDGE_COOP_SELF.replace("<domains>", str(DOMAINS)).replace("<domain>", domain) + "\n"
            expert_prompt = [{"role":"user", "content": expert_content}]

            # Generate knowledge passage
            generated_knowledge_passage = completion(model = model, messages = expert_prompt, temperature=temperature) #, **kwargs)
            generated_knowledge_passage = generated_knowledge_passage.choices[0].message.content
            knolwedge_passages.append(generated_knowledge_passage)
            
            expert_content += KNOWLEDGE.replace("<generated domain knowledge>", str(generated_knowledge_passage)) + "\n"
            expert_content += QUESTION.replace('<question>', question_context) + "\n"
            expert_content += ANSWER.replace('<answer>', generated_answer) + "\n"
            expert_content += FEEDBACK_COOP_SELF

            expert_prompt = [{"role": "user", "content": expert_content}]
            generated_feedback = completion(model = model, messages = expert_prompt, temperature=temperature) #, **kwargs)
            generated_feedback = generated_feedback.choices[0].message.content
            feedbacks.append(generated_feedback)

        # Judging process
        judge_content = JUDGE_SYSTEM_PROMPT
        judge_content += QUESTION.replace('<question>', question_context) + "\n"
        judge_content += "Proposed " + ANSWER.replace('<answer>', generated_answer) + "\n"
        for i, feedback in enumerate(feedbacks):
            judge_content += f"Feedback {i+1}: {feedback}\n"
        judge_content += JUDGE_COOP
        
        # Judge prompt
        judge_prompt = [{"role": "user", "content": judge_content}]
        generated_final_answer = completion(model = model, messages = judge_prompt, temperature=temperature) #, **kwargs)
        generated_final_answer = generated_final_answer.choices[0].message.content
        if "True" in generated_final_answer:
            abstain = False
        elif "False" in generated_final_answer:
            abstain = True
        return generated_answer, abstain, knolwedge_passages, feedbacks


    # Coop-others method
    def coop_others_hf_local(self, question_context, generated_answer, qa_model, qa_tokenizer, feedback_models, feedback_tokenizers, max_new_tokens, temperature, top_k, num_beams, **kwargs):
        feedback_content = EXPERT_SYSTEM_PROMPT
        judge_content = JUDGE_SYSTEM_PROMPT
        feedbacks = []
        abstain = True
        qa_tokenizer.pad_token = qa_tokenizer.eos_token
        for tokenizer in feedback_tokenizers:
            tokenizer.pad_token = tokenizer.eos_token
        
        for i in range(len(feedback_models)):
            model = feedback_models[i]
            tokenizer = feedback_tokenizers[i]
            expert_content = feedback_content

            # Generating Feedbacks
            expert_content += QUESTION.replace('<question>', question_context)
            expert_content += ANSWER.replace('<answer>', generated_answer)
            expert_content += FEEDBACK_COOP_OTHERS
            expert_prompt = [{"role":"user", "content": expert_content}]
            text = tokenizer.apply_chat_template(expert_prompt, tokenize = False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            generated_feedback = tokenizer.decode(tokens, skip_special_tokens = False)
            feedbacks.append(generated_feedback)
            
        # Judging process
        model = qa_model
        tokenizer = qa_tokenizer
        judge_content += QUESTION.replace('<question>', question_context)
        judge_content += "Proposed " + ANSWER.replace('<answer>', generated_answer)
        for i, feedback in enumerate(feedbacks):
            judge_content += f"Feedback {i+1}:" + feedback
        judge_content += JUDGE_COOP
        judge_prompt = [{"role":"user", "content":judge_content}]
        text = tokenizer.apply_chat_template(judge_prompt, tokenize = False)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )
        tokens = model_output[0][len(input_ids[0]):]
        generated_final_answer = tokenizer.decode(tokens, skip_special_tokens = False)
        if "True" in generated_final_answer:
            abstain = False
        elif "False" in generated_final_answer:
            abstain = True
        return generated_answer, abstain, feedbacks
    
    def coop_others_api(self, question_context, generated_answer, qa_model, feedback_models, temperature):
        feedback_content = EXPERT_SYSTEM_PROMPT
        judge_content = JUDGE_SYSTEM_PROMPT
        feedbacks = []
        abstain = True
        
        for i in range(len(feedback_models)):
            model = feedback_models[i]
            expert_content = feedback_content
                
            # Generating Feedbacks
            expert_content += QUESTION.replace('<question>', question_context)
            expert_content += ANSWER.replace('<answer>', generated_answer)
            expert_content += FEEDBACK_COOP_OTHERS
            expert_prompt = [{"role":"user", "content": expert_content}]
            generated_feedback = completion(model = model, messages = expert_prompt, temperature=temperature)# , **kwargs)
            generated_feedback = generated_feedback.choices[0].message.content
            feedbacks.append(generated_feedback)

        # Judging process
        model = qa_model
        judge_content += QUESTION.replace('<question>', question_context)
        judge_content += "Proposed " + ANSWER.replace('<answer>', generated_answer)
        for i, feedback in enumerate(feedbacks):
            judge_content += f"Feedback {i+1}:" + feedback
        judge_content += JUDGE_COOP
        judge_prompt = [{"role":"user", "content":judge_content}]
        generated_final_answer = completion(model = model, messages = judge_prompt, temperature=temperature) # **kwargs)
        generated_final_answer = generated_final_answer.choices[0].message.content
        if "True" in generated_final_answer:
            abstain = False
        elif "False" in generated_final_answer:
            abstain = True
        return generated_answer, abstain, feedbacks    
    
    
    # Compete method
    def check_answers(self, initial_answer, new_answer):
        decision = False
        if initial_answer in new_answer or new_answer in initial_answer:
            decision = True
        return decision  
        
    def compete_hf_local(self, question_context, generated_answer, feedback_models, feedback_tokenizers, question_form, max_new_tokens, temperature, top_k, num_beams, **kwargs):
        system_prompt_content = ALTERNATIVE_ANSWER_SYSTEM_PROMPT
        alternative_answers = []
        knowledge_passages = []
        new_answers = []
        decisions = []
        for tokenizer in feedback_tokenizers:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        for i in range(len(feedback_models)):
            model = feedback_models[i]
            tokenizer = feedback_tokenizers[i]
            prompt_content = system_prompt_content
        
            # Generating Alternative Answers
            prompt_content += QUESTION.replace('<question>', question_context)
            prompt_content += ANSWER.replace('<answer>', generated_answer)
            prompt_content += ALTERNATIVE_COMPETE
            prompt = [{"role":"user", "content":prompt_content}]
            text = tokenizer.apply_chat_template(prompt, tokenize = False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            alternative_answer = tokenizer.decode(tokens, skip_special_tokens = False)
            alternative_answers.append(alternative_answer)
            
            # Generate Knowledge Passages
            prompt_content = QUESTION.replace('<question>', question_context)
            prompt_content += KNOWLEDGE_COMPETE.replace('<alternative answer>', alternative_answer)
            prompt = [{"role":"user", "content":prompt_content}]
            text = tokenizer.apply_chat_template(prompt, tokenize = False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            knowledge_passage = tokenizer.decode(tokens, skip_special_tokens = False)
            knowledge_passages.append(knowledge_passage)
            
            # Generate New answer
            prompt_content = DECISION_COMPETE
            prompt_content += KNOWLEDGE.replace('<generated domain knowledge>', knowledge_passage)
            prompt_content += QUESTION.replace('<question>', question_context)
            prompt_content += ANSWER.replace('<answer>', alternative_answer)
            prompt =[{"role":"user", "content":prompt_content}]
            text = tokenizer.apply_chat_template(prompt, tokenize = False)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            model_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs
            )
            tokens = model_output[0][len(input_ids[0]):]
            new_answer = tokenizer.decode(tokens, skip_special_tokens = False)
            new_answers.append(new_answer)
            
            if question_form == 'multiple_choice':
                # Check if the new answer is the same as the initial answer
                decision = self.check_answers(generated_answer, new_answer)
                decisions.append(decision)
                final_decision = sum(decisions) > len(decisions) / 2

            # Ask the model if the generated answer is the same answer as the new answer
            elif question_form == 'free_form':
                judge_prompt = JUDGE_SYSTEM_PROMPT
                judge_prompt += JUDGE_COMPETE
                judge_prompt += ANSWER.replace('<answer>', generated_answer)
                judge_prompt += ANSWER.replace('<answer>', new_answer)
                prompt = [{"role":"user", "content":judge_prompt}]
                text = tokenizer.apply_chat_template(prompt, tokenize = False)
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs['input_ids'].to(model.device)
                attention_mask = inputs['attention_mask'].to(model.device)
                model_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    num_beams=num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    **kwargs
                )
                tokens = model_output[0][len(input_ids[0]):]
                decision = tokenizer.decode(tokens, skip_special_tokens = False)
                abstain = False if "Same" in decision else True
                decisions.append(abstain)
                final_decision = sum(decisions) > len(decisions) / 2
        return generated_answer, final_decision, new_answers, knowledge_passages
    
    
    def compete_api(self, question_context, generated_answer, feedback_models, question_form, temperature):
        system_prompt_content = ALTERNATIVE_ANSWER_SYSTEM_PROMPT
        alternative_answers = []
        knowledge_passages = []
        new_answers = []
        decisions = []
        
        for i in range(len(feedback_models)):
            model = feedback_models[i]
            prompt_content = system_prompt_content
        
            # Generating Alternative Answers
            prompt_content += QUESTION.replace('<question>', question_context)
            prompt_content += ANSWER.replace('<answer>', generated_answer)
            prompt_content += ALTERNATIVE_COMPETE
            prompt = [{"role":"user", "content":prompt_content}]
            alternative_answer = completion(model = model, messages = prompt, temperature=temperature) #, **kwargs)
            alternative_answer = alternative_answer.choices[0].message.content
            alternative_answers.append(alternative_answer)
            
            # Generate Knowledge Passages
            prompt_content = QUESTION.replace('<question>', question_context)
            prompt_content += KNOWLEDGE_COMPETE.replace('<alternative answer>', alternative_answer)
            prompt = [{"role":"user", "content":prompt_content}]
            knowledge_passage = completion(model = model, messages = prompt, temperature=temperature)#, **kwargs)
            knowledge_passage = knowledge_passage.choices[0].message.content
            knowledge_passages.append(knowledge_passage)
            
            # Generate New answer
            prompt_content = DECISION_COMPETE
            prompt_content += KNOWLEDGE.replace('<generated domain knowledge>', knowledge_passage)
            prompt_content += QUESTION.replace('<question>', question_context)
            prompt_content += ANSWER.replace('<answer>', alternative_answer)
            prompt = [{"role":"user", "content":prompt_content}]
            new_answer = completion(model = model, messages = prompt, temperature=temperature) #, **kwargs)
            new_answer = new_answer.choices[0].message.content
            new_answers.append(new_answer)
            
            if question_form == 'multiple_choice':
                # Check if the new answer is the same as the initial answer
                decision = self.check_answers(generated_answer, new_answer)
                decisions.append(decision)
                final_decision = sum(decisions) > len(decisions) / 2
                
            # Ask the model if the generated answer is the same answer as the new answer
            elif question_form == 'free_form':
                judge_prompt = JUDGE_SYSTEM_PROMPT
                judge_prompt += JUDGE_COMPETE
                judge_prompt += ANSWER.replace('<answer>', generated_answer)
                judge_prompt += ANSWER.replace('<answer>', new_answer)
                prompt = [{"role":"user", "content":judge_prompt}]
                decision = completion(model = model, messages = prompt, temperature=temperature) #, **kwargs)
                decision = decision.choices[0].message.content
                abstain = False if "Same" in decision else True
                decisions.append(abstain)
                final_decision = sum(decisions) > len(decisions) / 2
        return generated_answer, final_decision, new_answers, knowledge_passages