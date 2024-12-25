'''Code taken from SAFE https://github.com/google-deepmind/long-form-factuality/tree/main/eval/safe and adapted to TruthTorchLM'''

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
from typing import Union
from typing import Any
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

import TruthTorchLM.long_form_generation.utils.safe_utils as utils
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat
from litellm import completion

SUPPORTED_LABEL = 'Supported'
NOT_SUPPORTED_LABEL = 'Not Supported'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""
_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "{SUPPORTED_LABEL}" or \
"{NOT_SUPPORTED_LABEL}". Wrap your final answer in square brackets.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

from abc import ABC

class ClaimEvaluator(ABC):
    
    def __init__(self, rater:Union[PreTrainedModel, str], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,
                       max_steps: int = 5, max_retries: int = 10, num_searches: int = 3):
        self.rater = rater
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.num_searches = num_searches

    def __call__(self, atomic_fact:str) -> dict:
        return check_atomic_fact(atomic_fact, self.rater, self.tokenizer, self.max_steps, self.max_retries, self.num_searches)
    
    def __str__(self):
        return "SAFE claim evaluator with rater: " + str(self.rater) + " max_steps: " + str(self.max_steps) + " max_retries: " + str(self.max_retries) + " num_searches: " + str(self.num_searches)
    

@dataclasses.dataclass()
class GoogleSearchResult:
  query: str
  result: str


@dataclasses.dataclass()
class FinalAnswer:
  response: str
  answer: str


def _generate(prompt, model, tokenizer, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    if type(model) == str:
        response = completion(
                                model=model,
                                messages=messages,
                                **kwargs
                            )
        generated_text = response.choices[0].message['content']
    else:
        tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True, continue_final_message=False)
        generated_text = generate(text, model, tokenizer, **kwargs)['generated_text_skip_specials']
    return generated_text


def call_search(
                    search_query: str,
                    num_searches: int = 3,
                    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
                ) -> str:
    """Call Google Search to get the search result."""
    search_query += f' {search_postamble}' if search_postamble else ''

    serper_searcher = utils.SerperAPI(k=num_searches)
    return serper_searcher.run(search_query, k=num_searches)


def maybe_get_next_search(
                                atomic_fact: str,
                                past_searches: list[GoogleSearchResult],
                                model:Union[PreTrainedModel, str], 
                                tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,
                                num_searches: int = 3,
                                search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
                                **kwargs
                            ) -> Union[GoogleSearchResult ,None]:
  """Get the next query from the model."""
  knowledge = '\n'.join([s.result for s in past_searches])
  knowledge = 'N/A' if not knowledge else knowledge
  full_prompt = _NEXT_SEARCH_FORMAT.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = utils.strip_string(full_prompt)
  model_response = _generate(full_prompt, model, tokenizer, **kwargs)
  query = utils.extract_first_code_block(model_response, ignore_language=True)
  # print(f'Search query: {query}')

  if model_response and query:
    return GoogleSearchResult(query=query, result=call_search(search_query=query, num_searches=num_searches, search_postamble=search_postamble))

  return None


def maybe_get_final_answer(
                                atomic_fact: str,
                                searches: list[GoogleSearchResult],
                                model:Union[PreTrainedModel, str], 
                                tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,
                                **kwargs
                            ) -> Union[FinalAnswer, None]:
  """Get the final answer from the model."""
  knowledge = '\n'.join([search.result for search in searches])
  full_prompt = _FINAL_ANSWER_FORMAT.replace(
      _STATEMENT_PLACEHOLDER, atomic_fact
  )
  full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
  full_prompt = utils.strip_string(full_prompt)
  model_response = _generate(full_prompt, model, tokenizer, **kwargs)
  answer = utils.extract_first_square_brackets(model_response)
  answer = re.sub(r'[^\w\s]', '', answer).strip()

  if model_response and answer in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
    return FinalAnswer(response=model_response, answer=answer)

  return None


def check_atomic_fact(
                            atomic_fact: str,
                            rater:Union[PreTrainedModel, str], 
                            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]= None,
                            max_steps: int = 5,
                            max_retries: int = 10,
                            num_searches: int = 3,
                            search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
                            **kwargs
                        ) -> tuple[Union[FinalAnswer, None], dict[str, Any]]:
  """Check if the given atomic fact is supported."""
  search_results = []

  for i in range(max_steps):
    next_search, num_tries = None, 0

    while not next_search and num_tries <= max_retries:
      # print(f'Step {i} Search trial #{num_tries}')s
      next_search = maybe_get_next_search(atomic_fact=atomic_fact, past_searches=search_results, model=rater, tokenizer=tokenizer,
                                            num_searches=num_searches, search_postamble=search_postamble, **kwargs)
      num_tries += 1

    if next_search is None:
      utils.maybe_print_error('Unsuccessful parsing for `next_search`')
      break
    else:
      search_results.append(next_search)

  search_dicts = {
      'google_searches': [dataclasses.asdict(s) for s in search_results]
  }
  final_answer, num_tries = None, 0

  while not final_answer and num_tries <= max_retries:
    num_tries += 1
    final_answer = maybe_get_final_answer(atomic_fact=atomic_fact, searches=search_results, model=rater, tokenizer=tokenizer, **kwargs)

  if final_answer is None:
    utils.maybe_print_error('Unsuccessful parsing for `final_answer`')
    return {"answer": None, "response":None, "search_details":None}

  return {"answer": final_answer.answer, "response":final_answer.response, "search_details":search_dicts}