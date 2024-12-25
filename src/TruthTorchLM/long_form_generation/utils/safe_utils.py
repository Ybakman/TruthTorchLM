'''Code taken from SAFE https://github.com/google-deepmind/long-form-factuality/blob/main/eval/safe/query_serper.py and https://github.com/google-deepmind/long-form-factuality/blob/main/common/utils.py and adapted to TruthTorchLM'''

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
"""Class for querying the Google Serper API."""

import random
import time
from typing import Any, Optional, Literal, Union

import requests

_SERPER_URL = 'https://google.serper.dev'
NO_RESULT_MSG = 'No good Google Search result was found'


import copy
import io
import json
import os
import random
import re
import string
import types
from typing import Any

import termcolor

def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')

def clear_line() -> None:
  """Clears the current line."""
  print(' ' * os.get_terminal_size().columns, end='\r')

def print_color(message: str, color: str) -> None:
  """Prints a message with a color."""
  termcolor.cprint(message, color)

def maybe_print_error(
    message: Union[str, Exception, None],
    additional_info: str = '',
    verbose: bool = False
) -> None:
  """Prints the error message with additional info if flag is True."""
  if not strip_string(str(message)):
    return

  error = type(message).__name__ if isinstance(message, Exception) else 'ERROR'
  message = f'{error}: {str(message)}'
  message += f'\n{additional_info}' if verbose else ''
  clear_line()
  print_color(message, color='red')

def extract_first_code_block(
    input_string: str, ignore_language: bool = False
) -> str:
  """Extracts the contents of a string between the first code block (```)."""
  if ignore_language:
    pattern = re.compile(r'```(?:\w+\n)?(.*?)```', re.DOTALL)
  else:
    pattern = re.compile(r'```(.*?)```', re.DOTALL)

  match = pattern.search(input_string)
  return strip_string(match.group(1)) if match else ''

def extract_first_square_brackets(input_string: str) -> str:
  """Extracts the contents of the FIRST string between square brackets."""
  raw_result = re.findall(r'\[.*?\]', input_string, flags=re.DOTALL)

  if raw_result:
    return raw_result[0][1:-1]
  else:
    return ''

def maybe_print_error(
    message: Union[str, Exception, None],
    additional_info: str = '',
    verbose: bool = False
) -> None:
  """Prints the error message with additional info if flag is True."""
  if not strip_string(str(message)):
    return

  error = type(message).__name__ if isinstance(message, Exception) else 'ERROR'
  message = f'{error}: {str(message)}'
  message += f'\n{additional_info}' if verbose else ''
  clear_line()
  print_color(message, color='red')


class SerperAPI:
  """Class for querying the Google Serper API."""

  def __init__(
      self,
      gl: str = 'us',
      hl: str = 'en',
      k: int = 1,
      tbs: Optional[str] = None,
      search_type: Literal['news', 'search', 'places', 'images'] = 'search',
  ):
    self.serper_api_key = os.environ.get("SERPER_API_KEY", None)
    self.gl = gl
    self.hl = hl
    self.k = k
    self.tbs = tbs
    self.search_type = search_type
    self.result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

  def run(self, query: str, **kwargs: Any) -> str:
    """Run query through GoogleSearch and parse result."""
    assert self.serper_api_key, 'Missing serper_api_key.'
    results = self._google_serper_api_results(
        query,
        gl=self.gl,
        hl=self.hl,
        num=self.k,
        tbs=self.tbs,
        search_type=self.search_type,
        **kwargs,
    )
    return self._parse_results(results)

  def _google_serper_api_results(
      self,
      search_term: str,
      search_type: str = 'search',
      max_retries: int = 20,
      **kwargs: Any,
  ) -> dict[Any, Any]:
    """Run query through Google Serper."""
    headers = {
        'X-API-KEY': self.serper_api_key or '',
        'Content-Type': 'application/json',
    }
    params = {
        'q': search_term, "gl": 'us', "hl": 'en',
        **{key: value for key, value in kwargs.items() if value is not None},
    }
    response, num_fails, sleep_time = None, 0, 0

    while not response and num_fails < max_retries:
      try:
        response = requests.post(
            f'{_SERPER_URL}/{search_type}', headers=headers, params=params
        )
        # print(response)
      except AssertionError as e:
        raise e
      except Exception:  # pylint: disable=broad-exception-caught
        response = None
        num_fails += 1
        sleep_time = min(sleep_time * 2, 600)
        sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
        time.sleep(sleep_time)

    if not response:
      raise ValueError('Failed to get result from Google Serper API')

    response.raise_for_status()
    search_results = response.json()
    return search_results

  def _parse_snippets(self, results: dict[Any, Any]) -> list[str]:
    """Parse results."""
    snippets = []

    if results.get('answerBox'):
      answer_box = results.get('answerBox', {})
      answer = answer_box.get('answer')
      snippet = answer_box.get('snippet')
      snippet_highlighted = answer_box.get('snippetHighlighted')

      if answer and isinstance(answer, str):
        snippets.append(answer)
      if snippet and isinstance(snippet, str):
        snippets.append(snippet.replace('\n', ' '))
      if snippet_highlighted:
        snippets.append(snippet_highlighted)

    if results.get('knowledgeGraph'):
      kg = results.get('knowledgeGraph', {})
      title = kg.get('title')
      entity_type = kg.get('type')
      description = kg.get('description')

      if entity_type:
        snippets.append(f'{title}: {entity_type}.')

      if description:
        snippets.append(description)

      for attribute, value in kg.get('attributes', {}).items():
        snippets.append(f'{title} {attribute}: {value}.')

    result_key = self.result_key_for_type[self.search_type]

    if result_key in results:
      for result in results[result_key][:self.k]:
        if 'snippet' in result:
          snippets.append(result['snippet'])

        for attribute, value in result.get('attributes', {}).items():
          snippets.append(f'{attribute}: {value}.')

    if not snippets:
      return [NO_RESULT_MSG]

    return snippets

  def _parse_results(self, results: dict[Any, Any]) -> str:
    return ' '.join(self._parse_snippets(results))

