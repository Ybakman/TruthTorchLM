DEFAULT_TEMPLATE = "{context}"

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Give short and precise answers.'
DEFAULT_USER_PROMPT = 'Question: {question_context} Answer:'

PTRUE_SYSTEM_PROMPT = 'You are a helpful, respectful and honest question-answer evaluator. You will be given a question, some brainstormed ideas and a generated answer. Evaluate the generate answer as true or false considering the question and brainstormed ideas. Output "The generated answer is true" or "The generated answer is false".'
PTRUE_USER_PROMPT = 'Question:{question_context}\nHere are some ideas that were brainstormed:{ideas}\nGenerated answer:{generated_text}'
PTRUE_MODEL_OUTPUT = 'The generated answer is true'


GOOGLE_CHECK_QUERY_SYSTEM_PROMPT = 'You are a brilliant assistant.'
GOOGLE_CHECK_QUERY_USER_PROMPT = '''You are a query generator designed to help users verify a given claim using search engines. Your primary task is to generate a Python list of two effective and skeptical search engine queries. These queries should assist users in critically evaluating the factuality of a provided claim using search engines.
    You should only respond in format as described below (a Python list of queries). PLEASE STRICTLY FOLLOW THE FORMAT. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '['.
    [response format]: ['query1', 'query2']

    Here are three examples:
    question_context: "Who is the CEO of twitter?"
    claim: The CEO of twitter is Bill Gates.
    response: ["Who is the CEO of twitter?", "CEO Twitter"]

    question_context:
    claim: Michael Phelps is the most decorated Olympian of all time.
    response: ["Who is the most decorated Olympian of all time?", "Michael Phelps"]

    question_context: Who developed ChatGPT?
    claim: ChatGPT is created by Google.
    response: ["Who created ChatGPT?", "ChatGPT"]

    Now complete the following(ONLY RESPONSE IN A LIST FORMAT, DO NOT RETURN OTHER WORDS!!! START YOUR RESPONSE WITH '[' AND END WITH ']'):
    question_context: {question_context}
    claim: {input}
    response:'''

GOOGLE_CHECK_VERIFICATION_SYSTEM_PROMPT = 'You are a brilliant assistant.'
GOOGLE_CHECK_VERIFICATION_USER_PROMPT = '''You are given a piece of text and its question context if exist. Your task is to identify whether there are any factual errors within the text.
    When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences may be helpful. Some evidences may contradict to each other. You must be careful when using the evidences to judge the factuality of the given text.
    The response should be a dictionary with three keys - "reasoning", "factuality", "error", and "correction", which correspond to the reasoning, whether the given text is factual or not (Boolean - True or False), the factual error present in the text, and the corrected text.
    The following is the given text and its question context
    [question context]: {question_context}
    [text]: {claim}
    The following is the provided evidences
    [evidences]: {evidence}
    You should only respond in a python dictionary format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '{{'.
    [response format]: 
    {{
      "reasoning": "Why is the given text factual or non-factual? Be careful when you said something is non-factual. When you said something is non-factual, you must provide multiple evidences to support your decision.",
      "error": "None if the text is factual; otherwise, describe the error.",
      "correction": "The corrected text if there is an error.",
      "factuality": True if the given text is factual, False otherwise." 
    }}'''