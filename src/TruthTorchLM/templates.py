DEFAULT_TEMPLATE = "{context}"

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. Give short and precise answers.'

DEFAULT_SYSTEM_BENCHMARK_PROMPT = 'You are a helpful assistant. Answer the following question in a single brief but complete sentence.'

DEFAULT_USER_PROMPT = 'Question: {question_context} Answer:'

PTRUE_SYSTEM_PROMPT = 'You are a helpful, respectful and honest question-answer evaluator. You will be given a question, some brainstormed ideas and a generated answer. Evaluate the generate answer as true or false considering the question and brainstormed ideas. Output "The generated answer is true" or "The generated answer is false".'
PTRUE_USER_PROMPT = 'Question:{question_context}\nHere are some ideas that were brainstormed:{ideas}\nGenerated answer:{generated_text}'
PTRUE_MODEL_OUTPUT = 'The generated answer is true'


DEFAULT_JUDGE_SYSTEM_PROMPT = '''You are a question answer evaluator.'''


SELF_DETECTION_QUESTION_PROMPT = "Given a question, paraphrase it to have different words and expressions but have the same meaning as the original question. Please note that you should not answer the question, but rather provide a re-phrased. Question: {question}"
SELF_DETECTION_SYSTEM_PROMPT = 'You are a helpful assistant. Give short and precise answers.'

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


ENTAILMENT_PROMPT  = "Determine whether the answer {seq1} is Contradicted or Same with the answer {seq2} for the question {context}. You need to check whether the two answers exactly describe the same thing such as the same entity, digit, or arithmetical results. If the two answers are the same, give 'Same', otherwise give 'Contradicted' as the result."


DEFAULT_JUDGE_PROMPT = """
Your job is to look at a question, gold targets, and a predicted answer, and then assign a
grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.
The following are examples of CORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I’m not sure and would have to
double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha
Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born
on July 4, 1998, and Sasha was born on June 10, 2001.
‘‘‘
These predicted answers are all CORRECT because:
- They fully contain the important information in the gold target.
- They do not contain any information that contradicts the gold target.
- Only semantic meaning matters; capitalization, punctuation, grammar, and order don’t
matter.
- Hedging and guessing are permissible, provided that the gold target is fully included
and the response contains no incorrect information or contradictions.
The following are examples of INCORRECT predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it’s either Malia and Sasha. Or it could be Malia and Jackie. Or
it could be Joey and Malia.
Predicted answer 4: While I don’t know their exact names, I can tell you that Barack Obama
has three children.
Predicted answer 5: It’s possible you may mean Betsy and Olivia. However, you should clarify
further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama’s child is named James. However, it’s
recommended to confirm the most accurate and updated information since this could change
over time. This model may not always reflect the most current information.
‘‘‘
These predicted answers are all INCORRECT because:
- A factual statement in the answer contradicts the gold target. Incorrect statements
that have some hedging (e.g., "it is possible that", "although i’m not sure, i think
") are also considered incorrect.
The following are examples of NOT_ATTEMPTED predicted answers.
‘‘‘
Question: What are the names of Barack Obama’s children?
Gold target: Malia and Sasha
Predicted answer 1: I don’t know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I
can tell you that Barack Obama has two children.
11
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I’m
not sure about the other one.
‘‘‘
These predicted answers are all NOT_ATTEMPTED because:
- The important information in the gold target is not included in the answer.
- No statements in the answer contradict the gold target.
Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be
correct to the last significant figure in the gold answer. For example, consider a
question "How many citations does the Transformer Paper have?" with gold target "120k".
- Predicted answers "120k", "124k", and 115k" are all CORRECT.
- Predicted answers "100k" and "113k" are INCORRECT.
- Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED
because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the
predicted answer only needs to contain the information that is in the question.
- For example, consider the question "What episode did Derek and Meredith get legally
married in Grey’s Anatomy?" with gold target "Season 7, Episode 20: White Wedding".
Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer
.
- Do not punish predicted answers if they omit information that would be clearly inferred
from the question.
- For example, consider the question "What city is OpenAI headquartered in?" and the gold
target "San Francisco, California". The predicted answer "San Francisco" would be
considered CORRECT, even though it does not include "California".
- Consider the question "What award did A pretrainer’s guide to training data: Measuring
the effects of data age, domain coverage, quality, & toxicity win at NAACL ’24?", the
gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper"
would be considered CORRECT, because "award" is presumed in the question.
- For the question "What is the height of Jason Wei in meters?", the gold target is "1.73
m". The predicted answer "1.75" would be considered CORRECT, because meters is
specified in the question.
- For the question "What is the name of Barack Obama’s wife?", the gold target is "
Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because
the last name can be presumed.
- Do not punish for typos in people’s name if it’s clearly the same name.
- For example, if the gold target is "Hyung Won Chung", you can consider the following
predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won
Chung".
Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don’t
apologize or correct yourself if there was a mistake; we are just trying to grade the
answer.
‘‘‘
Question: {question}
Gold target: {ground_truths}
Predicted answer: {answer}
‘‘‘
Grade the predicted answer of this new question as one of:
CORRECT
INCORRECT
NOT_ATTEMPTED
Just return one the following words: 'CORRECT', 'INCORRECT', or 'NOT_ATTEMPTED'.
""".strip()

LFG_QUESTION_GENERATION_PROMPT = [{"role": "system", "content": 
'''You are an expert assistant skilled at generating focused and contextually relevant questions from statements. \
Your task is to create a question such that the answer would align closely with the provided statement. To ensure \
the question is precise and relevant, consider the context provided by the original question. Study the examples \
below from a variety of topics and follow the same pattern.

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Statement: George Orwell's novel 1984 explores the theme of government surveillance. 
Question: What theme does George Orwell's novel 1984 explore?   

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Statement: George Orwell's novel 1984 portrays a totalitarian regime that monitors every aspect of citizens' lives. 
Question: How does George Orwell's novel 1984 reflect the theme of totalitarian control, as commonly explored in 20th-century dystopian literature?

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Statement: The novel 1984 is written by George Orwell.
Question: Who has written the novel 1984?

Original Question: How has artificial intelligence influenced industries in the 21st century?
Statement: Artificial intelligence enables better decision-making through data analysis.
Question: How does artificial intelligence enhance the decision-making process in modern businesses?

Original Question: What factors contributed to the Great Depression, and how did governments respond?
Statement: Stock market speculation contributed to the Great Depression.
Question: Did stock market speculation contribute to the Great Depression?

Original Question: Who is Abraham Lincoln?
Statement: Abraham Lincoln is best known for leading the country through the Civil War.
Question: What is Abraham Lincoln's most significant historical contribution?

Original Question: Who is Abraham Lincoln?
Statement: Abraham Lincoln served from 1861 to 1865 as the president of he US.
Question: When did Abraham Lincoln serve as the president of the United States?

Now, follow the pattern demonstrated in the examples to generate a question for the given statement,\
without adding explanations, introductions, or conversational responses.'''},
{"role": "user", "content": 
'''Original question: {question_context}
Statement: {statement}
Question: '''}]

LFG_DECOMPOSITION_PROMPT = [{"role": "system", "content": 
'''You are a helpful assistant. List the specific factual propositions included in the given input as a python list. \
Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a list, \
without adding explanations, introductions, or conversational responses. Each sentence must be standalone, \
containing all necessary details to be understood independently of the original text and other sentences. \
This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. \
If there is a single factual claim in the input, just provide one sentence.

Examples: 

Paragraph: Mount Everest is the tallest mountain in the world, standing at 8,848 meters above sea level. \
It is located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China. \
The first successful ascent of Mount Everest was achieved in 1953 by Sir Edmund Hillary and Tenzing Norgay.
Propositions: 
['Mount Everest is the tallest mountain in the world.',
 'Mount Everest stands at 8,848 meters above sea level.',
 'Mount Everest is located in the Himalayas.',
 'Mount Everest is on the border between Nepal and the Tibet Autonomous Region of China.',
 'The first successful ascent of Mount Everest was achieved in 1953.',
 'Sir Edmund Hillary and Tenzing Norgay achieved the first successful ascent of Mount Everest.']

Paragraph: Medical ethics are also evolving to address issues related to genetic testing, privacy concerns, \
and the ethical implications of personalized medicine, highlighting the importance of maintaining patient autonomy, \
informed consent, and confidentiality in the era of advanced health technologies.
Propositions: 
['Medical ethics are evolving to address issues related to genetic testing.',
 'Medical ethics are evolving to address privacy concerns.',
 'Medical ethics are evolving to address the ethical implications of personalized medicine.',
 'Maintaining patient autonomy is important in the era of advanced health technologies.',
 'Informed consent is important in the era of advanced health technologies.',
 'Confidentiality is important in the era of advanced health technologies.']

For the new sample, simply list the factual claim in seperate sentences as a python list, \
without adding explanations, introductions, or conversational responses.'''},
{"role": "user", "content": 
 '''Paragraph: {TEXT}
 Propositions:'''}]