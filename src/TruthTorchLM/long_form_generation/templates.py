# For claim check methods

QUESTION_GENERATION_INSTRUCTION = [
    {
        "role": "system",
        "content": """You are an expert assistant skilled at generating focused and contextually relevant questions from claims. \
Your task is to create a question such that the answer would align closely with the provided claim. To ensure \
the question is precise and relevant, consider the context provided by the original question. Study the examples \
below from a variety of topics and follow the same pattern.

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Claim: George Orwell's novel 1984 explores the theme of government surveillance. 
Question: What theme does George Orwell's novel 1984 explore?   

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Claim: George Orwell's novel 1984 portrays a totalitarian regime that monitors every aspect of citizens' lives. 
Question: How does George Orwell's novel 1984 reflect the theme of totalitarian control, as commonly explored in 20th-century dystopian literature?

Original Question: What themes are commonly explored in 20th-century dystopian literature?
Claim: The novel 1984 is written by George Orwell.
Question: Who has written the novel 1984?

Original Question: How has artificial intelligence influenced industries in the 21st century?
Claim: Artificial intelligence enables better decision-making through data analysis.
Question: How does artificial intelligence enhance the decision-making process in modern businesses?

Original Question: What factors contributed to the Great Depression, and how did governments respond?
Claim: Stock market speculation contributed to the Great Depression.
Question: Did stock market speculation contribute to the Great Depression?

Original Question: Who is Abraham Lincoln?
Claim: Abraham Lincoln is best known for leading the country through the Civil War.
Question: What is Abraham Lincoln's most significant historical contribution?

Original Question: Who is Abraham Lincoln?
Claim: Abraham Lincoln served from 1861 to 1865 as the president of the US.
Question: When did Abraham Lincoln serve as the president of the United States?

Now, follow the pattern demonstrated in the examples to generate a question for the given claim,\
without adding explanations, introductions, or conversational responses.""",
    },
    {
        "role": "user",
        "content": """Original question: {question}
Claim: {claim}
Question: """,
    },
]

ANSWER_GENERATION_INSTRUCTION = [
    {
        "role": "system",
        "content": """You are a helpful assistant. Give a single claim answer to given question. \
Don\‘t provide any additional information. Just answer the question with a brief \
sentence in a single claim.""",
    },
    {"role": "user", "content": "{question}"},
]

# ANSWER_GENERATION_INSTRUCTION = [{"role": "system", "content": 'You are a helpful assistant. Give short and precise answers.'},
#                   {"role": "user", "content": "{question}"},]

# For decomposition methods

DECOMPOSITION_INSTRUCTION = [
    {
        "role": "system",
        "content": """You are a helpful assistant. List the specific factual claims included in the given input as a python list. \
Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a list, \
without adding explanations, introductions, or conversational responses. Each sentence must be standalone, \
containing all necessary details to be understood independently of the original text and other sentences. \
This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. \
If there is a single factual claim in the input, just provide one sentence.

Examples: 

Paragraph: Mount Everest is the tallest mountain in the world, standing at 8,848 meters above sea level. \
It is located in the Himalayas on the border between Nepal and the Tibet Autonomous Region of China. \
The first successful ascent of Mount Everest was achieved in 1953 by Sir Edmund Hillary and Tenzing Norgay. \
I hope you found these facts interesting! Do you have any specific questions or would you like to know more about the Mount Everest?
Claims: 
['Mount Everest is the tallest mountain in the world.',
 'Mount Everest stands at 8,848 meters above sea level.',
 'Mount Everest is located in the Himalayas.',
 'Mount Everest is on the border between Nepal and the Tibet Autonomous Region of China.',
 'The first successful ascent of Mount Everest was achieved in 1953.',
 'Sir Edmund Hillary and Tenzing Norgay achieved the first successful ascent of Mount Everest.']

Paragraph: Medical ethics are also evolving to address issues related to genetic testing, privacy concerns, \
and the ethical implications of personalized medicine, highlighting the importance of maintaining patient autonomy, \
informed consent, and confidentiality in the era of advanced health technologies.
Claims: 
['Medical ethics are evolving to address issues related to genetic testing.',
 'Medical ethics are evolving to address privacy concerns.',
 'Medical ethics are evolving to address the ethical implications of personalized medicine.',
 'Maintaining patient autonomy is important in the era of advanced health technologies.',
 'Informed consent is important in the era of advanced health technologies.',
 'Confidentiality is important in the era of advanced health technologies.']

For the new sample, simply list the factual claim in seperate sentences as a python list, \
without adding explanations, introductions, or conversational responses.""",
    },
    {
        "role": "user",
        "content": """Paragraph: {TEXT}
 Claims:""",
    },
]

DECOMPOSITION_INSTRUCTION_GRANULAR = [
    {
        "role": "system",
        "content": """You are a helpful assistant. List the specific factual claims included in the given input as a python list. \
Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence in a list, \
without adding explanations, introductions, or conversational responses. Each sentence must be standalone, \
containing all necessary details to be understood independently of the original text. \
This includes using full identifiers for any people, places, or objects mentioned, instead of pronouns or partial names. \
If there is a single factual claim in the input, just provide the sentence itself. If there is no factual claim in the input, \
provide an empty list.

Examples: 

Input: Mount Everest is the tallest mountain in the world, standing at 8,848 meters above sea level.
Claims: 
['Mount Everest is the tallest mountain in the world.',
 'Mount Everest stands at 8,848 meters above sea level.']

Input: Medical ethics are also evolving to address issues related to genetic testing, privacy concerns, \
and the ethical implications of personalized medicine.
Claims: 
['Medical ethics are evolving to address issues related to genetic testing.',
 'Medical ethics are evolving to address privacy concerns.',
 'Medical ethics are evolving to address the ethical implications of personalized medicine.']

Input: Abraham Lincoln was the 16th president of the United States.
Claims: 
['Abraham Lincoln was the 16th president of the United States.']

Input: I hope you found these facts interesting! Is there anything else you would like to know?
Claims: 
[]

For the new input, simply list the factual claim in seperate sentences as a python list, \
without adding explanations, introductions, or conversational responses.""",
    },
    {
        "role": "user",
        "content": """Input: {TEXT}
 Claims:""",
    },
]

UNSTRUCTURED_DECOMPOSITION_INSTRUCTION = [
    {
        "role": "system",
        "content": """You are a helpful assistant. List the specific factual claims included in the given input. \
Be complete and do not leave any factual claims out. Provide each factual claim as a separate sentence \
in a separate bullet point, without adding explanations, introductions, or conversational responses. \
Each sentence must be standalone, containing all necessary details to be understood independently of the \
original text and other sentences. This includes using full identifiers for any people, places, or objects \
mentioned, instead of pronouns or partial names. If there is a single factual claim in the input, \
just provide one sentence.""",
    },
    {"role": "user", "content": """{TEXT}"""},
]
