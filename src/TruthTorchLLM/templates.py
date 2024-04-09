DEFAULT_TEMPLATE = "{question}"

LLAMA_PROMPT_TEMPLATE = """[INST] <<SYS>>
You are a helpful assistant. Give short and precise answers.
<</SYS>>

Question: {question} Answer: [/INST]"""