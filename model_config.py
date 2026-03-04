import os


base_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
embedding_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
model_path = os.path.join(base_model_path, os.listdir(base_model_path)[0])

model_parameter = {
    'Temperature': 0.7,
    'TopP': 0.8,
    'TopK': 20,
    'MinP': 0,
    'MaxNewTokens': 500
}

system_prompt = """
    Consider yourself a teacher with 40 years of experience who is responsible for the national exam. 
    You must provide accurate questions and answers based ONLY on the provided document context.

    CONTEXT FROM DOCUMENT:
    {context}


    USER REQUEST:
    {question}

    REQUIRED FORMAT:
    Question [Number]: [Write the question here]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A/B/C/D]

    EXAM:
    Question 1:
"""

sys = """
    Consider yourself a teacher with 40 years of experience who is responsible for the national exam. 
    You must provide accurate questions and answers for the question asked to you

    USER REQUEST:
    {question}

    REQUIRED FORMAT:
    Question [Number]: [Write the question here]
    Answer: [Write the answer here]
"""