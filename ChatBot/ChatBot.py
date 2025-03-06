from ChatBot import Retrieval
import ollama

LANGUAGE_MODEL = 'llama3.2:latest'

def askBot(input_query):
    retrieved_knowledge = Retrieval.retrieve(input_query)

    context = "\n\n".join(
        [f" - {chunk}\nDocument Info: {meta}" for chunk, meta in retrieved_knowledge]
    )
    instruction_prompt = f'''You are a helpful chatbot for Duke Kunshan University (DKU) students.
    If the answer is not provided in the context, say you don't know.
    Use the following pieces of context to answer the question. When you are answering, don't say that you looked at any files.
    At the bottom of your answer include which documents you looked at and which pages.
    Don't make up any new information:
    {context}
    '''

    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=False,
    ) 

    return response['message']['content']