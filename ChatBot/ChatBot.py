from Retrieval import retrieve
import ollama

LANGUAGE_MODEL = 'llama3.2:latest'

def askBot (input_query):
    retrieved_knowledge = retrieve(input_query)

    context = "\n\n".join(
        [f" - {chunk}\nDocument Info: {meta}" for chunk, meta in retrieved_knowledge]
    )
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. When you are answering be reasonable and helpful.
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

    return response

input_query = input('\nAsk me a question: ')
while input_query != "/bye":
    retrieved_knowledge = retrieve(input_query)

    context = "\n\n".join(
        [f" - {chunk}\nDocument Info: {meta}" for chunk, meta in retrieved_knowledge]
    )
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. When you are answering be reasonable and helpful.
    At the bottom of your answer include which documents you looked at and which pages.
    Don't make up any new information:
    {context}
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=False,
    )

    print('Chatbot response:')
    print(stream['message']['content'])
    # for chunk in stream:
    #     print(chunk['message']['content'], end='', flush=True)
    input_query = input('\nAsk me a question: ')

print("Goodbye. Have a great day :D")