from Retrieval import retrieve
import ollama

LANGUAGE_MODEL = 'llama3.2:latest'

input_query = input('\nAsk me a question: ')
while input_query != "/bye":
    retrieved_knowledge = retrieve(input_query)

    # Convert to formatted string
    context = "\n\n".join(
        [f" - {chunk}\nMetadata: {meta}" for chunk, meta in retrieved_knowledge]
  )
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. When you are answering be reasonable and helpful.
    When citing from a documents tell the user which documents you looked at and which pages at the bottom of your answer.
    Don't make up any new information:
    {retrieved_knowledge}
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    input_query = input('\nAsk me a question: ')

print("Goodbye. Have a great day :D")