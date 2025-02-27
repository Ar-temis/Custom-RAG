from Retrieval import retrieve
import ollama

LANGUAGE_MODEL = 'llama3.2:latest'

input_query = input('\nAsk me a question: ')
while input_query != "/bye":
    retrieved_knowledge = retrieve(input_query)

    # print('Retrieved knowledge:')
    bad_query_count = 0
    for chunk, similarity in retrieved_knowledge:
        if similarity < 0.2:
            bad_query_count = bad_query_count + 1
        # print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. When you are answering be reasonable and helpful.
    Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    input_query = input('\nAsk me a question: ')

print("Goodbye. Have a great day :D")