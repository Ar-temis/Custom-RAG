import ollama
import numpy as np

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'
LANGUAGE_MODEL = 'llama3.2:latest'

dataset = []

def extractPairs(doc):
    pairs = []
    currentQ = None
    currentA = []

    file = open(doc)

    for line in file:
        text = line.strip()
        if text.endswith(':'):
            continue
        if text.endswith('?'):
            if currentQ:
                pairs.append(f"{currentQ} {' '.join(currentA)}")
            currentQ = text  # reset the current question and answers
            currentA = []
        elif text.startswith(('●', '○')):
            currentA.append(text.lstrip("●○").strip())
    if currentQ:
        pairs.append(f"{currentQ} {' '.join(currentA)}")

    return pairs
dataset = extractPairs("FAQ.txt")
VECTOR_DB = []

def add_chunk(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk(chunk)


print(f"Finished adding all files to vector database.")

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

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
