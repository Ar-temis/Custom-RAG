import chromadb
import ollama
client = chromadb.PersistentClient(path="./VectorDB/")

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'


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
dataset = extractPairs("./DataFiles/FAQ.txt")


VECTOR_DB = []
collection = client.get_or_create_collection("faq_vector")

def add_chunk(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  collection.add(
      documents=[chunk],
      embeddings=[embedding],
      ids=(str(hash(chunk)))
  )
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk(chunk)


print(f"Finished adding all files to vector database.")

