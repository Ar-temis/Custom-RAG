import ollama
import chromadb
from sentence_transformers import CrossEncoder


client = chromadb.PersistentClient(
  path="./ChatBot/VectorDB"
)
collection = client.get_collection(name="vectorDB")

#EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'

EMBEDDING_MODEL = 'bge-m3'
RERANK_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)


def retrieve(query, top_n=10):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  result = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_n,
    include=["documents", "metadatas"]
  )
  pairs = []
  passages = result["documents"][0]
  # original = "\n\n".join([f"\n{i+1}. {passage}" for i, passage in enumerate(passages)])
  # print(original)
  scores = RERANK_MODEL.rank(query=query, documents=passages)
  for score in scores:
    pairs.append((result["documents"][0][score['corpus_id']], result["metadatas"][0][score["corpus_id"]]))
  # print([f'\n{i+1}. {pair}' for i, pair in enumerate(pairs)])
  return pairs[:5]