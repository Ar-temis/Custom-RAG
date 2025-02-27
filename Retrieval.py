import ollama
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(
  path="./VectorDB/"
)
collection = client.get_collection(name="vectorDB")

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'


def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  result = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_n,
    include=["documents", "metadatas"]
  )
  
  docs = result['documents'][0]  # List of document texts
  metas = result['metadatas'][0]  # List of metadata dictionaries

  # Convert to formatted string
  context = "\n\n".join(
      [f"{doc}\nMetadata: {meta}" for i, (doc, meta) in enumerate(zip(docs, metas))]
  )

  return context