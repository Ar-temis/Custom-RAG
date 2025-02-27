import os
import chromadb
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
client = chromadb.PersistentClient(path="./VectorDB/")

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=480,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'

collection = client.get_or_create_collection("vectorDB")

def add_chunk(chunk, filename, page = 0):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  collection.add(
      documents=[chunk],
      embeddings=[embedding],
      ids=(str(hash(chunk))),
      metadatas=[{"filename": filename, "page": page}]
  )

def read_files(path):
    print(f"Reading from directory: {os.path.abspath(path)}")  # Debugging output
    if not os.path.exists(path):
        print(f"Error: The directory '{path}' does not exist.")
        return

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        print(f"Processing file: {file_path}")  
        
        if filename.endswith(".txt"):
            pairs = []
            currentQ = None
            currentA = []

            with open(file_path, "r", encoding="utf-8") as file:
                for chunk in text_splitter.split_text(file):
                    add_chunk(chunk, filename)


        if filename.endswith(".pdf"):
            pdf_loader = PyPDFLoader(file_path)
            for page in pdf_loader.lazy_load():
                entry = text_splitter.split_text(page.page_content)
                for chunk in entry:
                    add_chunk(chunk, filename, page.metadata["page"])
        
        if filename.endswith(".docx"):
            word_loader = UnstructuredWordDocumentLoader(file_path)
            for page in word_loader.lazy_load():
               entry = text_splitter.split_text(page.page_content)
               for chunk in entry:
                   add_chunk(chunk, filename) 

        print(f"Finished loading {filename}.")

read_files("./DataFiles/")

print(f"Finished adding all files to vector database.")