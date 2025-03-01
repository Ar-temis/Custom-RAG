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
        "?",
    ],
    chunk_size=480,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

EMBEDDING_MODEL = 'bge-m3'

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
    
    existing_files = []
    with open(os.path.join(path, ".FilesAdded.txt"), "r") as file:
        for line in file:
            existing_files.append(line.strip())
    log = open(os.path.join(path, ".FilesAdded.txt"), "a")
    for filename in os.listdir(path):
        # TODO: add a logging mechanism
        if filename == ".FilesAdded.txt" or filename in existing_files:
            continue
        file_path = os.path.join(path, filename)

        print(f"Processing file: {file_path}")  
        
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                for chunk in text_splitter.split_text(file):
                    add_chunk(chunk, filename)


        if filename.endswith(".pdf"):
            pdf_loader = PyPDFLoader(file_path)
            for page in pdf_loader.load():
                entry = text_splitter.split_text(page.page_content)
                for chunk in entry:
                    add_chunk(chunk, filename, page.metadata["page"])
        
        if filename.endswith(".docx"):
            word_loader = UnstructuredWordDocumentLoader(file_path)
            for page in word_loader.load():
               entry = text_splitter.split_text(page.page_content)
               for chunk in entry:
                   add_chunk(chunk, filename) 

        log.write('\n'+filename)
        print(f"Finished loading {filename}.")
    log.close()
read_files("./DataFiles/")

print(f"Finished adding all files to vector database.")