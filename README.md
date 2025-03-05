# Custom-RAG

## Introduction

This is a custom RAG you can use with any model that uses ollama. You can upload any pdf, word, or txt file into the DataFiles folder and use it in a RAG system.

I hope you can excuse the troublesome setup process. This is tested on Linux Mint Cinnamon and MacOS. I do not plan to support windows.

## Installation

First, ensure that ollama is installed on your system, and pull the following models using the commands below:

```sh
ollama pull bge-m3
```

```sh
ollama pull llama3.2:latest
```

Then, install the python dependencies. (I warn you that there are a lot of requirements.)

```sh
pip3 install -r requirements.txt
```

## Generating your vector database

This RAG system uses ChromaDB to store its vector embeddings. Go into the /Custom-RAG/ChatBot/DataFiles directory and ensure that .FilesAdded.txt is clear. This file will be used to log everything that is embedded. After that, upload your documents into the folder.

Go back to your ChatBot directory and run:

```sh
python3 GenerateDB.py
```

After it is done running, you should see a VectorDB directory with the vector database.

## Starting the system

You can run this from the terminal or through a really ugly chat-like UI I made in 2 days. Beware that if ollama is not started when you run the program, you have to run it in the background with:

```sh
ollama serve
```

The backend is written in Flask and the front-end is just pure JS.

You can also use any generative model ollama supports. Just pull the model you want in ollama, and change the `LANGUAGE_MODEL` in `ChatBot.py` to the model name.

### Running it in terminal

Run the `TerminalChat.py`:

```sh
python3 ./ChatBot/TerminalChat.py
```

### Running it on a localhost

Run the `app.py` in `/Custom-RAG`:

```sh
python3 app.py
```

## Explanation on this RAG

Files will be embedded using "bge m3" model from BAAI collection. I used this model because it is fast, multi-lingual, and can be run on almost any modern hardware.

The reranking or cross-encoder is "ms-marco-MiniLM-L-6-v2". This model is called using Sentence-transformers.

The generative model is "llama-3.2", which is a fast, decent 3B parameter model for this occasion. You can run deepseek with bigger parameters if you would like too.

### What is happening under the hood?
<img width="1062" alt="Screenshot 2025-03-06 at 02 30 56" src="https://github.com/user-attachments/assets/7e759d5c-ae24-447f-8abe-420623361364" />
