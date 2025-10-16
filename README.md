# 🧠 Simple GenAI Applications using LangChain, OpenAI & FAISS

> **Description:**  
> This repository contains two practical examples of building Generative AI (GenAI) applications using **LangChain**, **OpenAI GPT-4o**, and **FAISS**.  
> - The first notebook demonstrates **prompt engineering and chain composition** with LangChain.  
> - The second notebook implements a **Retrieval-Augmented Generation (RAG)** workflow, where data is scraped from a website, chunked, embedded, and queried using a vector database.

---

## 📘 Project Overview

This project showcases end-to-end integration between **LangChain**, **OpenAI API**, and **FAISS** to create both:
1. A **conversational AI system** with custom prompt templates.  
2. A **context-aware Q&A retriever** that can answer user queries based on ingested web data.

Each notebook focuses on a distinct capability of LangChain — from LLM chaining to vector search and retrieval.

---

## 📂 Repository Structure

├── OpenAI_Ollama.ipynb # RAG pipeline using LangChain, FAISS, and OpenAI
├── Simple_LangChain_App.ipynb # Prompt-driven LLM chat app using LangChain
├── .env.example # Example for environment variables
└── README.md # Project documentation


---

## 🧩 Notebooks Summary

### 1️⃣ `Simple_LangChain_App.ipynb`

A lightweight demonstration of **prompt engineering** and **LangChain’s chaining mechanism**.

**Features:**
- Loads OpenAI GPT-4o via `langchain_openai`.
- Uses `ChatPromptTemplate` to construct structured conversations.
- Demonstrates how to connect `Prompt → LLM → OutputParser` using LangChain’s pipeline.
- Generates informative AI responses (e.g., definitions of LangChain, motivational advice, etc.).

**Core Concepts:**
- `ChatOpenAI` for calling OpenAI’s GPT models.  
- `ChatPromptTemplate` for message design.  
- `StrOutputParser` for extracting clean text output.

---

### 2️⃣ `OpenAI_Ollama.ipynb`

A **Retrieval-Augmented Generation (RAG)** example that fetches data from the LangChain documentation, indexes it, and answers natural language queries.

**Workflow Steps:**
1. **Data Ingestion:**  
   Uses `WebBaseLoader` to scrape content from  
   `https://docs.langchain.com/langsmith/administration-overview`.

2. **Text Splitting:**  
   `RecursiveCharacterTextSplitter` divides long documents into manageable chunks (`chunk_size=1000`, `chunk_overlap=200`).

3. **Embedding Generation:**  
   Converts text chunks into vector embeddings using `OpenAIEmbeddings`.

4. **Vector Storage:**  
   Stores embeddings in a **FAISS vector database** for similarity search.

5. **Query & Retrieval:**  
   Retrieves relevant chunks from the vector DB based on a natural language query.

6. **Answer Generation:**  
   Combines retrieved documents with a GPT-4o model (`ChatOpenAI`) to produce context-aware answers using a **retrieval chain**.

**Key Components:**
- `langchain_community.vectorstores.FAISS`  
- `langchain_openai.OpenAIEmbeddings`  
- `langchain.chains.create_retrieval_chain`  
- `ChatPromptTemplate` and `create_stuff_documents_chain`


