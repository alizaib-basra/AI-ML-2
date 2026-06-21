# 🧠 Context-Aware RAG Chatbot

A conversational chatbot with memory and retrieval-augmented generation (RAG) built with Streamlit and OpenRouter.

## ✨ Features

- 🔍 **RAG Pipeline** — Retrieves relevant knowledge before generating answers
- 🧠 **Context Memory** — Remembers conversation history for follow-up questions
- 📚 **Vector Search** — Semantic similarity search using sentence-transformers
- 🎨 **Beautiful UI** — Dark-themed Streamlit interface
- 📎 **Source Transparency** — Shows which chunks were retrieved with relevance scores

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API key to .env
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=sk-or-...

# Run the app
streamlit run app.py
```

## 📁 Structure

```
rag-chatbot/
├── app.py              # Streamlit UI
├── src/
│   └── rag_engine.py   # RAG pipeline core
├── docs/
│   └── knowledge_base.txt  # AI/ML knowledge corpus
├── requirements.txt
└── .env.example
```

## 🧠 How It Works

1. Documents are split into chunks and embedded using `all-MiniLM-L6-v2`
2. Embeddings are stored in ChromaDB vector store
3. User query is embedded and top-K similar chunks retrieved
4. Retrieved context + conversation history sent to LLM via OpenRouter
5. Response generated with full awareness of context and chat history

## 📚 Knowledge Base Topics

- Artificial Intelligence overview
- Machine Learning (supervised, unsupervised, reinforcement)
- Deep Learning & neural network architectures
- NLP & Large Language Models
- RAG & Vector Embeddings
- LangChain framework
- AI Ethics & Safety
- Python for AI/ML
- Computer Vision
