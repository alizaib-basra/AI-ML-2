"""
RAG Engine — Core retrieval and generation logic.
Uses OpenRouter for LLM, sentence-transformers for embeddings, ChromaDB for vector store.
"""

import os
import json
import time
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
import requests

load_dotenv()

# ── Optional heavy imports (graceful fallback) ──────────────────────
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


# ── Text splitter ────────────────────────────────────────────────────
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    sections = text.split("========================================")
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            words = section.split()
            current = []
            current_len = 0
            for word in words:
                current.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    chunks.append(" ".join(current))
                    # Keep overlap
                    overlap_words = current[-overlap//5:]
                    current = overlap_words
                    current_len = sum(len(w) + 1 for w in current)
            if current:
                chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 50]


# ── RAG Engine ───────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = None
        self.embed_model = None
        self.collection = None
        self.use_chroma = CHROMA_AVAILABLE and EMBEDDINGS_AVAILABLE
        self._init_embedding_model()

    def _init_embedding_model(self):
        if EMBEDDINGS_AVAILABLE:
            self.embed_model = SentenceTransformer(EMBED_MODEL)

    def load_documents(self, doc_paths: List[str]) -> int:
        """Load and chunk documents, build vector index."""
        all_text = ""
        for path in doc_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    all_text += f.read() + "\n\n"

        self.chunks = split_text(all_text)

        if self.embed_model:
            self.embeddings = self.embed_model.encode(self.chunks, show_progress_bar=False)

            if self.use_chroma:
                client = chromadb.Client()
                try:
                    client.delete_collection("rag_kb")
                except:
                    pass
                self.collection = client.create_collection("rag_kb")
                self.collection.add(
                    documents=self.chunks,
                    ids=[f"chunk_{i}" for i in range(len(self.chunks))],
                    embeddings=self.embeddings.tolist()
                )

        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        if not self.chunks:
            return []

        if self.embed_model and self.embeddings is not None:
            query_emb = self.embed_model.encode([query])

            if self.use_chroma and self.collection:
                results = self.collection.query(
                    query_embeddings=query_emb.tolist(),
                    n_results=min(top_k, len(self.chunks))
                )
                docs = results["documents"][0]
                distances = results["distances"][0]
                return [
                    {"text": doc, "score": round(1 - dist, 3), "rank": i + 1}
                    for i, (doc, dist) in enumerate(zip(docs, distances))
                ]
            else:
                # Cosine similarity fallback
                from numpy.linalg import norm
                sims = []
                for i, emb in enumerate(self.embeddings):
                    sim = float(np.dot(query_emb[0], emb) / (norm(query_emb[0]) * norm(emb) + 1e-10))
                    sims.append((sim, i))
                sims.sort(reverse=True)
                return [
                    {"text": self.chunks[idx], "score": round(sim, 3), "rank": i + 1}
                    for i, (sim, idx) in enumerate(sims[:top_k])
                ]
        else:
            # Keyword fallback
            query_words = set(query.lower().split())
            scored = []
            for i, chunk in enumerate(self.chunks):
                chunk_words = set(chunk.lower().split())
                score = len(query_words & chunk_words) / (len(query_words) + 1)
                scored.append((score, i))
            scored.sort(reverse=True)
            return [
                {"text": self.chunks[idx], "score": round(score, 3), "rank": i + 1}
                for i, (score, idx) in enumerate(scored[:top_k])
            ]

    def generate(self, query: str, context_chunks: List[Dict], history: List[Dict]) -> Tuple[str, float]:
        """Generate response using retrieved context and conversation history."""
        context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

        system_prompt = """You are an intelligent AI assistant with access to a knowledge base about Artificial Intelligence, Machine Learning, Deep Learning, NLP, RAG, LangChain, and related topics.

INSTRUCTIONS:
- Answer questions using the provided context from the knowledge base
- If the context doesn't contain enough information, use your general knowledge but mention it
- Maintain a helpful, clear, and conversational tone
- Reference specific concepts from the context when relevant
- Keep answers concise but complete
- Remember the conversation history for follow-up questions"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in history[-6:]:  # Last 6 turns for context
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current query with context
        user_message = f"""Context from knowledge base:
{context}

Question: {query}"""
        messages.append({"role": "user", "content": user_message})

        start = time.time()
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rag-chatbot",
                "X-Title": "RAG Chatbot"
            },
            json={
                "model": MODEL,
                "max_tokens": 800,
                "messages": messages
            }
        )
        elapsed = round(time.time() - start, 2)

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")

        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer, elapsed

    def chat(self, query: str, history: List[Dict]) -> Dict:
        """Full RAG pipeline: retrieve + generate."""
        retrieved = self.retrieve(query)
        answer, latency = self.generate(query, retrieved, history)
        return {
            "answer": answer,
            "sources": retrieved,
            "latency_s": latency,
            "chunks_retrieved": len(retrieved)
        }
