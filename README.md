 🧠 RAG + LangGraph AI Assistant (Phase 1 → Phase 3)

A production-style **Retrieval-Augmented Generation (RAG) system** built using **FAISS, LangGraph, and LLMs**, featuring a full pipeline from document retrieval to reranking, context building, generation, and self-evaluation with retry loops.

---

# 🚀 Project Overview

This project implements an advanced AI assistant that can:

- Retrieve relevant knowledge from a vector database
- Rerank results for better relevance
- Build optimized context windows
- Generate answers using an LLM
- Evaluate responses automatically
- Retry if output quality is low (self-healing pipeline)

---

# 🧭 Project Evolution (Phases)

## 🟢 Phase 1 — Basic RAG Pipeline
- Document ingestion system
- Text chunking
- Embedding generation
- FAISS vector store creation
- Semantic search retrieval

### Goal:
Build a simple question-answering system using vector similarity search.

---

## 🟡 Phase 2 — Improved Retrieval System
- Added FAISS persistent index loading
- Built reusable `search()` interface
- Introduced reranking system for better results
- Normalized document format for stability

### Improvements:
- Better retrieval accuracy
- Reduced noise from irrelevant chunks
- Faster inference using cached embeddings

---

## 🔴 Phase 3 — LangGraph AI Agent System

Converted pipeline into a **graph-based AI workflow**:

### Nodes:
- `retrieve` → fetch documents from FAISS
- `rerank` → reorder best matches
- `generate` → LLM-based answer generation
- `evaluate` → quality scoring of answer
- `router` → decides retry or finish

### Features:
- 🔁 Automatic retry loop if answer is weak
- 🧠 Context-aware LLM prompting
- 📊 Self-evaluation using structured JSON scoring
- ⚙️ Fully modular LangGraph architecture

---

# 🧱 System Architecture


User Query
↓
Retrieve (FAISS Search)
↓
Rerank (Relevance Optimization)
↓
Context Builder
↓
LLM Generation
↓
Evaluator (JSON scoring)
↓
Router
├── PASS → END
└── FAIL → RETRY (loop back)


---

# 📂 Project Structure


langgraph/
│
├── graph.py # LangGraph workflow definition
├── nodes.py # retrieve, generate, evaluate nodes
├── llm.py # LLM API wrapper
├── reranker.py # reranking logic
├── state.py # shared state schema
├── test.py # testing entry point
│
├── db/
│ ├── vectorstore.py # FAISS search + embedding handling
│ ├── buildindex.py # index creation script
│
├── data/
│ ├── loader.py # document loading + chunking
│
└── README.md


---

# ⚙️ Tech Stack

- 🐍 Python
- 🧠 LLM (LLaMA via OpenAI-compatible API)
- 📦 FAISS (Vector Database)
- 🔗 LangGraph (AI workflow orchestration)
- 🔍 Sentence Embeddings
- ⚡ Reranking model

---

# 🔄 How It Works

### 1. Retrieval
User query is embedded and searched in FAISS index.

### 2. Reranking
Top-k results are reranked for semantic relevance.

### 3. Context Building
Relevant chunks are combined into a token-safe context window.

### 4. Generation
LLM generates answer based only on retrieved context.

### 5. Evaluation
Another LLM call checks:
- correctness
- relevance
- completeness

Returns:

```json
{
  "pass": true,
  "score": 8,
  "reason": "Good context usage"
}
6. Routing
If pass → END
If fail → retry retrieval
📊 Example Output
Query:
What is AI?
System Flow:
Retrieve → Rerank → Context → Generate → Evaluate → PASS
Answer:

Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.

🧠 Key Features
✔ FAISS vector search
✔ Reranking layer for accuracy
✔ LangGraph multi-node architecture
✔ Self-evaluating LLM responses
✔ Retry-based correction loop
✔ Modular and scalable design
🔥 Why This Project Matters

This is not a simple chatbot.

It demonstrates:

Real-world RAG architecture
Agentic AI workflow (LangGraph)
Self-correcting LLM systems
Production-style modular design
🚀 Future Improvements
Add Streamlit UI chat interface
Add FastAPI backend
Add memory (conversation history DB)
Add tool calling (web search, calculator)
Deploy on cloud (Docker + GCP/AWS)
👨‍💻 Author

Noman Ashraf

AI Engineer | LLM Systems | RAG Architect

📌 Repository Link

https://github.com/Noman-ashraf1/rag-langgraph-assistant


---

# 💡 If you want next upgrade

I can also help you make:

### 🔥 1. Architecture diagram image (for README)
### 🔥 2. Resume bullet points (ATS optimized)
### 🔥 3. LinkedIn post (viral AI project post)
### 🔥 4. “Interview explanation script” (how to explain this project in 2 minutes)

Just say: **“next level portfolio”**
