# 📄 RAG Document Q&A System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

An end-to-end Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask natural language questions about them. Built with LangChain, FAISS, LLaMA 3.1, and Streamlit — with retrieval evaluation, embedding model benchmarking, and hallucination mitigation.

## 🚀 Live Demo
🔗 [Try it here](https://rag-app-system-gr3ywxg6vsnecsjya923fe.streamlit.app)

---

## 🏗️ Architecture
```
PDF Upload → Text Extraction (PyPDF2) → Adaptive Chunking (LangChain)
                                                    ↓
                                    Embeddings (MiniLM-L6-v2)
                                                    ↓
                                       FAISS Vector Store
                                                    ↓
User Question → Embed → Similarity Search → Top-4 Chunks
                                                    ↓
                          Hallucination Filter (keyword overlap)
                                                    ↓
                              LLaMA 3.1 8B (Groq API) → Answer + Citations
```

---

## ✨ Features

- 📤 Upload multiple PDFs directly through the browser
- 🧠 Adaptive chunking — automatically adjusts chunk size based on document length
- ⚡ FAISS vector store with sub-10ms similarity search
- 🤖 LLaMA 3.1 8B via Groq for fast, accurate answer generation
- 🛡️ Dual-layer hallucination mitigation
- 📎 Source citations shown under every answer
- 💬 Full chat history with session state
- 🔄 Rebuild index on demand

---

## 📊 Evaluation Results

### Retrieval Metrics (Apple 10-K 2023, 20-question test set, k=4)

| Metric | Score |
|---|---|
| Precision@4 | 0.537 |
| Recall@4 | 0.647 |
| MRR | 0.712 |
| Avg Latency | 10.3ms |

### Embedding Model Comparison

| Metric | MiniLM-L6-v2 | mpnet-base-v2 |
|---|---|---|
| Vector Dimension | 384 | 768 |
| Index Build Time | 3.17s | 19.53s |
| Precision@4 | 0.537 | 0.562 |
| Recall@4 | 0.647 | 0.692 |
| MRR | 0.713 | 0.817 |
| Avg Query Latency | 5.8ms | 31.5ms |

**Finding:** mpnet-base-v2 improves MRR by 10.4% but is 5x slower. MiniLM-L6-v2 is the better production choice for real-time applications.

---

## 🛡️ Hallucination Mitigation

Two-layer approach:
1. **Prompt grounding** — LLM is explicitly instructed to respond "I don't have enough information" if context is insufficient
2. **Keyword overlap filter** — before calling the LLM, retrieved chunks are checked for meaningful word overlap with the question. If overlap is below threshold, query is blocked without an API call

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| LLM | LLaMA 3.1 8B via Groq API |
| UI | Streamlit |
| PDF Parsing | PyPDF2 |

---

## 📦 Installation
```bash
git clone https://github.com/SanikaaJadhav/rag-qa-system.git
cd rag-qa-system
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at https://console.groq.com

---

## 🚀 Usage
```bash
streamlit run app.py
```

1. Upload PDF files using the sidebar
2. Click **Build / Rebuild Index**
3. Ask questions in the chat box!

---

## 🔬 Running Evaluations
```bash
# Retrieval evaluation
python eval.py

# Embedding model comparison
python compare_embeddings.py

# Hallucination mitigation test
python test_hallucination.py
```

---

## 📁 Project Structure
```
rag-qa-system/
├── app.py                    # Streamlit UI
├── rag_pipeline.py           # Core RAG logic + hallucination mitigation
├── eval.py                   # Retrieval evaluation (Precision@k, Recall@k, MRR)
├── compare_embeddings.py     # Embedding model benchmarking
├── test_hallucination.py     # Hallucination mitigation tests
├── eval_data.json            # 20-question test set (Apple 10-K)
├── eval_results.json         # Evaluation results
├── embedding_comparison.json # Model comparison results
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## 🧠 Key Technical Decisions

**Adaptive Chunking:** Dynamically adjusts chunk size based on document length — 200 chars for resumes, 700 chars for financial reports — optimizing retrieval quality across document types.

**RAG over Fine-tuning:** Chosen because it allows real-time document updates without retraining, provides source citations, and works with any document at zero additional training cost.

**Local FAISS over Cloud Vector DB:** Zero cost, sub-10ms search, with architecture designed to swap in Pinecone for production multi-user scaling.

**Custom LLM Wrapper:** Built a custom `GroqLLM` class extending LangChain's base `LLM` class for full control over inference parameters.

---

## 🔮 Future Improvements

- Semantic chunking using LangChain's SemanticChunker
- Hierarchical indexing for multi-document reasoning
- Conversation memory for follow-up questions
- Pinecone integration for cloud-scale deployment
- Switch to mpnet-base-v2 for high-stakes use cases (10.4% MRR improvement)
