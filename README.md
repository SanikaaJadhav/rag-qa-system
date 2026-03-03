# 📄 RAG Document Q&A System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-purple)

An end-to-end Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask natural language questions about them. Built with LangChain, FAISS, and LLaMA 3.1 via Groq.

## 🚀 Live Demo
🔗 [Try it here](https://huggingface.co/spaces/SanikaaJadhav/rag-qa-system)

## Architecture
```
PDF Documents → Text Extraction → Adaptive Chunking → Embeddings (MiniLM-L6)
                                                              ↓
User Question → Embed Question → FAISS Similarity Search → Top-K Chunks
                                                              ↓
                                              LLaMA 3.1 (Groq) → Grounded Answer
```

## Features

- 📤 Upload multiple PDFs directly through the browser
- 🧠 Adaptive chunking — automatically adjusts chunk size based on document length
- ⚡ FAISS vector store for millisecond similarity search
- 🤖 LLaMA 3.1 (via Groq) for fast, accurate answer generation
- 📎 Source citations shown under every answer
- 💬 Full chat history with session state
- 🔄 Rebuild index on demand when new documents are added

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain (LCEL) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| LLM | LLaMA 3.1 8B via Groq API |
| UI | Streamlit |
| PDF Parsing | PyPDF2 |

## 📦 Installation
```bash
# Clone the repo
git clone https://github.com/SanikaaJadhav/rag-qa-system.git
cd rag-qa-system

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at https://console.groq.com

## Usage
```bash
streamlit run app.py
```

Then:
1. Upload PDF files using the sidebar
2. Click **Build / Rebuild Index**
3. Ask questions in the chat box!

## 💡 Example Questions

For a financial report (e.g., Apple 10-K):
- *"What was the total revenue in 2023?"*
- *"What are the main risk factors?"*
- *"How much was spent on R&D?"*

For a resume:
- *"What internships has this person done?"*
- *"What technical skills do they have?"*

## 🧠 Key Technical Decisions

**Adaptive Chunking:** Instead of fixed chunk sizes, the system automatically adjusts based on document length — smaller chunks for short documents like resumes, larger chunks for long documents like annual reports.

**RAG over Fine-tuning:** RAG was chosen over fine-tuning because it allows real-time document updates without retraining, provides source citations, and works with any document without additional training cost.

**FAISS over Cloud Vector DB:** Local FAISS index for fast prototyping and zero cost, with architecture designed to swap in Pinecone or Weaviate for production scaling.

## 📁 Project Structure
```
rag-qa-system/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG logic
├── test_pipeline.py    # Pipeline testing
├── requirements.txt    # Dependencies
├── .env                # API keys (not committed)
└── data/               # PDF storage (not committed)
```

## 🔮 Future Improvements

- Semantic chunking using LangChain's SemanticChunker
- Hierarchical indexing for multi-document reasoning
- Swap FAISS for Pinecone for cloud-scale deployment
- Add conversation memory for follow-up questions
- Support for more file types (DOCX, TXT, CSV)