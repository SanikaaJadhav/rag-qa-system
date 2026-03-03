import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from groq import Groq
from typing import Optional, List

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ── Custom LLM using Groq (free, fast) ────────────────────────────────────────
class GroqLLM(LLM):
    api_key: str
    model: str = "llama-3.1-8b-instant"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content


# ── PDF Loading ────────────────────────────────────────────────────────────────
def load_pdfs(data_folder="data"):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            documents.append((text, filename))
            print(f"✅ Loaded: {filename} ({len(reader.pages)} pages)")
    return documents


# ── Chunking ───────────────────────────────────────────────────────────────────
def get_chunk_settings(documents):
    """
    Automatically decide chunk size based on average document length.
    Short docs (resumes) = smaller chunks, more overlap
    Long docs (10-Ks, reports) = larger chunks, less overlap
    """
    total_chars = sum(len(text) for text, _ in documents)
    avg_chars = total_chars / len(documents)

    if avg_chars < 5000:        # Short doc (resume, 1-2 pages)
        return 200, 80
    elif avg_chars < 50000:     # Medium doc (report, 10-20 pages)
        return 400, 100
    else:                       # Long doc (10-K, 50+ pages)
        return 700, 50


def chunk_documents(documents):
    chunk_size, chunk_overlap = get_chunk_settings(documents)
    print(f"📐 Auto chunk size: {chunk_size}, overlap: {chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = []
    for text, filename in documents:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})
    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks


# ── Vector Store ───────────────────────────────────────────────────────────────
def build_vector_store(chunks):
    print("⏳ Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    print("⏳ Building FAISS index...")
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")
    print("✅ FAISS index built and saved!")
    return vector_store


def load_vector_store(index_dir="faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded from disk")
    return vector_store


# ── QA Chain ───────────────────────────────────────────────────────────────────
def build_qa_chain(vector_store, similarity_threshold=0.0):
    llm = GroqLLM(api_key=GROQ_API_KEY)

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the documents to answer this."

Context:
{context}

Question: {question}

Answer:""")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        if not docs:
            return ""
        return "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

    def is_relevant(docs, question):
        if not docs:
            return False
        stop_words = {"what","was","is","the","a","an","of","in","did","how","who","where","when","are","does","do"}
        question_words = set(question.lower().split()) - stop_words
        combined = " ".join([d.page_content.lower() for d in docs])
        overlap = sum(1 for w in question_words if w in combined)
        return overlap >= 1

    class HallucinationSafeChain:
        def invoke(self, question):
            docs = retriever.invoke(question)
            if not is_relevant(docs, question):
                return "⚠️ Insufficient context found in documents to answer this question reliably."
            context = format_docs(docs)
            formatted_prompt = prompt.format(context=context, question=question)
            return llm._call(formatted_prompt)

    safe_chain = HallucinationSafeChain()
    print("✅ QA chain ready with hallucination mitigation")
    return safe_chain, retriever
# ── Entry Point ────────────────────────────────────────────────────────────────
def initialize_rag(force_rebuild=False, data_dir="data", index_dir="faiss_index", similarity_threshold=0.75):
    if os.path.exists(index_dir) and not force_rebuild:
        print("📂 Found existing FAISS index, loading...")
        vector_store = load_vector_store(index_dir)
    else:
        print("📄 No index found, processing PDFs...")
        documents = load_pdfs(data_dir)
        if not documents:
            raise ValueError("❌ No PDFs found!")
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks, index_dir)

    chain, retriever = build_qa_chain(vector_store, similarity_threshold=0.0)
    return chain, retriever