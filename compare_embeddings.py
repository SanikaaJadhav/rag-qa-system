import json
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

# ── Load test data ─────────────────────────────────────────────────────────────
with open("eval_data.json") as f:
    test_cases = json.load(f)

# ── Load PDF ───────────────────────────────────────────────────────────────────
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_index(model_name, text):
    print(f"\n⏳ Building index with {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    
    start = time.time()
    vs = FAISS.from_texts(chunks, embeddings)
    index_time = time.time() - start
    
    print(f"✅ Built in {index_time:.1f}s | Chunks: {len(chunks)} | Vector dim: {embeddings._client.get_sentence_embedding_dimension()}")
    return vs, embeddings, index_time, len(chunks)

def evaluate_model(vector_store, test_cases, k=4):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    precisions, recalls, rrs, latencies = [], [], [], []

    for case in test_cases:
        keywords = case["relevant_keywords"]
        start = time.time()
        chunks = retriever.invoke(case["question"])
        latency = time.time() - start

        combined = " ".join([c.page_content.lower() for c in chunks])
        relevant_hits = sum(1 for kw in keywords if kw.lower() in combined)

        precisions.append(min(relevant_hits / k, 1.0))
        recalls.append(relevant_hits / len(keywords))
        latencies.append(latency * 1000)

        rr = 0.0
        for i, chunk in enumerate(chunks):
            if any(kw.lower() in chunk.page_content.lower() for kw in keywords):
                rr = 1 / (i + 1)
                break
        rrs.append(rr)

    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 3),
        "recall_at_k": round(sum(recalls) / len(recalls), 3),
        "mrr": round(sum(rrs) / len(rrs), 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
    }

# ── Models to compare ──────────────────────────────────────────────────────────
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

text = load_pdf("data/Apple-10-K-2023.pdf")
comparison_results = []

for model_name in models:
    vs, emb, index_time, n_chunks = build_index(model_name, text)
    metrics = evaluate_model(vs, test_cases)
    
    dim = emb._client.get_sentence_embedding_dimension()
    result = {
        "model": model_name.split("/")[-1],
        "vector_dimension": dim,
        "index_build_time_s": round(index_time, 2),
        "num_chunks": n_chunks,
        **metrics
    }
    comparison_results.append(result)

# ── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📊 EMBEDDING MODEL COMPARISON RESULTS")
print("=" * 70)
print(f"{'Metric':<25} {'MiniLM-L6-v2':>20} {'mpnet-base-v2':>20}")
print("-" * 70)

metrics_to_show = [
    ("Vector Dimension", "vector_dimension"),
    ("Index Build Time (s)", "index_build_time_s"),
    ("Precision@4", "precision_at_k"),
    ("Recall@4", "recall_at_k"),
    ("MRR", "mrr"),
    ("Avg Latency (ms)", "avg_latency_ms"),
]

for label, key in metrics_to_show:
    v1 = comparison_results[0][key]
    v2 = comparison_results[1][key]
    print(f"{label:<25} {str(v1):>20} {str(v2):>20}")

print("=" * 70)

# Save results
with open("embedding_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("\n💾 Results saved to embedding_comparison.json")