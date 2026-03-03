import json
import time
from rag_pipeline import initialize_rag

# ── Load test data ─────────────────────────────────────────────────────────────
with open("eval_data.json") as f:
    test_cases = json.load(f)

print("🔄 Initializing RAG pipeline...")
chain, retriever = initialize_rag()
print(f"✅ Pipeline ready. Running evaluation on {len(test_cases)} questions...\n")

# ── Metrics ────────────────────────────────────────────────────────────────────
def check_relevance(chunks, keywords):
    """Check if any keyword appears in any retrieved chunk."""
    combined = " ".join([c.page_content.lower() for c in chunks])
    return [any(kw.lower() in combined for kw in keywords) for _ in chunks]

def precision_at_k(chunks, keywords, k):
    """Fraction of top-k chunks that are relevant."""
    top_k = chunks[:k]
    combined = " ".join([c.page_content.lower() for c in top_k])
    relevant = sum(1 for kw in keywords if kw.lower() in combined)
    return min(relevant / k, 1.0)

def recall_at_k(chunks, keywords, k):
    """Fraction of keywords found in top-k chunks."""
    top_k = chunks[:k]
    combined = " ".join([c.page_content.lower() for c in top_k])
    found = sum(1 for kw in keywords if kw.lower() in combined)
    return found / len(keywords)

def reciprocal_rank(chunks, keywords):
    """1/rank of first relevant chunk."""
    for i, chunk in enumerate(chunks):
        if any(kw.lower() in chunk.page_content.lower() for kw in keywords):
            return 1 / (i + 1)
    return 0.0

# ── Run Evaluation ─────────────────────────────────────────────────────────────
k = 4
results = []
total_time = 0

for i, case in enumerate(test_cases):
    q = case["question"]
    keywords = case["relevant_keywords"]

    start = time.time()
    chunks = retriever.invoke(q)
    latency = time.time() - start
    total_time += latency

    p_k = precision_at_k(chunks, keywords, k)
    r_k = recall_at_k(chunks, keywords, k)
    rr = reciprocal_rank(chunks, keywords)

    results.append({
        "question": q,
        "precision@k": round(p_k, 3),
        "recall@k": round(r_k, 3),
        "reciprocal_rank": round(rr, 3),
        "latency_ms": round(latency * 1000, 1)
    })

    print(f"Q{i+1}: {q[:60]}...")
    print(f"     Precision@{k}: {p_k:.3f} | Recall@{k}: {r_k:.3f} | RR: {rr:.3f} | {latency*1000:.1f}ms\n")

# ── Aggregate Results ──────────────────────────────────────────────────────────
avg_precision = sum(r["precision@k"] for r in results) / len(results)
avg_recall = sum(r["recall@k"] for r in results) / len(results)
mrr = sum(r["reciprocal_rank"] for r in results) / len(results)
avg_latency = total_time * 1000 / len(results)

print("=" * 60)
print("📊 EVALUATION RESULTS SUMMARY")
print("=" * 60)
print(f"  Model:          all-MiniLM-L6-v2")
print(f"  Documents:      Apple 10-K 2023")
print(f"  Test Questions: {len(test_cases)}")
print(f"  k:              {k}")
print(f"─" * 60)
print(f"  Precision@{k}:    {avg_precision:.3f}")
print(f"  Recall@{k}:       {avg_recall:.3f}")
print(f"  MRR:            {mrr:.3f}")
print(f"  Avg Latency:    {avg_latency:.1f}ms")
print("=" * 60)

# Save results to JSON
with open("eval_results.json", "w") as f:
    json.dump({
        "model": "all-MiniLM-L6-v2",
        "summary": {
            "precision_at_k": round(avg_precision, 3),
            "recall_at_k": round(avg_recall, 3),
            "mrr": round(mrr, 3),
            "avg_latency_ms": round(avg_latency, 1)
        },
        "per_question": results
    }, f, indent=2)

print("\n💾 Results saved to eval_results.json")