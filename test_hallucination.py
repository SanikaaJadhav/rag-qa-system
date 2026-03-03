from rag_pipeline import initialize_rag

chain, retriever = initialize_rag()

# These should answer fine
good_questions = [
    "What was Apple's total revenue in 2023?",
    "What products does Apple sell?",
]

# These should trigger mitigation
bad_questions = [
    "What is the recipe for chocolate cake?",
    "Who won the 2024 US election?",
    "What is the population of Mars?",
]

print("\n✅ QUESTIONS WITH CONTEXT:")
for q in good_questions:
    docs = retriever.invoke(q)
    print(f"\nQ: {q}")
    print(f"Docs found: {len(docs)}")
    if docs:
        print(f"Answer: {chain.invoke(q)[:200]}...")

print("\n⚠️ QUESTIONS WITHOUT CONTEXT:")
for q in bad_questions:
    docs = retriever.invoke(q)
    print(f"\nQ: {q}")
    print(f"Docs found: {len(docs)}")
    if not docs:
        print("Answer: ⚠️ Insufficient context found in documents to answer this question reliably.")
    else:
        print(f"Answer: {chain.invoke(q)[:200]}...")