from rag_pipeline import initialize_rag

chain, retriever = initialize_rag()

question = "What is the main topic of this document?"

# Get source documents
docs = retriever.invoke(question)

print("\n📌 Question:", question)
print("💡 Answer:", chain.invoke(question))
print("\n📎 Sources:")
for doc in docs:
    print(" -", doc.metadata.get("source", "unknown"))