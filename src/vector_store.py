import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="knowledge_base")


def add_topic_to_vector_store(topic, chunks, embeddings):
    existing = collection.get(include=['metadatas']) if collection.count() else {"metadatas": []}
    existing_topics = {m['topic'] for m in existing['metadatas']}

    if topic in existing_topics:
        print(f"⚠️ Topic '{topic}' already exists in vector store.")
        return False

    ids = [f"{topic}_{i}" for i in range(len(chunks))]
    metadatas = [{"topic": topic} for _ in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Topic '{topic}' added to vector DB")
    return True


def search_index(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)

    if results['documents'][0]:
        print("\n📌 Top Matching Chunks:")
        for i, doc in enumerate(results['documents'][0]):
            print(f" 🔹 Chunk {i+1}: {doc[:150]}...\n")
        return results['documents'][0]  # ✅ Return as list of chunks
    else:
        print("\n⚠️ No matching chunks found in vector DB.")
        return []
