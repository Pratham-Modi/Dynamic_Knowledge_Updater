import wikipedia
from sentence_transformers import SentenceTransformer
import time
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_and_embed(topic, chunk_size=1000):
    """
    Fetch content from Wikipedia for a given topic, split it into chunks,
    compute embeddings, and save the raw content into the data folder.
    
    Args:
        topic (str): The topic to fetch from Wikipedia
        chunk_size (int): Number of characters per chunk
    
    Returns:
        chunks (list[str]): List of text chunks
        embeddings (list[list[float]]): List of embeddings for each chunk
    """
    print(f"\n📚 Fetching data for: {topic}")
    
    try:
        text = wikipedia.page(topic).content
    except Exception:
        try:
            text = wikipedia.search(topic)[0]
            text = wikipedia.page(text).content
        except Exception as e:
            print(f"❌ Wikipedia fetch failed: {e}")
            return [], []

    # Save raw content to data folder
    safe_topic = topic.replace(" ", "_").replace("/", "_")
    file_path = os.path.join(DATA_DIR, f"{safe_topic}.txt")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Saved raw content to {file_path}")
    except Exception as e:
        print(f"⚠️ Failed to save content: {e}")

    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"🧠 Chunked into {len(chunks)} parts of ~{chunk_size} chars")

    # Compute embeddings
    start = time.time()
    embeddings = model.encode(chunks).tolist()
    print(f"✅ Embedding complete in {time.time() - start:.2f}s")

    return chunks, embeddings
