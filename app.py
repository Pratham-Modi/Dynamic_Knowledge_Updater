import streamlit as st
from src.embedder import fetch_and_embed
from src.vector_store import add_topic_to_vector_store, search_index
from src.gemini_bot import get_fallback_response

CHUNK_SIZE = 500  # For consistent chunking

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Knowledge Updater", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("📘 AI Knowledge Updater")

# --- Topic Fetch Section ---
with st.expander("📥 Fetch & Store New Topic"):
    topic = st.text_input("Enter a topic (e.g., Artificial Intelligence, Elon Musk, FIFA World Cup)")
    if st.button("Fetch & Store"):
        with st.spinner("Fetching and embedding..."):
            chunks, embeddings = fetch_and_embed(topic, chunk_size=CHUNK_SIZE)
            if chunks:
                success = add_topic_to_vector_store(topic, chunks, embeddings)
                if success:
                    st.success(f"✅ Successfully added topic: {topic}")
                else:
                    st.warning(f"⚠️ Topic '{topic}' already exists.")
            else:
                st.error("❌ Failed to fetch topic. Try a different one.")

st.markdown("---")

# --- Chat Section ---
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question:")

if st.button("Generate Response"):
    if query:
        with st.spinner("Searching knowledge base..."):
            relevant = search_index(query, k=3)

            # If we found relevant docs, use them first
            if relevant:
                context = "\n".join(relevant)
                prompt = f"""You are a helpful assistant. Answer the question strictly using the context below.
If the context does not help, say that the information is not found in the context.

Context:
{context}

Question: {query}
Answer:"""

                response = get_fallback_response(prompt)

                # ✅ Fallback if Gemini still returns a weak/unrelated answer
                if any(keyword in response.lower() for keyword in ["does not contain", "no information", "not found"]):
                    response = get_fallback_response(query)

            else:
                # No vector data found, fallback directly
                response = get_fallback_response(query)

        st.session_state.history.append((query, response))

# --- Chat History ---
if st.session_state.history:
    st.subheader("📜 Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**🧠 Q: {q}**")
        st.markdown(f"📖 A: {a}")
        st.markdown("---")

# --- Clear Chat Button ---
if st.button("🧹 Clear Chat History"):
    st.session_state.history = []
    st.success("Chat history cleared.")
