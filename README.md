# 🧠 Dynamic Knowledge Updater

This project is a smart and efficient **Knowledge Updater chatbot** that allows users to:

- 🔍 Ask questions on general topics  
- 🧠 Get answers from a **locally stored knowledge base (ChromaDB)**  
- 💬 Automatically fallback to **Google Gemini API** if no match is found  
- 📚 Easily **add new knowledge** by topic (from Wikipedia)

The entire system is built with an interactive **Streamlit** UI, and it continuously improves its answers by learning from newly added topics.

---

## 🚀 Built With

- 🖥️ `Streamlit` – UI for chatbot interaction  
- 📚 `Wikipedia` + `sentence-transformers` – For topic data fetching and embedding  
- 🧠 `ChromaDB` – Vector store to hold embeddings  
- 🤖 `google-generativeai` – Gemini API fallback  
- 🔐 `python-dotenv` – For managing API keys securely

---

## ✨ Features

✅ Ask factual/general questions (e.g., "Who built the Taj Mahal?")  
✅ Embed and store new topics locally and save raw content in the `data/` folder
✅ Vector search for quick answer matching  
✅ Fallback to Gemini API if no match is found  
✅ Maintains chat history in-app
✅ Clean, user-friendly interface via Streamlit

---

## 🧱 Folder Structure

```bash
Task3_KnowledgeUpdater/
│
├── src/
│   ├── embedder.py        # Wikipedia data fetch + embedding
│   ├── vector_store.py    # ChromaDB operations
│   └── gemini_bot.py      # Gemini fallback handling
│
├── app.py                 # Streamlit UI and main logic
├── requirements.txt       # Required packages
├── .env                   # API key (Git-ignored)
├── .gitignore             # Git ignore for __pycache__ and .env
└── README.md              # Project overview
```

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/Dynamic_Knowledge_Updater.git
cd Dynamic_Knowledge_Updater

```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
# OR
source venv/bin/activate  # For Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API Key in `.env`

Create a `.env` file and paste:

```env
GOOGLE_API_KEY=your_gemini_key_here
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 💡 Example Usage

💬 **Q: Where is the Taj Mahal located and what is it made of and who built it?**  
📖 A: The Taj Mahal is located in Agra, India. It is made of white marble and was commissioned by Shah Jahan.

💬 **Q: Which is the sixth planet?**  
📖 A: Saturn is the sixth planet from the Sun.

💬 **Q: Who won the first FIFA World Cup?**  
📖 A: Uruguay won the first FIFA World Cup in 1930.

---

## 📦 requirements.txt

```bash
streamlit
wikipedia
sentence-transformers
chromadb
google-generativeai
python-dotenv
```

---

## 📌 Notes

- Stores and reuses knowledge intelligently  
- Automatically reaches out to Gemini when local info is missing  
- Raw Wikipedia content for each topic is saved in the data/ folder automatically
- `.env` is excluded via `.gitignore` to protect API keys  
- Runs locally with no external database setup required

---

## 👨‍💻 Developed by

**Pratham Modi**  
📅 September 2025
