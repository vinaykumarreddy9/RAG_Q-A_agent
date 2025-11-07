# 🤖 Renewable Energy Q&A Agent (LangGraph + Streamlit)

An **asynchronous Retrieval-Augmented Generation (RAG)** system built using **LangGraph**, **Groq LLMs**, and **ChromaDB**, designed to answer user questions related to **renewable energy** while automatically evaluating response quality.

This project demonstrates how to architect a **multi-agent AI pipeline** — from query analysis to document retrieval, answer generation, and self-evaluation — all visualized through a **Streamlit web interface**.

---

## 🚀 Project Overview

The **Renewable Energy Q&A Agent** is a LangGraph-based workflow that:
1. Determines if a user’s query is **relevant** to renewable energy topics.
2. Retrieves contextually relevant information from a **Chroma vector database**.
3. Generates a concise, accurate answer using a **Groq LLM**.
4. Evaluates the response based on **relevance** and **faithfulness** metrics.
5. Displays results and context via a **Streamlit UI**.

---

## 🧩 System Architecture

### 🧠 1. `planning_agent`
Analyzes the user’s query to decide whether it falls within the renewable energy domain.

- **Input:** User query  
- **Output:** Planner decision (`True` or `False`)

---

### 📚 2. `retrieve_agent`
Fetches relevant text chunks from **ChromaDB** based on semantic similarity.

- **Embedding Model:** `all-MiniLM-L6-v2` (Hugging Face)  
- **Vector Store:** `Chroma` persisted in `/chroma_db`

---

### 📝 3. `answering_agent`
Generates a focused answer **only from retrieved documents**, following strict no-hallucination rules.

- **LLM:** Groq’s `llama-3.3-70b-versatile`
- **Goal:** Produce a precise answer with no external knowledge.

---

### 🧮 4. `evaluator_agent`
Scores the generated answer using the same LLM as an evaluator.

- **Metric:** Relevance (0–100)
- **Purpose:** Automated response quality evaluation

---

### ⚡ Workflow Graph
```
User Query
   │
   ▼
[Planning Agent] ──> [Retrieve Agent] ──> [Answering Agent] ──> [Evaluator Agent]
       │                                                                │
       ▼                                                                ▼
      [END]  ← if query not relevant                                  [END]
       
```

---

## 🖥️ Streamlit Interface

The **Streamlit frontend (`app.py`)** provides a clean, interactive UI to:
- Input questions
- View the agent’s answer
- Inspect retrieved documents
- Check automatic evaluation scores

---

## 🧱 Project Structure

```
📦 Renewable-Energy-QA-Agent
├── agent.py              # Core LangGraph agent workflow
├── app.py                # Streamlit frontend for interaction
├── data_ingestion.py     # Data loader and Chroma vector store creator
├── evaluation.py         # Batch evaluation script
├── requirements.txt      # Required dependencies
├── .env.example          # Example environment variable configuration
├── .gitignore            # Ignored files and folders
└── chroma_db/            # Persisted vector database (auto-generated)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vinaykumarreddy9/RAG_Q-A_agent.git
cd renewable-energy-qa-agent
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # for macOS/Linux
venv\Scripts\activate       # for Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_langsmith_api_key_here
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=Renewable-Energy-QA
GROQ_API_KEY=your_groq_api_key_here
```

*(You can rename `.env.example` and update your API keys.)*

---

## 🧠 Building the Knowledge Base

Run the ingestion script to embed and store your documents in Chroma:

```bash
python data_ingestion.py
```

This script:
- Loads `.txt` files from the `/data` folder
- Splits them into 1000-character chunks
- Embeds using MiniLM-L6-v2
- Saves embeddings in `chroma_db/`

---

## 💬 Running the Q&A Agent (Streamlit App)

```bash
streamlit run app.py
```

Then open your browser at:
👉 [http://localhost:8501](http://localhost:8501)

You can now:
- Type a renewable-energy-related question
- View the AI’s response, score, and supporting context

---

## 🧪 Running Evaluation Experiments

To test the agent’s performance against a curated dataset, run:

```bash
python evaluation.py
```

This script:
- Uses predefined questions and ground-truth answers
- Invokes the entire agent pipeline
- Scores each result based on *faithfulness* and *relevance*

Example output:
```
Question: How many people work in renewable energy?
Agent’s Answer: ~12 million people worldwide.
Judge’s Evaluation: {"faithfulness": 5, "relevance": 5}
```

---

## 📈 Integration with LangSmith

LangSmith is automatically integrated for trace visualization and debugging.

To view all requests/responses:

1. Enable tracing in `.env`
2. Run your agent
3. Go to [https://smith.langchain.com/projects](https://smith.langchain.com/projects)

You’ll see:
- Every agent node call  
- Prompts and model responses  
- Timing and token usage  

---

## 🧰 Technologies Used

| Category | Tool / Library |
|-----------|----------------|
| Framework | LangGraph |
| LLM | Groq `llama-3.3-70b-versatile` |
| Vector DB | Chroma |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| UI | Streamlit |
| Evaluation | Automated LLM-based scoring |
| Observability | LangSmith |

---

## 🧠 Key Features

✅ Fully asynchronous agent pipeline  
✅ Automatic document retrieval and contextual response  
✅ Self-evaluation with relevance scoring  
✅ Streamlit-based interactive frontend  
✅ LangSmith-powered trace logging  
✅ Reproducible ingestion and evaluation scripts

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify it for research or educational purposes.

---

## ✨ Author

**Developed by:** [Kovvuri Vinay Kumar Reddy]  
📧 Email: [vinaykumarreddy8374@gmail.com]  
🌐 GitHub: [https://github.com/vinaykumarreddy9](https://github.com/yourusername)
