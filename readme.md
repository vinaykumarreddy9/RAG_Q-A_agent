## Project Report

### How the Agent Works

This project implements a sophisticated, multi-step AI agent using LangGraph to perform Retrieval-Augmented Generation (RAG). The agent's workflow is fully asynchronous and is structured as a state machine with four distinct nodes:

1.  **Planning Agent:** This initial node acts as an intelligent gatekeeper. It analyzes the user's query and, using a specifically trained prompt, decides whether the question is relevant to the knowledge base on renewable energy. If the query is out of scope, the process terminates immediately, saving computational resources.

2.  **Retrieval Agent:** If the query is deemed relevant, this node is activated. It takes the user's query, generates a vector embedding using Sentence-Transformers, and queries a Chroma vector database to find the top 3 most relevant document chunks from the knowledge base.

3.  **Answering Agent:** The retrieved documents are passed as context to this node. It uses a powerful prompt with strict guardrails to instruct the Groq Llama 3.1 model to synthesize an answer. The prompt explicitly forbids the model from using external knowledge and forces it to base its response exclusively on the provided text, ensuring a factual and grounded answer.

4.  **Evaluator Agent:** As a final quality-control step, the generated answer and the original question are passed to this node. It uses the LLM to perform a self-evaluation, scoring the answer's relevance to the question on a scale of 0 to 100.

### Challenges Faced

Developing this agent presented several key challenges:

*   **Asynchronous Control Flow:** The entire agent was built using Python's `asyncio` for performance. Managing the asynchronous state and ensuring proper execution flow within LangGraph and the Streamlit front-end required careful handling. Integrating the async agent with the synchronous nature of Streamlit was achieved using `asyncio.run()`, which encapsulates the event loop management.

*   **Prompt Engineering and Grounding:** A significant challenge was preventing the LLM in the `answering_agent` from "leaking" its pre-trained knowledge. The initial prompts were too general, leading to answers that were not strictly based on the retrieved documents. This was overcome by iterating on the prompt to include a "Core Task" workflow and strong negative constraints (e.g., "DO NOT summarize the entire context"), which forced the model to act as a "precision Q&A engine" rather than a general chatbot.

*   **Reliable Output Parsing:** Both the `planning_agent` (requiring a "True"/"False" output) and the `evaluator_agent` (requiring an integer score) were prone to returning conversational text instead of the desired format. This was mitigated by creating highly restrictive prompts that explicitly demanded a single-word or single-number response, making the output more reliable for programmatic use.