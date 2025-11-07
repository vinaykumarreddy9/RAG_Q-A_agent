from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState, StateGraph, END
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.documents import Document
import os
import posthog

load_dotenv()

DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

model = ChatGroq(model=GROQ_MODEL)

# MessageState that manages the holds the messages in the workflow.
class AgentState(MessagesState):
    user_query : str
    planner_decision : bool = True
    retrieved_docs : Optional[List[Document]]
    evaluator_response : int
    final_answer : str


PLANNER_PROMPT_TEMPLATE = """
You are an intelligent gatekeeper for a Q&A system.
Your purpose is to determine if a user's question is relevant to the available knowledge base.

The knowledge base contains specific information about:
- The environmental benefits of renewable energy (e.g., reducing greenhouse gases).
- The market and industrial trends of renewable energy (e.g., investment, trends, job creation).
- The role of renewables in combating climate change.

Analyze the following user question and decide if it falls within this scope.
For example, questions about solar panels, wind turbines, or energy costs are relevant.
Questions about unrelated topics like cooking recipes, sports, or general history are NOT relevant.

User Question: "{query}"

Can this question be answered using the knowledge base described above?
Respond with only the single word 'True' or 'False'.
"""



async def planning_agent(state : AgentState) -> AgentState:
    """Analyzes the user query and decides whether the query is answerable or not."""

    print("----Planning node started----")

    query = state["user_query"]
    
    PLANNER_PROMPT = PromptTemplate(
        template=PLANNER_PROMPT_TEMPLATE,
        input_variables=["query"]
    ).format(query = query)
    
    response = await model.ainvoke(PLANNER_PROMPT)
    
    print("----planning completed----")

    if "True" in response.content:
        return {
            "planner_decision" : True
        }
    else:
        return {
            "planner_decision" : False,
            "final_answer" : "Query is NOT relevant. Ending workflow.",
            "evaluator_response" : 0
        }
    

def planning_router(state):
    """
    Routes the workflow based on the 'plan_decision'.
    """
    print("--- 🚦 EXECUTING ROUTER ---")
    decision = state.get("planner_decision")

    if decision:
        print("--- ROUTE: Decision is TRUE. Routing to RETRIEVE. ---")
        return "retrieve_agent"
    else:
        print("--- ROUTE: Decision is FALSE. Routing to END. ---")
        return END
        
def no_capture(*args, **kwargs):
    pass

async def retrieve_agent(state : AgentState) -> AgentState:
    """retrieves document from the vector store based on user query"""
    print("----Starting Retriever agent----")

    os.environ["CHROMA_TELEMETRY"] = "0"
    posthog.capture = no_capture

    question = state.get("user_query", "")

    # 1. Initialize the embedding function
    # This MUST be the same model used in ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # 2. Load the existing vector store
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = await retriever.ainvoke(question)
    formatted_context = "\n\n---\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )
    
    print(f"--- RETRIEVED DOCUMENTS ---")
    
    return {
        "retrieved_docs" : retrieved_docs
    }


ANSWER_PROMPT_TEMPLATE = """
You are a precision Q&A engine. Your sole purpose is to accurately answer the user's question using **only the specific, relevant pieces of information** from the provided context.

**Core Task:**
1.  First, carefully analyze the user's **Question** to understand exactly what is being asked.
2.  Next, scan the entire **Context** to locate the exact sentences or facts that directly address the question.
3.  Finally, construct your **Answer** by synthesizing *only these relevant pieces of information* into a clear and concise response.

**Strict Rules:**
- **DO NOT** summarize the entire context. Your goal is to answer the question, not to provide a general overview of the documents.
- **DO NOT** include information from the context that is interesting but not directly relevant to the question.
- **DO NOT** use any external knowledge. If the answer is not in the context, you MUST state: "Based on the provided documents, I cannot answer this question."
- **DO NOT** add any prefixes like "In the provided context" or "According to the context" to the responses.

---
**Context:**
{context}
---
**Question:**
{question}
---

**Answer:**
"""


async def answering_agent(state : AgentState) -> AgentState:
    """Generalizes the retrived documents and converts it in to user understandable format."""
    print("----Starting answering agent----")

    question = state.get("user_query", "")
    context = state.get("retrieved_docs", "")

    ANSWERING_PROMPT = PromptTemplate(
        template=ANSWER_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    ).format(context = context, question = question)

    if not context:
        print("--- No context found. Cannot generate answer. ---")
        return {
            "final_answer" : "Based on the provided documents, I cannot answer this question.",
            "evaluator_response" : 0
        }
    
    response = await model.ainvoke(ANSWERING_PROMPT)
    print("----Answering agent ended")
    return {
        "final_answer" : response.content
    }


EVALUATOR_PROMPT_TEMPLATE = """
You are a strict and impartial AI evaluator. Your task is to evaluate the relevance of a given Answer to a specific Question."

You will provide a score from 0 to 100, where 100 represents perfect relevance and 0 represents complete irrelevance.

- A score of 100 means the answer directly and completely addresses the user's question.
- A score of 75 means the answer is highly relevant but might contain minor extra details.
- A score of 50 means the answer is on the same general topic but does not answer the specific question asked.
- A score of 0 means the answer is about a completely different topic.

Analyze the following Question and Answer.

---
**Question:**
{question}
---
**Answer:**
{answer}
---

Based on your evaluation, provide the relevance score.
**Respond ONLY with an integer between 0 and 100. Do not provide any explanation, text, or justification. Just the number.**
"""

async def evaluator_agent(state : AgentState) -> AgentState:
    """Evaluates the response by checking the relevence of the generated answer to the user query."""
    
    print("----Starting evaluator agent----")

    answer = state.get("final_answer", "")
    question = state.get("user_query", "")

    EVALUATOR_PROMPT = PromptTemplate(
        template=EVALUATOR_PROMPT_TEMPLATE,
        input_variables=["question", "answer"]
    ).format(question = question, answer = answer)

    if not answer:
        print("--- ⚠️ No answer to reflect on. Skipping. ---")
        return {
            "evaluator_response" : 0
        }
    response = await model.ainvoke(EVALUATOR_PROMPT)

    print("----Evaluation completed----")

    return {
        "evaluator_response" : response.content
    }



workflow = StateGraph(AgentState)
workflow.add_node(planning_agent, "planning_agent")
workflow.add_node(retrieve_agent, "retrieve_agent")
workflow.add_node(answering_agent, "answering_agent")
workflow.add_node(evaluator_agent, "evaluator_agent")

workflow.set_entry_point("planning_agent")
workflow.add_conditional_edges("planning_agent",planning_router, {"retrieve_agent" : "retrieve_agent", END : END})
workflow.add_edge("retrieve_agent","answering_agent")
workflow.add_edge("answering_agent", "evaluator_agent")
workflow.add_edge("evaluator_agent",END)

graph = workflow.compile()