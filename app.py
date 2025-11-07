import streamlit as st
import asyncio
from agent import graph


st.set_page_config(
    page_title="Async Q&A Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Asynchronous Q&A Agent for Renewable Energy")
st.markdown("""
This agent uses a RAG (Retrieval-Augmented Generation) pipeline built with LangGraph to answer your questions. 
The entire workflow, from planning to evaluation, runs asynchronously.
""")
st.info("The knowledge base contains information on renewable energy and its market & trends.")



user_question = st.text_input(
    "Ask your question:", 
    placeholder="e.g., What are the market trends for solar energy?"
)

if st.button("Get Answer"):
    if user_question:
        with st.spinner("The agent is processing your request..."):
            try:
                async def run_graph():
                    """Defines and runs the async graph invocation."""
                    initial_state = {"user_query": user_question}
                    return await graph.ainvoke(initial_state)

                final_state = asyncio.run(run_graph())

                st.divider()

                final_answer = final_state.get("final_answer", "No answer was generated.")
                score = final_state.get("evaluator_response", 0)
                
                st.subheader("📝 Agent's Answer:")
                st.markdown(final_answer)

                if final_state.get("planner_decision", False):
                    try:
                        score_value = int(str(score).strip())
                        st.metric(label="Relevance Score", value=f"{score_value}/100")
                    except (ValueError, TypeError):
                        st.warning(f"Could not parse the evaluation score. Raw output: {score}")

                retrieved_docs = final_state.get("retrieved_docs")
                if retrieved_docs:
                    with st.expander("📚 Show Retrieved Context"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Document {i+1}:**")
                            st.text(doc.page_content)
                            st.json(doc.metadata, expanded=False)

            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")
                st.error("Please check the console for more details.")
    else:
        st.warning("Please enter a question to get started.")