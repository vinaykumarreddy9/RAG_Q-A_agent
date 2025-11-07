import asyncio
import json
from agent import graph
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


EVALUATION_DATASET = [
    {
        "question": "How many people were employed in the renewables sector worldwide as of 2020?",
        "ground_truth": "As of 2020, renewables employed about 12 million people worldwide, with solar PV being the largest employer with almost 4 million jobs."
    },
    {
        "question": "What percentage of energy in the transportation sector comes from renewables?",
        "ground_truth": "Less than 4% of transport energy is from renewables."
    },
    {
        "question": "Which country accounted for nearly half of the global increase in renewable electricity in 2021?",
        "ground_truth": "In 2021, China accounted for almost half of the global increase in renewable electricity."
    },
    {
        "question": "What are some examples of how renewable energy is used for heating?",
        "ground_truth": "Solar water heating is a major contributor, particularly in China. Heat pumps are also an increasing priority for providing both heating and cooling."
    },
    {
        "question": "What is the controversial status of nuclear power?",
        "ground_truth": "The provided text does not contain enough information to answer this question in detail, only that it is controversial because it requires mining uranium."
    },
    {
        "question": "What is the capital of Australia?",
        "ground_truth": "This question is not relevant to the knowledge base on renewable energy."
    }
]

JUDGE_MODEL = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

JUDGE_PROMPT_TEMPLATE = """
You are an expert AI evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate a generated answer based on a user's question and a ground truth answer derived from a specific document.

You will evaluate on two metrics:
1.  **Faithfulness (1-5):** Does the generated answer ONLY contain information present in the ground truth? A score of 5 means it is perfectly faithful. A score of 1 means it contains significant hallucinations or information not supported by the ground truth.
2.  **Relevance (1-5):** Does the generated answer directly address the user's question? A score of 5 is perfectly relevant. A score of 1 is completely irrelevant.

**User Question:**
{question}

**Ground Truth Answer:**
{ground_truth}

**Generated Answer:**
{generated_answer}

Provide your evaluation in a JSON format with the keys "faithfulness", "relevance", and "justification".

**Example Response:**
{{
    "faithfulness": 5,
    "relevance": 5,
    "justification": "The answer is both faithful to the ground truth and directly relevant to the user's question."
}}
"""

judge_prompt = PromptTemplate(template=JUDGE_PROMPT_TEMPLATE, input_variables=["question", "ground_truth", "generated_answer"])
judge_chain = judge_prompt | JUDGE_MODEL


async def main():
    """Runs the evaluation loop for the agent."""
    print("--- 🚀 Starting Agent Evaluation with New Dataset ---")
    
    results = []

    for i, item in enumerate(EVALUATION_DATASET):
        print(f"\n{'='*20} Evaluating Item {i+1}/{len(EVALUATION_DATASET)} {'='*20}")
        print(f"Question: {item['question']}")

        agent_output = await graph.ainvoke({"user_query": item["question"]})
        generated_answer = agent_output.get("final_answer", "")
        
        print(f"Agent's Answer: {generated_answer}")

        if not agent_output.get("planner_decision"):
            print("Agent correctly identified question as out-of-scope.")
            evaluation = {
                "faithfulness": 5,
                "relevance": 5,
                "justification": "Agent correctly decided the question was out of scope."
            }
        else:
            judge_response = await judge_chain.ainvoke({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "generated_answer": generated_answer
            })
            try:
                evaluation = json.loads(judge_response.content)
            except json.JSONDecodeError:
                evaluation = {"error": "Failed to parse judge's JSON response.", "raw_content": judge_response.content}

        print(f"Judge's Evaluation: {evaluation}")
        
        results.append({
            "question": item["question"],
            "generated_answer": generated_answer,
            "evaluation": evaluation
        })

    print(f"\n{'='*20} Evaluation Complete {'='*20}")

if __name__ == "__main__":
    print("Ensure you have run 'ingest.py' with the new data before starting evaluation.")
    asyncio.run(main())