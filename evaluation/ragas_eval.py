import getpass
import os

import pandas as pd
from datasets import Dataset
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from ollama import AsyncClient
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.embeddings.base import embedding_factory
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import llm_factory
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from agents import NLIAgent, RetrieveContextAgent
from agents.states import AgentState, SelectedMenu
from infrastructure import get_qdrant_client

COLLECTION_NAME = "menu"
TOP_K = 4

LLM_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M "
model = ChatOllama(
    model=LLM_MODEL,
    temperature=1.0,
    max_tokens=None,
    timeout=69_420,
    num_predict=512,
    max_retries=2,
)


client = get_qdrant_client()

encoder = FastEmbedEmbeddings()

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=encoder,
)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})


EVAL_DATA = [
    {
        "question": "I'm a muslim and I want healthy diet. What menu should I eat?",
        "ground_truth": (
            "The user should eat halal-friendly healthy foods such as "
            "grilled chicken, seafood, rice bowls, salads, or vegetable dishes "
            "that do not contain pork or bacon."
        ),
    },
    # {
    #     "question": "What spicy chicken dishes are available?",
    #     "ground_truth": (
    #         "Available spicy chicken dishes include spicy wings, hot chicken, "
    #         "buffalo chicken, and spicy grilled chicken meals."
    #     ),
    # },
    # {
    #     "question": "Which restaurants have seafood menu items?",
    #     "ground_truth": (
    #         "Restaurants with seafood menu items include places serving shrimp, "
    #         "fish, salmon, crab, or other seafood dishes."
    #     ),
    # },
    # {
    #     "question": "What low calorie meals can I order?",
    #     "ground_truth": (
    #         "Low calorie meals include salads, grilled proteins, vegetable bowls, "
    #         "and light seafood dishes."
    #     ),
    # },
    # {
    #     "question": "Which restaurant has steak dishes?",
    #     "ground_truth": (
    #         "Restaurants with steak dishes include steakhouse or grill restaurants "
    #         "serving filet mignon, ribeye, sirloin, or hibachi steak."
    #     ),
    # },
    # {
    #     "question": "Can you recommend high protein meals?",
    #     "ground_truth": (
    #         "High protein meals include grilled chicken, steak, seafood, "
    #         "egg-based dishes, and protein bowls."
    #     ),
    # },
    # {
    #     "question": "What vegetarian food is available?",
    #     "ground_truth": (
    #         "Vegetarian foods include salads, vegetable bowls, meat-free pasta, "
    #         "and plant-based dishes without meat."
    #     ),
    # },
    # {
    #     "question": "Which restaurants serve wings?",
    #     "ground_truth": (
    #         "Wing restaurants and sports bars serve chicken wings including "
    #         "buffalo wings and boneless wings."
    #     ),
    # },
]

questions = []
answers = []
contexts = []
ground_truths = []


for sample in EVAL_DATA:
    question = sample["question"]
    ground_truth = sample["ground_truth"]

    print(f"Question: {question}")

    retrieved_docs = retriever.invoke(question)

    retrieved_contexts = []

    for idx, doc in enumerate(retrieved_docs):
        text = doc.page_content

        retrieved_contexts.append(text)

        print(f"\n--- Context {idx + 1} ---")
        print(text[:500])

    try:
        retrieve_context_agent = RetrieveContextAgent(model, vector_store=vector_store)
        nli_agent = NLIAgent(model)

        graph = (
            StateGraph(AgentState)
            .add_node("retrieve_context", retrieve_context_agent.act)
            .add_node("nli_agent", nli_agent.act)
            .add_edge(START, "retrieve_context")
            .add_edge("retrieve_context", "nli_agent")
            .add_edge("nli_agent", END)
            .compile(checkpointer=InMemorySaver())
        )
        final_state = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "session-1"}},
        )
        answer = ""
        for menu in final_state["selected_menu_list"].menu_list:
            a = f"{menu.name}: {menu.justification} & "
            answer += a

    except Exception as e:
        print(f"Error: {e}")
        print(e)
        answer = "Generation failed."

    print("\n--- Generated Answer ---")
    print(answer)

    print("\n--- Ground Truth ---")
    print(ground_truth)

    questions.append(question)
    answers.append(answer)
    contexts.append(retrieved_contexts)
    ground_truths.append(ground_truth)


dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

results_df = pd.DataFrame(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

results_df.to_csv(
    "ragas_raw_results.csv",
    index=False,
)

print("\nSaved raw evaluation data to ragas_raw_results.csv")

print("\nRunning Ragas evaluation...\n")
ollama_client = AsyncClient()

evaluator_llm = LangchainLLMWrapper(model)
evaluator_embeddings = LangchainEmbeddingsWrapper(FastEmbedEmbeddings())

# Correct initialization
faithfulness_metric = Faithfulness(llm=evaluator_llm)
answer_relevancy_metric = AnswerRelevancy(
    llm=evaluator_llm, embeddings=evaluator_embeddings
)
context_precision_metric = ContextPrecision(llm=evaluator_llm)
context_recall_metric = ContextRecall(llm=evaluator_llm)
metrics_list = [
    faithfulness_metric,
    answer_relevancy_metric,
    context_precision_metric,
    context_recall_metric,
]

result = evaluate(dataset=dataset, metrics=metrics_list)

print("Ragas Scores")

print(result)

result_df = result.to_pandas()

print("\nDetailed Scores:\n")
print(result_df)

result_df.to_csv(
    "ragas_scores.csv",
    index=False,
)

print("\nSaved Ragas scores to ragas_scores.csv")
