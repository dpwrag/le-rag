import getpass
import os
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

from infrastructure import get_qdrant_client
from agents import RetrieveContextAgent

COLLECTION_NAME = "menu"
TOP_K = 4

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
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

agent = RetrieveContextAgent.build(
    model=model,
    vector_store=vector_store,
)

EVAL_DATA = [
    {
        "question": "I'm a muslim and I want healthy diet. What menu should I eat?",
        "ground_truth": (
            "The user should eat halal-friendly healthy foods such as "
            "grilled chicken, seafood, rice bowls, salads, or vegetable dishes "
            "that do not contain pork or bacon."
        ),
    },
    {
        "question": "What spicy chicken dishes are available?",
        "ground_truth": (
            "Available spicy chicken dishes include spicy wings, hot chicken, "
            "buffalo chicken, and spicy grilled chicken meals."
        ),
    },
    {
        "question": "Which restaurants have seafood menu items?",
        "ground_truth": (
            "Restaurants with seafood menu items include places serving shrimp, "
            "fish, salmon, crab, or other seafood dishes."
        ),
    },
    {
        "question": "What low calorie meals can I order?",
        "ground_truth": (
            "Low calorie meals include salads, grilled proteins, vegetable bowls, "
            "and light seafood dishes."
        ),
    },
    {
        "question": "Which restaurant has steak dishes?",
        "ground_truth": (
            "Restaurants with steak dishes include steakhouse or grill restaurants "
            "serving filet mignon, ribeye, sirloin, or hibachi steak."
        ),
    },
    {
        "question": "Can you recommend high protein meals?",
        "ground_truth": (
            "High protein meals include grilled chicken, steak, seafood, "
            "egg-based dishes, and protein bowls."
        ),
    },
    {
        "question": "What vegetarian food is available?",
        "ground_truth": (
            "Vegetarian foods include salads, vegetable bowls, meat-free pasta, "
            "and plant-based dishes without meat."
        ),
    },
    {
        "question": "Which restaurants serve wings?",
        "ground_truth": (
            "Wing restaurants and sports bars serve chicken wings including "
            "buffalo wings and boneless wings."
        ),
    },
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
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

    except Exception as e:
        print(f"Error: {e}")

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

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

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
