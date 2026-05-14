from deepeval import evaluate
from deepeval.models import OllamaModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from langchain_core.messages import HumanMessage

from infrastructure import create_graph


def serialize_selected_menu(selected_menu_list) -> str:
    """
    Convert SelectedMenuList into readable text output.
    """
    if not selected_menu_list:
        return ""

    outputs = []

    for menu in selected_menu_list.menu_list:
        outputs.append(
            f"""
Menu: {menu.name}
Description: {menu.description}
Score: {menu.score}
Justification: {menu.justification}
"""
        )

    return "\n".join(outputs)


def serialize_retrieved_docs(retrieved_docs) -> list[str]:
    """
    Convert LangChain Documents into list[str]
    required by DeepEval contextual metrics.
    """
    return [doc.page_content for doc in retrieved_docs]


def run_evaluation(user_query: str):
    graph = create_graph()

    model = OllamaModel(
        model="hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # Invoke graph
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_query)]},
        config={"configurable": {"thread_id": "deepeval-run"}},
    )

    retrieved_docs = result["menu_list"]
    selected_menu_list = result["selected_menu_list"]

    # Final generated recommendation output
    actual_output = serialize_selected_menu(selected_menu_list)

    # Retrieval context
    retrieval_context = serialize_retrieved_docs(retrieved_docs)

    # Optional:
    # Ground truth ideal answer
    # You should build this from curated datasets later
    expected_output = """
Recommend spicy Thai dishes with seafood and avoid dairy.
"""

    # Build test case
    test_case = LLMTestCase(
        input=user_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, include_reason=True, model=model),
        FaithfulnessMetric(threshold=0.7, include_reason=True, model=model),
        ContextualPrecisionMetric(threshold=0.7, include_reason=True, model=model),
        ContextualRecallMetric(threshold=0.7, include_reason=True, model=model),
        ContextualRelevancyMetric(threshold=0.7, include_reason=True, model=model),
    ]

    # Run evaluation
    results = evaluate(
        test_cases=[test_case],
        metrics=metrics,
    )

    return results


if __name__ == "__main__":
    query = "I want spicy seafood noodles with no dairy"

    results = run_evaluation(query)

    print(results)
