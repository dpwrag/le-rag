import logging

from langchain_core.messages import HumanMessage

from setup import create_graph


logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.INFO)

graph = create_graph()

query = "I'm a muslim and I want to eat healthy cuisine, what should I eat?"
for event in graph.stream(
    {"messages": [HumanMessage(content=query)]},
    stream_mode="debug",  # emits {node_name: state_delta} instead
    config={"configurable": {"thread_id": "session-1"}},
):
    print(event)
final_state = graph.get_state({"configurable": {"thread_id": "session-1"}})

# Access your specific attributes
print(final_state.values["chain_of_thought"])
# final_state = graph.invoke(
#     {"messages": [{"role": "user", "content": query}]},
#     config={"configurable": {"thread_id": "session-1"}},
# )

# final_message = final_state["messages"][-1]

# print("\n--- [Final Result] ---")
# raw_content = final_message.content

# if isinstance(raw_content, list):
#     for block in raw_content:
#         if block.get("type") == "text":
#             final_text = block["text"]
#             break
# else:
#     final_text = raw_content

# print(final_text)
