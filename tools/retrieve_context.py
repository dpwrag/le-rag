from langchain.tools import tool
# By default, the function’s docstring becomes the tool’s description that helps the model understand when to use it:


def create_retrieve_context(vector_store):
    """Creates the retrieve context tool

    :param vector_store: Your vector store
    :return: the retrieve context tool function
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve food menu name, and, if available, its description"""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context
