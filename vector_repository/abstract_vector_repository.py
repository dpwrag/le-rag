from abc import ABC, abstractmethod
from langchain_core.documents import Document


class AbstractVectorRepository(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Split and add documents to the vector store

        :param documents: le documents
        :return: list of document_ids
        """
        pass

    @abstractmethod
    def similarity_search(self, query, k) -> list[Document]:
        """Performs similarity search

        :param query: The query
        :param k: The specified k nearest neighbour
        :return: list of Documents
        """
        pass
