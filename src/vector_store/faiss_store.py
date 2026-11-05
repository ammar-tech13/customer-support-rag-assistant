
"""FAISS vector store using Google's embedding API."""
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.config import VECTOR_STORE_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS vector store using Google embeddings."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        store_path: Optional[str] = None
    ):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not configured.")

        self.embedding_model = embedding_model
        self.store_path = Path(store_path) if store_path else VECTOR_STORE_DIR
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=GOOGLE_API_KEY
        )

        self.embedding_dim = 3072
        self.vector_store: Optional[FAISS] = None
        self.metadata_store = {}

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_documents(documents)

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id:
                self.metadata_store[chunk_id] = doc.metadata

        logger.info("Added %s documents to vector store", len(documents))

    def get_retriever(
        self,
        k: int = 5,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Add documents first.")

        kwargs = dict(search_kwargs or {})
        kwargs["k"] = k

        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []

        try:
            kwargs = {"k": k}
            if filter:
                kwargs["filter"] = filter

            return self.vector_store.similarity_search_with_score(
                query, **kwargs
            )
        except Exception as e:
            logger.exception("Error searching vector store: %s", e)
            return []

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )

        formatted_results = []
        for doc, score in results:
            similarity = 1 / (1 + float(score))

            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(similarity),
                "distance": float(score)
            })

        return formatted_results

    def save(self, name: str = "vector_store") -> None:
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return

        self.vector_store.save_local(
            str(self.store_path),
            index_name=name
        )

        metadata_file = self.store_path / f"{name}_metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata_store, f)

        logger.info("Vector store saved to %s", self.store_path)

    def load(self, name: str = "vector_store") -> bool:
        store_file = self.store_path / f"{name}.faiss"

        if not store_file.exists():
            logger.warning("Vector store file not found: %s", store_file)
            return False

        try:
            self.vector_store = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name=name
            )

            metadata_file = self.store_path / f"{name}_metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    self.metadata_store = pickle.load(f)

            logger.info("Vector store loaded from %s", self.store_path)
            return True

        except Exception as e:
            logger.exception("Error loading vector store: %s", e)
            self.vector_store = None
            return False

    def get_stats(self) -> Dict[str, Any]:
        total_chunks = 0

        if self.vector_store is not None:
            total_chunks = self.vector_store.index.ntotal

        return {
            "total_chunks": total_chunks,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "metadata_count": len(self.metadata_store)
        }
