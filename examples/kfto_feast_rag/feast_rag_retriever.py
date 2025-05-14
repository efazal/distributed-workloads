from typing import List, Optional, Dict
import numpy as np
from feast import FeatureStore, FeatureView
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class VectorStore(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, query_string: Optional[str], top_k: int) -> List[Dict]:
        pass

class FeastVectorStore(VectorStore):
    def __init__(self, store: FeatureStore, rag_view: FeatureView):
        self.store = store
        self.rag_view = rag_view
        self.store.apply([rag_view])

    def query(self, query_vector: np.ndarray, query_string: str, top_k: int = 5):
        results = self.store.retrieve_online_documents_v2(
            features=["wiki_passage_content:text", "wiki_passage_content:embedding"], 
            query=query_vector.tolist(), 
            query_string=query_string, 
            top_k=top_k, 
            distance_metric="L2")
        return results


class FeastRAGRetriever:
    def __init__(self, embedding_model: SentenceTransformer, vector_store: FeastVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_vector = self.embedding_model.encode(query)
        docs = self.vector_store.query(query_vector=query_vector, query_string=query, top_k=top_k)
        return [doc["text"] for doc in docs]


class RAGPipeline:
    def __init__(
        self,
        retriever: FeastRAGRetriever,
        generator_model: str = "ibm/granite-3.2-2b-instruct"
    ):
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = AutoModelForCausalLM.from_pretrained(generator_model)

    def format_prompt(self, query: str, documents: List[str]) -> str:
        context = "\n".join(documents)
        return f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    def generate(self, query: str, top_k: int = 5, max_new_tokens: int = 200) -> str:
        docs = self.retriever.retrieve(query, top_k=top_k)
        prompt = self.format_prompt(query, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
