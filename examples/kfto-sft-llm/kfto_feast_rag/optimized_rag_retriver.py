from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union, Tuple

import numpy
import numpy as np
import torch
from feast import FeatureStore, FeatureView
from sentence_transformers import SentenceTransformer
from transformers import RagRetriever


class VectorStore(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int):
        pass


class FeastVectorStore(VectorStore):
    def __init__(self, store_repo_path: str, rag_view: FeatureView, features: List[str]):
        self._store = None  # Lazy load
        self._store_repo_path = store_repo_path
        self.rag_view = rag_view
        self.features = features

    @property
    def store(self):
        if self._store is None:
            from feast import FeatureStore
            self._store = FeatureStore(repo_path=self._store_repo_path)
            self._store.apply([self.rag_view])
        return self._store

    def query(
            self,
            query_vector: Optional[np.ndarray] = None, # Input from FeastRAGRetriever will be np.ndarray
            query_string: Optional[str] = None,
            top_k: int = 10,
    ):
        distance_metric = "COSINE" if query_vector is not None else None

        milvus_query_arg = None
        if query_vector is not None:
            if isinstance(query_vector, torch.Tensor):
                print("WARNING: FeastVectorStore.query received torch.Tensor for query_vector. Converting to NumPy.")
                query_vector_np_ready = query_vector.detach().cpu().numpy()
            elif isinstance(query_vector, np.ndarray):
                query_vector_np_ready = query_vector
            else:
                raise TypeError(f"query_vector in FeastVectorStore.query has unexpected type: {type(query_vector)}. Expected torch.Tensor or np.ndarray.")

            temp_vector = query_vector_np_ready.astype(np.float32)

            if temp_vector.ndim > 1:
                temp_vector = temp_vector.flatten()
            temp_vector_list = temp_vector.tolist()
            milvus_query_arg = temp_vector_list

        results = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=milvus_query_arg,
            query_string=query_string,
            top_k=top_k,
            distance_metric=distance_metric,
        )
        return results


# Dummy index - an index is required by the HF Transformers RagRetriever class
class FeastIndex:
    def __init__(self):
        pass

    def get_top_docs(self, query_vectors, n_docs=5):
        raise NotImplementedError("get_top_docs is not yet implemented.")

    def get_doc_dicts(self, doc_ids):
        raise NotImplementedError("get_doc_dicts is not yet implemented.")


class FeastRAGRetriever(RagRetriever):
    VALID_SEARCH_TYPES = {"text", "vector", "hybrid"}

    def __init__(
            self,
            question_encoder_tokenizer,
            question_encoder,
            generator_tokenizer,
            generator_model,
            feast_repo_path: str,
            search_type: str,
            config,
            index,
            format_document: Optional[Callable[[Dict], str]] = None,
            id_field: str = "",
            query_encoder_model: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2",
            **kwargs,
    ):
        if search_type.lower() not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Unsupported search_type {search_type}. "
                f"Must be one of: {self.VALID_SEARCH_TYPES}"
            )
        super().__init__(
            config=config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
            init_retrieval=False,
            **kwargs,
        )
        self.question_encoder = question_encoder
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.feast_repo_path = feast_repo_path
        self.search_type = search_type.lower()
        self.format_document = format_document or self._default_format_document
        self.id_field = id_field

        # Initialize these lazily
        self._query_encoder = None
        self._feast_store = None
        self._vector_store = None
        self._query_encoder_model_name = query_encoder_model if isinstance(query_encoder_model, str) else None


    @property
    def query_encoder(self):
        if self._query_encoder is None:
            if self._query_encoder_model_name:
                self._query_encoder = SentenceTransformer(self._query_encoder_model_name)
            else:
                raise ValueError("query_encoder_model_name must be set to initialize SentenceTransformer lazily.")
        return self._query_encoder

    @property
    def feast_store(self):
        # Initialize FeatureStore lazily
        if self._feast_store is None:
            from feast import FeatureStore
            self._feast_store = FeatureStore(repo_path=self.feast_repo_path)
        return self._feast_store

    @property
    def vector_store(self):
        # Initialize FeastVectorStore lazily
        if self._vector_store is None:
            from feast_setup.ragproject_repo import wiki_passage_feature_view
            self._vector_store = FeastVectorStore(
                store_repo_path=self.feast_repo_path,
                rag_view=wiki_passage_feature_view,
                features=["wiki_passages:passage_text", "wiki_passages:embedding", "wiki_passages:passage_id"]
            )
        return self._vector_store

    def __call__(
            self,
            question_input_ids: Optional[torch.Tensor] = None,
            question_hidden_states: Optional[Union[torch.Tensor, np.ndarray]] = None,
            n_docs: int = 5,
            return_tensors: Optional[str] = None,
            **kwargs,
    ) -> Dict[str, Union[torch.Tensor, List[Dict]]]:

        if question_hidden_states is None:
            raise ValueError("`question_hidden_states` must be provided to FeastRAGRetriever's __call__ method.")

        target_device = None
        if question_input_ids is not None and question_input_ids.is_cuda:
            target_device = question_input_ids.device
        elif isinstance(question_hidden_states, torch.Tensor) and question_hidden_states.is_cuda:
            target_device = question_hidden_states.device
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            target_device = torch.device("cpu")

        if isinstance(question_hidden_states, np.ndarray):
            question_hidden_states_tensor = torch.tensor(question_hidden_states, dtype=torch.float32).to(target_device)
        elif isinstance(question_hidden_states, torch.Tensor):
            question_hidden_states_tensor = question_hidden_states.to(target_device)
        else:
            raise TypeError(f"question_hidden_states has unexpected type: {type(question_hidden_states)}. Expected torch.Tensor or np.ndarray.")

        print(f"DEBUG: __call__() processed question_hidden_states type: {type(question_hidden_states_tensor)}, device: {question_hidden_states_tensor.device}")

        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states_tensor, n_docs)

        context_texts = [self.format_document(doc) for doc in docs]

        tokenized_context = self.generator_tokenizer(
            context_texts,
            padding="max_length",
            truncation=True,
            max_length=self.generator_tokenizer.model_max_length,
            return_tensors="pt",
        )

        context_input_ids = tokenized_context["input_ids"]
        context_attention_mask = tokenized_context["attention_mask"]

        # Ensure all output tensors are on the correct target device
        context_input_ids = context_input_ids.to(target_device)
        context_attention_mask = context_attention_mask.to(target_device)
        retrieved_doc_embeds = retrieved_doc_embeds.to(target_device)
        doc_ids = doc_ids.to(target_device)

        doc_scores_list = [doc["score"] for doc in docs]
        doc_scores = torch.tensor(doc_scores_list, dtype=torch.float32).to(target_device)

        return_dict = {
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "doc_scores": doc_scores,
            "doc_ids": doc_ids,
            "docs": docs,
            "retrieved_doc_embeds": retrieved_doc_embeds,
            "question_input_ids": question_input_ids.to(target_device) if question_input_ids is not None else None,
            "question_hidden_states": question_hidden_states_tensor,
        }

        print(f"DEBUG: __call__() returning type: {type(return_dict)}")
        print(f"DEBUG: __call__() returning keys: {return_dict.keys()}")
        for k, v in return_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  Return Tensor {k} DTYPE: {v.dtype}, SHAPE: {v.shape}, DEVICE: {v.device}")

        return return_dict


    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int = 10):
        # Pooling logic (converting to 1D vector)
        if question_hidden_states.ndim == 3 and question_hidden_states.shape[0] == 1:
            pooled_query_vector = question_hidden_states[0].mean(axis=0)
        elif question_hidden_states.ndim == 2 and question_hidden_states.shape[0] == 1:
            pooled_query_vector = question_hidden_states[0]
        elif question_hidden_states.ndim == 1:
            pooled_query_vector = question_hidden_states
        else:
            raise ValueError(f"Unexpected question_hidden_states shape from RAG model: {question_hidden_states.shape}")

        query_vector_for_feast = pooled_query_vector

        print(f"DEBUG: FeastRAGRetriever.retrieve called by RAG model internal forward: ")
        print(f"  Converted query_vector_for_feast DTYPE: {query_vector_for_feast.dtype}, SHAPE: {query_vector_for_feast.shape}")

        feast_results = self.vector_store.query(
            query_vector=query_vector_for_feast,
            query_string=None,
            top_k=n_docs
        )

        documents_dict = feast_results.to_dict()
        # print(f"DEBUG: FeastRAGRetriever.retrieve: feast documents_dict: {documents_dict}")
        retrieved_doc_embeds = torch.tensor(documents_dict["embedding"], dtype=torch.float32)
        if retrieved_doc_embeds.ndim == 2:
            retrieved_doc_embeds = retrieved_doc_embeds.unsqueeze(0)
        passage_ids_str = documents_dict.get("passage_id", [])
        distances = documents_dict.get("distance", [])
        doc_ids_int_list = [int(id_str) for id_str in passage_ids_str]
        doc_ids = torch.tensor(doc_ids_int_list, dtype=torch.long)

        docs = []
        num_retrieved = len(documents_dict.get("passage_text", []))
        for i in range(num_retrieved):
            doc_content = {
                "text": documents_dict["passage_text"][i],
                "id": documents_dict["passage_id"][i],
                "score": distances[i] if i < len(distances) and distances[i] is not None else 0.0,
                "embedding": documents_dict["embedding"][i],
                "title": documents_dict["passage_text"][i]
            }
            docs.append(doc_content)

        # print(f"FeastRAGRetriever.retrieve: RETRIEVED_DOC_EMBEDS: {retrieved_doc_embeds}, DOC_IDS: {doc_ids}, DOCS: {docs}")

        return retrieved_doc_embeds, doc_ids, docs


    def retrieve_from_text(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector_np = self.query_encoder.encode(query, convert_to_numpy=True)

        # print(f"SEARCH TYPE: {self.search_type}\n  QUERY: {query} \n QUERY_VECTOR: {query_vector_np}")

        if self.search_type == "text":
            return self.vector_store.query(query_string=query, top_k=top_k)
        elif self.search_type == "vector":
            return self.vector_store.query(query_vector=query_vector_np, query_string="", top_k=top_k)
        elif self.search_type == "hybrid":
            return self.vector_store.query(
                query_string=query,
                query_vector=query_vector_np,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unsupported search type: {self.search_type}")

    def generate_answer(
            self, query: str, top_k: int = 5, max_new_tokens: int = 100
    ) -> str:
        # Retrieve top-k relevant documents
        documents = self.retrieve_from_text(query, top_k=top_k)
        document_dict = documents.to_dict()

        num_results = len(document_dict["passage_text"])
        contexts = []
        for i in range(num_results):
            passage_text = document_dict["passage_text"][i]
            contexts.append(passage_text)

        context = "\n\n".join(contexts)
        prompt = (
            f"Use the following context to answer the question. Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token

        inputs = self.generator_tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )

        # Get the device of the generator_model
        model_device = self.generator_model.device # Assuming the model is already on a device
        if model_device is None: # Handle cases where model might not be explicitly on a device yet
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)
        output_ids = self.generator_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.generator_tokenizer.pad_token_id,
        )
        return self.generator_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _default_format_document(self, doc: dict) -> str:
        lines = []
        for key, value in doc.items():
            # Skip vectors by checking for long float lists
            if (
                    isinstance(value, list)
                    and len(value) > 10
                    and all(isinstance(x, (float, int)) for x in value)
            ):
                continue
            lines.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        return "\n".join(lines)