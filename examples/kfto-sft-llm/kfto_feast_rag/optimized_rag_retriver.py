from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union, Tuple, Any

import numpy
import numpy as np
import torch
from feast import FeatureStore, FeatureView
from sentence_transformers import SentenceTransformer
from torch import Tensor
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
            query_vector: Optional[np.ndarray] = None,
            query_string: Optional[str] = None,
            top_k: int = 10,
    ):
        distance_metric = "COSINE" if query_vector is not None else None

        # Ensure query_vector received here is a List[float].
        if query_vector is not None and not isinstance(query_vector, list):
            raise TypeError(f"FeastVectorStore.query received unexpected type for query_vector: {type(query_vector)}. Expected List[float].")
        if query_vector is not None and (len(query_vector) == 0 or not all(isinstance(x, float) for x in query_vector)):
            raise ValueError(f"FeastVectorStore.query received non-float elements in query_vector: {query_vector[:5]}")

        results = self.store.retrieve_online_documents_v2(
            features=self.features,
            query=query_vector,
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
        print("TENSOR DIMS: ", question_hidden_states_tensor.ndim)
        # The goal of below logic is to get 3D tensor (batch_size, 1, hidden_size)
        if question_hidden_states_tensor.ndim == 2: # 2D(batch_size, hidden_size)
            final_query_vector_for_rag = question_hidden_states_tensor.unsqueeze(1) # Result 3D
        elif question_hidden_states_tensor.ndim == 1: # 1D(hidden_size,)
            final_query_vector_for_rag = question_hidden_states_tensor.unsqueeze(0).unsqueeze(1) # Result 3D
        elif question_hidden_states_tensor.ndim == 3 and question_hidden_states_tensor.shape[1] > 1: # (batch_size, sequence_length, hidden_size)
            # Perform mean pooling then unsqueeze
            print("WARNING: question_hidden_states_tensor is 3D with sequence_length > 1. Performing mean pooling.")
            pooled_output = torch.mean(question_hidden_states_tensor, dim=1) # Result: (batch_size, hidden_size)
            final_query_vector_for_rag = pooled_output.unsqueeze(1) # Add the '1' dim: (batch_size, 1, hidden_size)
        elif question_hidden_states_tensor.ndim == 3 and question_hidden_states_tensor.shape[1] == 1:
            # Already 3D
            final_query_vector_for_rag = question_hidden_states_tensor
        else:
            raise ValueError(f"Unexpected question_hidden_states_tensor shape for RAG bmm: {question_hidden_states_tensor.shape}")

        print("QUERY TYPE: ", final_query_vector_for_rag.dtype)
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(final_query_vector_for_rag, n_docs)

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
            "question_hidden_states": final_query_vector_for_rag,
        }

        print(f"DEBUG: __call__() returning type: {type(return_dict)}")
        print(f"DEBUG: __call__() returning keys: {return_dict.keys()}")
        for k, v in return_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  Return Tensor {k} DTYPE: {v.dtype}, SHAPE: {v.shape}, DEVICE: {v.device}")

        return return_dict


    def retrieve(self, question_hidden_states: torch.Tensor, n_docs: int = 10) -> tuple[Tensor, Tensor, list[Any]]:
        query_tensor_cpu_np = question_hidden_states.detach().cpu().numpy()

        if query_tensor_cpu_np.ndim == 3 and query_tensor_cpu_np.shape[0] == 1:
            # For a single query, extract the (hidden_size,) vector
            query_vector_for_feast_np = query_tensor_cpu_np[0].flatten()
        elif query_tensor_cpu_np.ndim == 3 and query_tensor_cpu_np.shape[0] > 1:
            raise ValueError(f"FeastRAGRetriever.retrieve received batch_size > 1. Feast's `query` expects List[float] for a single query. Shape: {query_tensor_cpu_np.shape}")
        else:
            raise ValueError(f"Unexpected question_hidden_states shape from __call__ to retrieve: {question_hidden_states.shape}")

        print(f"DEBUG: FeastRAGRetriever.retrieve called by RAG model internal forward: ")
        print(f"  Input question_hidden_states DTYPE: {question_hidden_states.dtype}, SHAPE: {question_hidden_states.shape}")
        print(f"  Converted query_vector_for_feast DTYPE: {query_vector_for_feast_np.dtype}, SHAPE: {query_vector_for_feast_np.shape}")

        feast_results = self.vector_store.query(
            query_vector=query_vector_for_feast_np.tolist(),
            query_string=None,
            top_k=n_docs
        )

        documents_dict = feast_results.to_dict()

        print(f"DEBUG: documents_dict keys after Feast query: {documents_dict.keys()}")

        passage_texts = documents_dict.get("passage_text", [])
        embeddings = documents_dict.get("embedding", [])
        distances = documents_dict.get("distance", [])
        passage_ids_str = documents_dict.get("passage_id", [])

        num_retrieved = len(passage_texts)
        doc_ids = []

        if not passage_ids_str and num_retrieved > 0:
            print("WARNING: 'passage_id' field is present but empty. Generating sequential dummy IDs.")
            doc_ids_long = torch.tensor(list(range(num_retrieved)), dtype=torch.long)
        elif passage_ids_str:
            try:
                doc_ids_int_list = [int(id_str) for id_str in passage_ids_str]
                doc_ids = torch.tensor(doc_ids_int_list, dtype=torch.long)
                print(f"DEBUG: Converted doc_ids from strings: {doc_ids.tolist()}")
            except ValueError as e:
                print(f"ERROR: Could not convert passage_id strings to integers: {e}. Generating sequential dummy IDs.")
                doc_ids = torch.tensor(list(range(num_retrieved)), dtype=torch.long)
        else:
            doc_ids = torch.tensor([], dtype=torch.long)


        retrieved_doc_embeds = torch.tensor(embeddings, dtype=torch.float32)
        if retrieved_doc_embeds.ndim == 2:
            retrieved_doc_embeds = retrieved_doc_embeds.unsqueeze(0)


        docs = []
        num_retrieved_final = min(len(passage_texts), len(embeddings), len(distances), len(doc_ids))

        for i in range(num_retrieved_final):
            doc_content = {
                "text": passage_texts[i],
                "id": doc_ids[i].item(),
                "score": distances[i] if i < len(distances) and distances[i] is not None else 0.0,
                "embedding": embeddings[i],
                "title": passage_texts[i]
            }
            docs.append(doc_content)

        return retrieved_doc_embeds, doc_ids, docs


    def retrieve_from_text(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector_np = self.query_encoder.encode(query, convert_to_numpy=True)

        # print(f"SEARCH TYPE: {self.search_type}\n  QUERY: {query} \n QUERY_VECTOR: {query_vector_np}")

        if self.search_type == "text":
            return self.vector_store.query(query_string=query, top_k=top_k)
        elif self.search_type == "vector":
            return self.vector_store.query(query_vector=query_vector_np.tolist(), query_string="", top_k=top_k)
        elif self.search_type == "hybrid":
            return self.vector_store.query(
                query_string=query,
                query_vector=query_vector_np.tolist(),
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