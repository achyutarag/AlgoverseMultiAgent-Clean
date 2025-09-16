from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from .base_agent import BaseAgent, AgentResponse
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """Wrapper for local embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.model.eval()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []
        # Convert to numpy array and then to list for compatibility
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Alias for embed_documents for compatibility."""
        if isinstance(texts, str):
            return [self.embed_query(texts)]
        return self.embed_documents(texts)

class RetrieverAgent(BaseAgent):
    """
    The Retriever Agent is responsible for retrieving relevant documents
    based on the provided sub-queries using local embedding models.
    """
    
    def __init__(
        self, 
        documents: List[Document] = None, 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize the Retriever Agent.
        
        Args:
            documents: List of Document objects to create the vector store
            model_config: Configuration for the embedding model
            model_name: Name of the local embedding model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu')
        """
        super().__init__("retriever_agent", model_config, model_name)
        
        # Initialize local embeddings
        self.embeddings = LocalEmbeddings(
            model_name=model_name,
            device=device or ("cuda" if model_config.get("use_cuda", False) else None)
        )
        self.vector_store = None
        
        if documents:
            self._create_vector_store(documents)
    
    def _create_vector_store(self, documents: List[Document]):
        """Create or update the vector store with the given documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store")
            
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store."""
        if not self.vector_store:
            self._create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process the input sub-query and retrieve relevant documents using local embeddings.
        
        Args:
            input_data: Dictionary containing:
                - 'query': The sub-query to retrieve documents for
                - 'k': Number of documents to retrieve (default: 3)
                - 'filter': Filter criteria for the documents
                - 'score_threshold': Minimum similarity score threshold (0-1)
                - 'include_scores': Whether to include similarity scores (default: True)
                
        Returns:
            AgentResponse containing the retrieved documents and metadata
        """
        if not self.vector_store:
            return AgentResponse(
                content="Error: No vector store initialized",
                metadata={"error": "Vector store not initialized"}
            )
            
        query = input_data.get('query', '').strip()
        if not query:
            return AgentResponse(
                content="Error: No query provided",
                metadata={"error": "No query provided"}
            )
            
        try:
            # Get retrieval parameters
            k = min(int(input_data.get('k', 3)), 20)  # Cap at 20 for performance
            filter_criteria = input_data.get('filter', {})
            score_threshold = float(input_data.get('score_threshold', 0.0))
            include_scores = bool(input_data.get('include_scores', True))
            
            # Log the retrieval request
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform similarity search with scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_criteria
            )
            
            # Process results
            results = []
            similarities = []
            
            for doc, score in docs_and_scores:
                # Convert score to similarity (higher is better)
                similarity = 1.0 / (1.0 + score)
                
                # Skip if below threshold
                if similarity < score_threshold:
                    continue
                    
                # Add to results
                doc_data = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(similarity) if include_scores else None
                }
                results.append(doc_data)
                similarities.append(similarity)
            
            # Log retrieval results
            logger.info(f"Retrieved {len(results)} documents with average similarity: "
                       f"{np.mean(similarities) if similarities else 0:.4f}")
            
            # Update history
            self._update_history("user", f"Retrieve documents for: {query}")
            self._update_history(
                "assistant", 
                f"Retrieved {len(results)} documents "
                f"(avg similarity: {np.mean(similarities) if similarities else 0:.2f})"
            )
            
            # Prepare response
            response_data = {
                "query": query,
                "documents": [
                    {k: v for k, v in doc.items() if k != 'score' or include_scores}
                    for doc in results
                ]
            }
            
            if include_scores and similarities:
                response_data["scores"] = [float(s) for s in similarities]
            
            return AgentResponse(
                content=json.dumps(response_data, ensure_ascii=False),
                metadata={
                    "query": query,
                    "num_documents": len(results),
                    "average_score": float(np.mean(similarities)) if similarities else 0.0,
                    "min_score": float(min(similarities)) if similarities else 0.0,
                    "max_score": float(max(similarities)) if similarities else 0.0,
                    "retrieval_parameters": {
                        "k": k,
                        "filter": filter_criteria,
                        "score_threshold": score_threshold,
                        "model": self.embeddings.model_name
                    }
                }
            )
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(
                content=error_msg,
                metadata={
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }
            )
    
    def save_index(self, path: Union[str, Path]):
        """
        Save the vector store to disk.
        
        Args:
            path: Directory path to save the index
        """
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))
        
        # Save model config
        config = {
            "model_name": self.embeddings.model_name,
            "class_name": self.__class__.__name__
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load_index(
        cls, 
        path: Union[str, Path], 
        model_config: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> 'RetrieverAgent':
        """
        Load a vector store from disk.
        
        Args:
            path: Directory path containing the saved index
            model_config: Configuration for the embedding model
            model_name: Override the model name from saved config
            
        Returns:
            An instance of RetrieverAgent with the loaded index
        """
        path = Path(path)
        
        # Load config if exists
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            model_name = model_name or config.get("model_name", "all-MiniLM-L6-v2")
        else:
            model_name = model_name or "all-MiniLM-L6-v2"
        
        # Initialize the agent
        instance = cls(
            documents=None,
            model_config=model_config,
            model_name=model_name,
            device=model_config.get("device") if model_config else None
        )
        
        # Load the vector store
        instance.vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=instance.embeddings
        )
        
        return instance
