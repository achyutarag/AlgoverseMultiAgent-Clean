# AlgoverseMultiAgent

# MA-RAG: Multi-Agent Retrieval-Augmented Generation Pipeline

## Purpose
MA-RAG is an advanced multi-agent system designed to answer complex questions through intelligent document retrieval and reasoning. The pipeline decomposes queries into manageable sub-tasks, retrieves relevant information from a knowledge base using semantic search, and synthesizes comprehensive answers through a coordinated team of specialized agents. By leveraging Google Gemini for reasoning tasks and local embedding models for efficient retrieval, MA-RAG achieves high accuracy on both simple factual questions and complex multi-hop reasoning tasks, making it ideal for research, question-answering systems, and knowledge-intensive applications.

## Features

- **Multi-Agent Architecture**: Planner, QA, Extractor, Step Definer, Retriever, and Final Assembler
- **Google Gemini Integration**: Powered by `gemini-2.5-pro-preview-03-25`
- **FAISS Vector Store**: Fast, scalable similarity search using FAISS with SentenceTransformer embeddings
- **Local Embeddings**: Support for local embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2) with GPU/CPU support
- **Robust JSON Parsing**: Handles malformed LLM responses with automatic cleanup and retry mechanisms
- **State Management**: Comprehensive state tracking with reasoning trajectory and execution history
- **Evaluation Framework**: Built-in support for TriviaQA and HotpotQA datasets with EM and F1 metrics
- **Mixed Model Support**: Optimized orchestrator for different model types (SLMs for retrieval, LLMs for reasoning)
- **Tokenization Utilities**: Centralized text preprocessing and tokenization for consistent performance
- **Error Handling**: Automatic retry mechanisms with exponential backoff
- **MCP Support**: Model Context Protocol integration for enhanced reasoning state management

- ## Architecture

### Agent Components

#### Planner Agent
**Purpose**: Performs query disambiguation and task decomposition. Analyzes input queries to identify ambiguities and creates structured reasoning plans with chain-of-thought prompting. Breaks down complex questions into manageable, sequential steps with clear dependencies and objectives. Classifies queries as simple, multi-hop, comparative, or analytical to determine the appropriate reasoning strategy.

**Key Responsibilities**:
- Query disambiguation and clarification
- Task decomposition into executable steps
- Dependency identification between steps
- Critical step marking for error handling

#### Step Definer Agent
**Purpose**: Converts abstract reasoning steps into specific, executable sub-queries tailored for retrieval. Bridges high-level intent from the planner with low-level execution needs. Conditions on the original query, overall plan, current step, and accumulated history to generate precise retrieval queries.

**Key Responsibilities**:
- Context grounding from previous steps
- Subquery generation for precise retrieval
- Priority assignment for sub-queries
- Context type identification (factual, statistical, comparative)

#### Retriever Agent
**Purpose**: Performs fast, scalable semantic search over large document corpora using FAISS vector store. Embeds queries and documents using local SentenceTransformer models to find the most relevant documents based on semantic similarity. Handles document indexing, batch processing, and similarity threshold filtering.

**Key Responsibilities**:
- Document embedding and vector store management
- Semantic similarity search
- Top-k document retrieval with similarity filtering
- Batch processing for efficient embedding generation

#### Extractor Agent
**Purpose**: Performs fine-grained selection and aggregation of sentences/spans aligned with subqueries. Filters out noise and redundant content to address context inefficiency. Extracts only the most relevant passages from retrieved documents, enabling effective evidence aggregation while avoiding the "lost-in-the-middle" problem.

**Key Responsibilities**:
- Fine-grained passage extraction from documents
- Noise filtering and relevance scoring
- Evidence aggregation from multiple sources
- Context efficiency optimization

#### QA Agent
**Purpose**: Synthesizes answers using in-context learning with step-specific context. Produces responses for each step which are passed to subsequent iterations, enabling grounded reasoning throughout the trajectory. Generates answers with confidence scores, reasoning explanations, and source attribution.

**Key Responsibilities**:
- Answer synthesis from extracted evidence
- Confidence scoring based on evidence quality
- Reasoning explanation generation
- Source attribution and evidence linking

#### Final Assembler
**Purpose**: Assembles the final answer from all step results, providing comprehensive synthesis and quality assessment. Combines answers from multiple steps into a coherent, well-structured response. Evaluates evidence quality, summarizes reasoning trajectories, and produces the final output with complete metadata.

**Key Responsibilities**:
- Multi-step answer synthesis
- Reasoning trajectory summarization
- Evidence quality assessment
- Source consolidation and deduplication

#### State Manager
**Purpose**: Manages the evolving context and state throughout the reasoning trajectory. Handles step dependencies and maintains execution history Hi = {(s1, a1), ..., (si, ai)}. Tracks completed steps, step results, conversation history, and execution metadata to ensure proper state propagation between agents.

**Key Responsibilities**:
- Execution state tracking
- History management (Hi)
- Dependency resolution
- Context accumulation across steps

### Key Files
- `pipeline.py`: Main pipeline configuration and MARAGPipeline class
- `agents/orchestrator.py`: Main orchestrator with state management
- `agents/mixed_model_orchestrator.py`: Optimized orchestrator for mixed model usage
- `agents/llm_wrapper.py`: LLM abstraction layer supporting Google Gemini and HuggingFace models
- `agents/tokenization_utils.py`: JSON parsing and text preprocessing utilities
- `agents/retriever_agent.py`: FAISS-based retrieval with local embeddings
- `agents/state_manager.py`: State tracking and execution history management
- `evaluate_datasets.py`: Evaluation framework for HotpotQA, and 2WikiMultiHop




## Prerequisites

- Python 3.8+
- Google Cloud Account with Gemini API access
- Service Account Key (JSON file) or Gemini API Key
- (Optional) CUDA-capable GPU for faster embeddings and local models

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/achyutarag/AlgoverseMultiAgent-Clean.git
cd AlgoverseMultiAgent-Clean
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: For GPU support with FAISS, install `faiss-gpu` instead of `faiss-cpu`:
```bash
pip install faiss-gpu
```

### 3. Set Up Google Cloud Credentials

#### Option A: Environment Variable (Recommended)
```bash
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-key.json"

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

#### Option B: Place Key in Project
1. Download your service account key from Google Cloud Console
2. Place it in the project root as `service-account-key.json`
3. The system will automatically detect it

#### Option C: API Key (Alternative)
```bash
# Set as environment variable
export GEMINI_API_KEY="your-api-key-here"
```

**Important**: Never commit your service account key or API key to version control. Add `service-account-key.json` to `.gitignore`.

### 4. Verify Setup
```bash
python test_gemini_debug.py
```

## Usage

### Basic Pipeline Execution

#### Using the Mixed Model Orchestrator (Recommended)
```python
from agents.mixed_model_orchestrator import run_optimized_marag_pipeline

# Run the pipeline
result = await run_optimized_marag_pipeline(
    query="What is the capital of France?",
    planning_model="gemini-2.5-pro-preview-03-25",
    step_definition_model="gemini-2.5-pro-preview-03-25", 
    qa_model="gemini-2.5-pro-preview-03-25"
)

print(result.final_answer)
```

#### Using the Full Pipeline Class
```python
from pipeline import MARAGPipeline

# Initialize pipeline with custom configuration
pipeline = MARAGPipeline(
    config={
        "model": {
            "name": "gemini-2.5-pro-preview-03-25",
            "temperature": 0.3,
            "max_tokens": 1024
        },
        "retrieval": {
            "top_k": 5,
            "min_similarity": 0.6
        },
        "embedding": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "device": "cuda"  # or "cpu"
        }
    }
)

# Add documents to the knowledge base
documents = [
    {
        "page_content": "Your document text here...",
        "metadata": {
            "source": "document_source",
            "id": "doc1",
            "title": "Document Title"
        }
    }
]
pipeline.add_documents(documents)

# Query the pipeline
result = await pipeline.query("Your question here?", timeout=30.0)
print(result.final_answer)
print(f"Query Type: {result.query_type}")
print(f"Sources: {len(result.sources)}")
```

### Evaluation on Datasets
```bash
# Evaluate on TriviaQA
python evaluate_datasets.py --dataset triviaqa --num_examples 20

# Evaluate on HotpotQA  
python evaluate_datasets.py --dataset hotpotqa --num_examples 20
```

### Test Individual Agents
```bash
# Test planner agent
python -m agents.test_planner_debug

# Test full pipeline
python -m agents.test_hotpotqa

# Test retrieval only
python -m agents.test_retrieval_only
```

## Architecture

### Agent Components
- **Planner Agent**: Breaks down complex queries into sub-tasks with query disambiguation
- **QA Agent**: Synthesizes answers from retrieved information with confidence scoring
- **Extractor Agent**: Extracts relevant information from documents with relevance scoring
- **Step Definer**: Defines execution steps for complex queries with dependency management
- **Retriever Agent**: Retrieves relevant documents using FAISS vector store with similarity search
- **Final Assembler**: Combines results into final answer with reasoning synthesis
- **State Manager**: Tracks execution state, reasoning trajectory, and conversation history

### Key Files
- `pipeline.py`: Main pipeline configuration and MARAGPipeline class
- `agents/orchestrator.py`: Main orchestrator with state management
- `agents/mixed_model_orchestrator.py`: Optimized orchestrator for mixed model usage
- `agents/llm_wrapper.py`: LLM abstraction layer supporting Google Gemini and HuggingFace models
- `agents/tokenization_utils.py`: JSON parsing and text preprocessing utilities
- `agents/retriever_agent.py`: FAISS-based retrieval with local embeddings
- `agents/state_manager.py`: State tracking and execution history management
- `evaluate_datasets.py`: Evaluation framework for TriviaQA and HotpotQA

### Vector Store & Embeddings
- **Vector Store**: FAISS for fast similarity search
- **Embedding Models**: 
  - Default: `all-MiniLM-L6-v2` (fast, lightweight)
  - Alternative: `sentence-transformers/all-mpnet-base-v2` (higher quality)
- **Device Support**: Automatic GPU/CPU detection, supports CUDA and MPS (Apple Silicon)

## Configuration

### Model Configuration
Update model names and settings in `pipeline.py` or pass as config:
```python
DEFAULT_CONFIG = {
    "model": {
        "name": "gemini-2.5-pro-preview-03-25",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 50
    },
    "embedding": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "device": "cuda",
        "batch_size": 32
    },
    "retrieval": {
        "top_k": 5,
        "min_similarity": 0.6,
        "max_documents": 10
    },
    "extraction": {
        "max_documents": 3,
        "min_relevance": 0.5,
        "max_tokens": 1000
    },
    "qa": {
        "min_confidence": 0.5,
        "max_followup_questions": 3
    },
    "max_steps": 5,
    "max_subqueries": 3
}
```

### Environment Variables
```bash
# Required (choose one)
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
# OR
GEMINI_API_KEY="your-api-key"

# Optional
CUDA_VISIBLE_DEVICES=0  # Specify GPU device
```

### Customizing Embedding Models
```python
from agents.retriever_agent import RetrieverAgent

retriever = RetrieverAgent(
    model_name="sentence-transformers/all-mpnet-base-v2",  # or "all-MiniLM-L6-v2"
    device="cuda",  # or "cpu", "mps"
    top_k=5,
    min_similarity=0.6
)
```

## Evaluation

The pipeline includes comprehensive evaluation on:
- **TriviaQA**: Factual question answering
- **HotpotQA**: Multi-hop reasoning

Results are saved to CSV files with metrics:
- **Exact Match (EM) Score**: Binary match between prediction and ground truth
- **F1 Score**: Token-level overlap between prediction and ground truth
- **Response Time**: Latency per query
- **Token Usage**: Total tokens consumed (if tracked)

### Running Evaluation
```bash
# Evaluate with custom parameters
python evaluate_datasets.py --dataset hotpotqa --num_examples 50

# Results are saved to CSV files with timestamps
# Example: hotpotqa_results_20251106_144430.csv
```

## Troubleshooting

### Common Issues

1. **"404 models/gemini-1.5-pro is not found"**
   - Update model name to `gemini-2.5-pro-preview-03-25` in your configuration

2. **"GOOGLE_APPLICATION_CREDENTIALS not set"**
   - Set the environment variable or place key file in project root
   - Ensure the path is correct and the file is readable

3. **"Permission denied" errors with service account key**
   - Check file permissions: `chmod 600 service-account-key.json` (Linux/Mac)
   - Ensure the service account has Gemini API access enabled

4. **JSON parsing errors**
   - The system automatically handles malformed JSON responses
   - Check logs for specific parsing issues
   - The `tokenization_utils.py` module includes robust JSON cleanup

5. **Memory issues**
   - Reduce `num_examples` in evaluation scripts
   - Use smaller embedding models (e.g., `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`)
   - Reduce `batch_size` in embedding configuration
   - Use `.cursorignore` to exclude heavy files

6. **FAISS import errors**
   - Ensure you have `faiss-cpu` or `faiss-gpu` installed
   - For GPU support: `pip install faiss-gpu`
   - For CPU only: `pip install faiss-cpu`

7. **CUDA/GPU issues**
   - Check CUDA installation: `nvidia-smi`
   - Set `device="cpu"` in config if GPU is unavailable
   - For Apple Silicon, use `device="mps"` (if supported)

8. **Slow retrieval performance**
   - Use GPU for embeddings: `device="cuda"` in embedding config
   - Reduce `top_k` value for faster retrieval
   - Use smaller embedding models for faster inference

### Performance Tips
- Use the mixed model orchestrator for better performance (SLMs for retrieval, LLMs for reasoning)
- Adjust batch sizes for your hardware (larger batches = faster but more memory)
- Monitor API usage in Google Cloud Console to avoid quota limits
- Cache embeddings when processing the same documents multiple times
- Use `max_steps` and `max_subqueries` to limit pipeline depth for faster responses

### Debugging
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check reasoning trajectory:
```python
result = await pipeline.query("Your question")
for entry in result.reasoning_trajectory:
    print(f"{entry['action']} - {entry['step_id']}")
```

## Results

The pipeline achieves strong performance on:
- **TriviaQA**: High accuracy on factual questions
- **HotpotQA**: Effective multi-hop reasoning with step-by-step decomposition

Evaluation results are automatically saved with timestamps for tracking improvements over time.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for LLM capabilities
- Hugging Face datasets and transformers for evaluation and embeddings
- FAISS by Facebook Research for efficient similarity search
- SentenceTransformers for embedding models
- The open-source community for inspiration

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review the agent documentation in the `agents/` directory
- Check evaluation results in CSV files for performance insights

---

**Happy coding!** 🚀
