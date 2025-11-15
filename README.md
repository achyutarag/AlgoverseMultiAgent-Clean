# AlgoverseMultiAgent


# MA-RAG: Multi-Agent Retrieval-Augmented Generation Pipeline


## Features

- **Multi-Agent Architecture**: Planner, QA, Extractor, Step Definer, Retriever, and Final Assembler
- **Google Gemini Integration**: Powered by `gemini-2.5-pro-preview-03-25`
- **Robust JSON Parsing**: Handles malformed LLM responses
- **Evaluation Framework**: Built-in support for TriviaQA and HotpotQA datasets
- **Mixed Model Support**: Optimized orchestrator for different model types

## Prerequisites

- Python 3.8+
- Google Cloud Account with Gemini API access
- Service Account Key (JSON file)

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

### 3. Set Up Google Cloud Credentials

#### Option A: Environment Variable (Recommended)
```bash
# Set the path to your service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

#### Option B: Place Key in Project
1. Download your service account key from Google Cloud Console
2. Place it in the project root as `service-account-key.json`
3. The system will automatically detect it

### 4. Verify Setup
```bash
python test_gemini_debug.py
```

## Usage

### Basic Pipeline Execution
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
```

## Architecture

### Agent Components
- **Planner Agent**: Breaks down complex queries into sub-tasks
- **QA Agent**: Synthesizes answers from retrieved information
- **Extractor Agent**: Extracts relevant information from documents
- **Step Definer**: Defines execution steps for complex queries
- **Retriever Agent**: Retrieves relevant documents
- **Final Assembler**: Combines results into final answer

### Key Files
- `pipeline.py`: Main pipeline configuration
- `agents/orchestrator.py`: Main orchestrator
- `agents/mixed_model_orchestrator.py`: Optimized orchestrator
- `agents/llm_wrapper.py`: LLM abstraction layer
- `agents/tokenization_utils.py`: JSON parsing utilities

## Configuration

### Model Configuration
Update model names in `pipeline.py`:
```python
MODEL_CONFIG = {
    "model_name": "gemini-2.5-pro-preview-03-25",
    "model_type": "google_gemini"
}
```

### Environment Variables
```bash
# Required
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Optional
GEMINI_API_KEY="your-api-key"  # Alternative to service account
```

## Evaluation

The pipeline includes comprehensive evaluation on:
- **TriviaQA**: Factual question answering
- **HotpotQA**: Multi-hop reasoning

Results are saved to CSV files with metrics:
- Exact Match (EM) Score
- F1 Score
- Response Time

## 🐛 Troubleshooting

### Common Issues

1. **"404 models/gemini-1.5-pro is not found"**
   - Update model name to `gemini-2.5-pro-preview-03-25`

2. **"GOOGLE_APPLICATION_CREDENTIALS not set"**
   - Set the environment variable or place key file in project root

3. **JSON parsing errors**
   - The system automatically handles malformed JSON responses

4. **Memory issues**
   - Reduce `num_examples` in evaluation scripts
   - Use `.cursorignore` to exclude heavy files

### Performance Tips
- Use the mixed model orchestrator for better performance
- Adjust batch sizes for your hardware
- Monitor API usage in Google Cloud Console

## Results

The pipeline achieves strong performance on:
- **TriviaQA**: High accuracy on factual questions
- **HotpotQA**: Effective multi-hop reasoning


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for LLM capabilities
- Hugging Face datasets for evaluation
- The open-source community for inspiration

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the agent documentation

---

**Happy coding!** 
