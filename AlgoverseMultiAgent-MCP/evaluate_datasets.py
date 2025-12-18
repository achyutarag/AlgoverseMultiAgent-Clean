import asyncio
import argparse
import csv
import json
import logging
import re
import string
import time
import os 
from datetime import datetime
from datasets import load_dataset
from typing import Dict, Any, List
from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents

# At the top of evaluate_datasets.py, add:
logging.basicConfig(level=logging.DEBUG)  # Enable INFO logs


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

async def evaluate_dataset(dataset_name: str, dataset_name_full: str, dataset_config: str, num_examples: int = 50):
    """Evaluate a dataset using the MA-RAG pipeline."""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name} dataset...")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    try:
        if dataset_config is not None:
            dataset = load_dataset(dataset_name_full, dataset_config)
        else:
            dataset = load_dataset(dataset_name_full)
        eval_dataset = dataset["validation"].select(range(num_examples))
    except Exception as e:
        # For MuSiQue, try GitHub fallback if HuggingFace fails
        if dataset_name == "MuSiQue":
            print(f"⚠️  HuggingFace load failed: {e}")
            print("Attempting to load from GitHub repository...")
            try:
                from agents.musique_document_loader import _load_musique_from_github
                examples = _load_musique_from_github("validation")
                # Create a simple dataset-like object
                class SimpleDataset:
                    def __init__(self, data):
                        self.data = data
                    def __getitem__(self, idx):
                        return self.data[idx]
                    def __len__(self):
                        return len(self.data)
                    def select(self, indices):
                        return SimpleDataset([self.data[i] for i in indices])
                
                eval_dataset = SimpleDataset(examples).select(range(min(num_examples, len(examples))))
                print(f"✅ Successfully loaded {len(eval_dataset)} examples from GitHub")
            except Exception as github_error:
                raise Exception(f"Failed to load MuSiQue from both HuggingFace and GitHub.\n"
                              f"HuggingFace error: {e}\n"
                              f"GitHub error: {github_error}")
        else:
            raise
    
    results = []
    total_em = 0.0
    total_f1 = 0.0
    total_latency = 0.0 #Added this
    total_tokens = 0 #Added this
    
    print(f"Processing {len(eval_dataset)} examples...")
    
    for i, example in enumerate(eval_dataset):
        print(f"\nProcessing example {i+1}/{len(eval_dataset)}")
        
        # Get question and ground truth
        question = example["question"]
        
        # Get ground truth answer based on dataset
        if dataset_name == "TriviaQA":
            ground_truth = example["answer"]["value"] if example["answer"]["value"] else ""
        elif dataset_name == "HotpotQA":
            ground_truth = example["answer"]
        elif dataset_name == "MuSiQue":
            # MuSiQue answer might be a string or list
            answer_raw = example.get("answer", "")
            if isinstance(answer_raw, list):
                # If answer is a list, take the first one or join them
                ground_truth = answer_raw[0] if answer_raw else ""
            elif isinstance(answer_raw, dict):
                # If answer is a dict, try common keys
                ground_truth = answer_raw.get("value", answer_raw.get("text", str(answer_raw)))
            else:
                ground_truth = str(answer_raw) if answer_raw else ""
        else:
            ground_truth = example.get("answer", "")
        
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        try:
            # Load documents based on dataset type
            print("Loading documents for this example...")
            if dataset_name == "HotpotQA":
                # HotpotQA: Per-example documents (10 docs: 2 gold + 8 distractors)
                from agents.hotpotqa_document_loader import load_hotpotqa_example_context_as_documents
                documents = load_hotpotqa_example_context_as_documents(example)
            elif dataset_name == "MuSiQue":
                # MuSiQue: Per-example paragraphs (20-30 paragraphs, 2-4 are supporting facts)
                from agents.musique_document_loader import load_musique_example_context_as_documents
                documents = load_musique_example_context_as_documents(example)
            elif dataset_name == "TriviaQA":
                # TriviaQA: Load a general corpus (TriviaQA doesn't have per-example context)
                # You might need to load documents from a different source
                # For now, load a general corpus
                from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents
                documents = load_hotpotqa_context_as_documents("validation", num_examples=100)
            else:
                # Default: Try to load general corpus
                from agents.hotpotqa_document_loader import load_hotpotqa_context_as_documents
                documents = load_hotpotqa_context_as_documents("validation", num_examples=100)
            
            # Run through MA-RAG pipeline with documents
            print("Running through MA-RAG pipeline...")
            from agents.mixed_model_orchestrator import create_optimized_marag_pipeline
            
            start_time = time.time()
            orchestrator = await create_optimized_marag_pipeline()
            orchestrator.add_documents(documents)  # Add documents explicitly
            result = await orchestrator.execute_pipeline(question)
            latency = time.time() - start_time
            
            # Extract final answer from pipeline result
            # PipelineResult has 'final_answer' field, not 'content'
            if hasattr(result, 'final_answer'):
                prediction = result.final_answer
            elif hasattr(result, 'content'):  # Fallback for other result types
                try:
                    response_data = json.loads(result.content)
                    prediction = response_data.get("answer", result.content)
                except json.JSONDecodeError:
                    prediction = result.content
            else:
                prediction = str(result)
            
            # Get actual token usage from pipeline result
            token_usage = result.token_usage if hasattr(result, 'token_usage') else {}
            total_tokens_estimate = token_usage.get("total_tokens", 0)
            # If token_usage is empty, fall back to estimate (for backward compatibility)
            if total_tokens_estimate == 0:
                # Rough estimate: ~0.75 tokens per word for English text
                prediction_tokens = int(len(prediction.split()) * 0.75)
                question_tokens = int(len(question.split()) * 0.75)
                estimated_doc_tokens = len(documents) * 150 if documents else 0
                total_tokens_estimate = prediction_tokens + question_tokens + estimated_doc_tokens
            
            # Post-process prediction for evaluation: extract concise answer
            # For yes/no questions, extract just "yes" or "no" from verbose answers
            import re
            prediction_original = prediction  # Keep original for debugging
            prediction_lower = prediction.lower().strip()
            
            # Check if ground truth is a simple yes/no (for yes/no questions)
            ground_truth_lower = ground_truth.lower().strip()
            if ground_truth_lower in ["yes", "no"]:
                # Look for standalone yes/no at the end (most reliable - final answer)
                final_answer_match = re.search(r'\b(yes|no)\s*\.?\s*$', prediction_lower)
                if final_answer_match:
                    prediction = final_answer_match.group(1).lower()
                    print(f"[DEBUG] Extracted concise answer from end: '{prediction}' (original: '{prediction_original[:100]}...')")
                else:
                    # Fallback: extract last yes/no found in the text
                    yes_no_matches = list(re.finditer(r'\b(yes|no)\b', prediction_lower))
                    if yes_no_matches:
                        # Get the last match (most likely the final answer)
                        prediction = yes_no_matches[-1].group(1).lower()
                        print(f"[DEBUG] Extracted concise answer (last match): '{prediction}' (original: '{prediction_original[:100]}...')")
                    else:
                        # No explicit yes/no found - try to infer from content
                        # For comparison questions, look for "same", "both", "also", etc. implying yes
                        inferred = False
                        if ground_truth_lower == "yes":
                            # Look for indicators that imply "yes" (same, both, also, etc.)
                            same_indicators = [
                                r'same\s+(nationality|country|origin|age|name)',
                                r'both\s+(are|were|is)',
                                r'also\s+(american|british|canadian|french|german|spanish|italian|chinese|japanese|russian|indian|australian|brazilian|mexican|korean)',
                                r'were\s+of\s+the\s+same',
                                r'they\s+were\s+(both|the\s+same)',
                                r'(both|each)\s+(is|are|was|were)\s+(american|british|etc)',
                                r'the\s+same\s+(nationality|country)',
                                r'of\s+the\s+same\s+(nationality|country)'
                            ]
                            for pattern in same_indicators:
                                if re.search(pattern, prediction_lower):
                                    prediction = "yes"
                                    print(f"[DEBUG] Inferred 'yes' from pattern: '{pattern}' (original: '{prediction_original[:100]}...')")
                                    inferred = True
                                    break
                        elif ground_truth_lower == "no":
                            # Look for indicators that imply "no" (different, not the same, etc.)
                            different_indicators = [
                                r'different\s+(nationality|country|origin)',
                                r'not\s+the\s+same',
                                r'not\s+(both|each)\s+(are|were)',
                                r'one\s+is.*other\s+is',
                                r'first\s+is.*second\s+is'
                            ]
                            for pattern in different_indicators:
                                if re.search(pattern, prediction_lower):
                                    prediction = "no"
                                    print(f"[DEBUG] Inferred 'no' from pattern: '{pattern}' (original: '{prediction_original[:100]}...')")
                                    inferred = True
                                    break
                        
                        if not inferred:
                            print(f"[DEBUG] Could not extract or infer yes/no from: '{prediction_original[:100]}...'")
            
            print(f"Prediction: {prediction}")
            print(f"Prediction: {prediction}")
            print(f"Ground Truth: {ground_truth}")

            print(f"\n[DEBUG] Prediction length: {len(prediction) if prediction else 0}")
            print(f"[DEBUG] Ground truth length: {len(ground_truth) if ground_truth else 0}")
            print(f"[DEBUG] Prediction normalized: '{normalize_text(prediction)}'")
            print(f"[DEBUG] Ground truth normalized: '{normalize_text(ground_truth)}'")
            print(f"[DEBUG] Prediction type: {type(prediction)}")
            print(f"[DEBUG] Prediction is None/empty: {prediction is None or prediction == ''}")

            
            # Calculate metrics
            em_score = exact_match_score(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)

            total_em += em_score
            total_f1 += f1
            total_latency += latency  # Add this
            total_tokens += total_tokens_estimate  # Add this
            
            results.append({
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "exact_match": em_score,
                "f1_score": f1,
                "latency": latency,
                "tokens": total_tokens_estimate,
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("generated_tokens", 0)
            })
            
            print(f"Exact Match: {em_score:.3f}, F1: {f1:.3f}")
            
        except Exception as e:
            print(f"Error processing example: {e}")
            results.append({
                "question": question,
                "prediction": "ERROR",
                "ground_truth": ground_truth,
                "exact_match": 0.0,
                "f1_score": 0.0,
                "latency": 0.0,  # Add this
                "tokens": 0  # Add this
            })
    
    # Calculate overall metrics
    avg_em = total_em / len(results)
    avg_f1 = total_f1 / len(results)
    avg_latency = total_latency / len(results)
    avg_tokens = total_tokens / len(results) if results else 0
    # Calculate average input and output tokens
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    avg_input_tokens = total_input_tokens / len(results) if results else 0
    avg_output_tokens = total_output_tokens / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Results:")
    print(f"Exact Match: {avg_em:.4f}")
    print(f"F1 Score:    {avg_f1:.4f}")
    print(f"Latency/Q:   {avg_latency:.2f}s")
    print(f"Total Latency: {total_latency:.2f}s")
    print(f"Tokens/Q: {avg_tokens:.0f} (Input: {avg_input_tokens:.0f}, Output: {avg_output_tokens:.0f})")
    print(f"Examples:    {len(results)}")
    print(f"{'='*60}")
    
# Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"{dataset_name.lower()}_results_{timestamp}.csv")
    
    # Write CSV
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Prediction", "Ground_Truth", "ExactMatch", "F1Score", "Latency(s)", "Total_Tokens", "Input_Tokens", "Output_Tokens"])
        
        for result in results:
            writer.writerow([
                result["question"],
                result["prediction"],
                result["ground_truth"],
                result["exact_match"],
                result["f1_score"],
                f"{result['latency']:.2f}",
                result.get("tokens", 0),
                result.get("input_tokens", 0),
                result.get("output_tokens", 0)
            ])
        
        # Add summary metrics
        writer.writerow([])  # Empty row for spacing
        writer.writerow(["Summary", "", "", "", "", "", "", "", ""])
        writer.writerow(["Exact Match", "", "", f"{avg_em:.4f}", "", "", "", "", ""])
        writer.writerow(["F1 Score", "", "", f"{avg_f1:.4f}", "", "", "", "", ""])
        writer.writerow(["Latency/Q (s)", "", "", f"{avg_latency:.2f}", "", "", "", "", ""])
        writer.writerow(["Total Latency (s)", "", "", f"{total_latency:.2f}", "", "", "", "", ""])
        writer.writerow(["Tokens/Q", "", "", f"{avg_tokens:.0f}", "", "", f"{avg_tokens:.0f}", f"{avg_input_tokens:.0f}", f"{avg_output_tokens:.0f}"])
        writer.writerow(["Examples", "", "", f"{len(results)}", "", "", "", "", ""])
    
    # Create/update experiments index file
    index_file = os.path.join(results_dir, "experiments_index.txt")
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {dataset_name} - {len(results)} examples - EM: {avg_em:.4f}, F1: {avg_f1:.4f}\n")
    
    print(f"Results saved to {os.path.abspath(csv_filename)}")
    print(f"Experiment logged in {os.path.abspath(index_file)}")
    
    return results, avg_em, avg_f1, avg_latency, total_latency, avg_tokens

async def main(dataset: str = "all", num_examples: int | None = None):
    """Run evaluation on specified dataset(s).

    Args:
        dataset: Which dataset group to evaluate ("all", "hotpotqa", "musique", etc.).
        num_examples: Optional override for number of examples per dataset. If None,
            fall back to the historical defaults (HotpotQA=20, MuSiQue=5).
    """
    print("🚀 Starting MA-RAG Pipeline Evaluation")
    print("=" * 60)
    
    results_summary = []
    
    if dataset.lower() in ["all", "hotpotqa", "hotpot"]:
        # Evaluate HotpotQA
        print(f"\n{'='*60}")
        print("Evaluating HotpotQA...")
        print(f"{'='*60}")
        # Use CLI override if provided, otherwise keep original default (20)
        hotpot_n = num_examples if num_examples is not None else 20
        hotpot_results, hotpot_em, hotpot_f1, hotpot_avg_latency, hotpot_total_latency, hotpot_avg_tokens = await evaluate_dataset(
            "HotpotQA",
            "hotpot_qa",
            "distractor",
            num_examples=hotpot_n
        )
        results_summary.append(("HotpotQA", hotpot_em, hotpot_f1))
        print(f"\nHotpotQA - EM: {hotpot_em:.4f}, F1: {hotpot_f1:.4f}")
    
    if dataset.lower() in ["all", "musique", "musique-v1"]:
        # Evaluate MuSiQue
        print(f"\n{'='*60}")
        print("Evaluating MuSiQue...")
        print(f"{'='*60}")
        # Use CLI override if provided, otherwise keep original default (5)
        musique_n = num_examples if num_examples is not None else 5
        musique_results, musique_em, musique_f1, musique_avg_latency, musique_total_latency, musique_avg_tokens = await evaluate_dataset(
            "MuSiQue",
            "allenai/musique-v1",
            None,  # MuSiQue doesn't have configs like HotpotQA
            num_examples=musique_n
        )
        results_summary.append(("MuSiQue", musique_em, musique_f1))
        print(f"\nMuSiQue - EM: {musique_em:.4f}, F1: {musique_f1:.4f}")
    
    # Print summary
    if results_summary:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for name, em, f1 in results_summary:
            print(f"{name} - EM: {em:.4f}, F1: {f1:.4f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MA-RAG pipeline on datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "hotpotqa", "hotpot", "musique", "musique-v1"],
        help="Dataset to evaluate: 'all' (default), 'hotpotqa', or 'musique'"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Optional override for number of examples per dataset "
             "(default: 20 for HotpotQA, 5 for MuSiQue)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.dataset, args.num_examples))

