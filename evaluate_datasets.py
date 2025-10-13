import asyncio
import csv
import json
import re
import string
from datasets import load_dataset
from typing import Dict, Any, List
from agents.mixed_model_orchestrator import run_optimized_marag_pipeline

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
    dataset = load_dataset(dataset_name_full, dataset_config)
    eval_dataset = dataset["validation"].select(range(num_examples))
    
    results = []
    total_em = 0.0
    total_f1 = 0.0
    
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
        else:
            ground_truth = example.get("answer", "")
        
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        try:
            # Run through MA-RAG pipeline
            print("Running through MA-RAG pipeline...")
            result = await run_optimized_marag_pipeline(question)
            
            # Extract final answer from pipeline result
            if hasattr(result, 'content'):
                # Try to extract answer from JSON response
                try:
                    response_data = json.loads(result.content)
                    prediction = response_data.get("answer", result.content)
                except json.JSONDecodeError:
                    # If not JSON, use the raw content
                    prediction = result.content
            else:
                prediction = str(result)
            
            print(f"Prediction: {prediction}")
            
            # Calculate metrics
            em_score = exact_match_score(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)
            
            total_em += em_score
            total_f1 += f1
            
            results.append({
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "exact_match": em_score,
                "f1_score": f1
            })
            
            print(f"Exact Match: {em_score:.3f}, F1: {f1:.3f}")
            
        except Exception as e:
            print(f"Error processing example: {e}")
            results.append({
                "question": question,
                "prediction": "ERROR",
                "ground_truth": ground_truth,
                "exact_match": 0.0,
                "f1_score": 0.0
            })
    
    # Calculate overall metrics
    avg_em = total_em / len(results)
    avg_f1 = total_f1 / len(results)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Results:")
    print(f"Exact Match: {avg_em:.4f}")
    print(f"F1 Score:    {avg_f1:.4f}")
    print(f"Examples:    {len(results)}")
    print(f"{'='*60}")
    
    # Save results to CSV
    csv_filename = f"{dataset_name.lower()}_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Prediction", "Ground_Truth", "ExactMatch", "F1Score"])
        
        for result in results:
            writer.writerow([
                result["question"],
                result["prediction"],
                result["ground_truth"],
                result["exact_match"],
                result["f1_score"]
            ])
    
    print(f"Results saved to {csv_filename}")
    
    return results, avg_em, avg_f1

async def main():
    """Run evaluation on both datasets."""
    print("🚀 Starting MA-RAG Pipeline Evaluation")
    print("=" * 60)
        
    # Evaluate TriviaQA
    trivia_results, trivia_em, trivia_f1 = await evaluate_dataset(
        "TriviaQA", 
        "mandarjoshi/trivia_qa", 
        "rc",  # Add this configuration
        num_examples=5
    )

    # Evaluate HotpotQA
    hotpot_results, hotpot_em, hotpot_f1 = await evaluate_dataset(
        "HotpotQA", 
        "hotpot_qa", 
        "distractor",  # Add this configuration
        num_examples=5
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"TriviaQA - EM: {trivia_em:.4f}, F1: {trivia_f1:.4f}")
    print(f"HotpotQA - EM: {hotpot_em:.4f}, F1: {hotpot_f1:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())

