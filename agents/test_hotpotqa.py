"""
Multi-dataset test setup for MA-RAG pipeline.
Tests on both HotpotQA and 2WikiMultiQA.
"""



import asyncio
import json
from datasets import load_dataset
from agents.mixed_model_orchestrator import run_optimized_marag_pipeline


async def test_multi_dataset():
    """Test on both HotpotQA and 2WikiMultiQA."""
    
    # Load both datasets
    print("Loading datasets...")
    hotpotqa = load_dataset("hotpot_qa", "distractor")
    wikimqa = load_dataset("framolfese/2WikiMultihopQA")
    
    # Test subsets
    hotpotqa_test = list(hotpotqa["train"].select(range(10)))  # 10 examples
    wikimqa_test = list(wikimqa["train"].select(range(10)))   # 10 examples
    
    datasets = [
        ("HotpotQA", hotpotqa_test),
        ("2WikiMultiQA", wikimqa_test)
    ]
    
    all_results = {}
    
    for dataset_name, test_subset in datasets:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_name}")
        print(f"{'='*50}")
        
        results = []
        for i, example in enumerate(test_subset):
            print(f"\nExample {i+1}/3:")
            print(f"Question: {example['question']}")
            
            try:
                result = await run_optimized_marag_pipeline(
                    query=example["question"]
                )
                
                results.append({
                    "question": example["question"],
                    "predicted": result.final_answer,
                    "ground_truth": example["answer"],
                    "confidence": result.confidence
                })
                
                print(f"✅ Success")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
        
        all_results[dataset_name] = results
        print(f"\n{dataset_name} Results: {len(results)}/3 successful")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for dataset_name, results in all_results.items():
        print(f"{dataset_name}: {len(results)}/3 successful")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(test_multi_dataset())