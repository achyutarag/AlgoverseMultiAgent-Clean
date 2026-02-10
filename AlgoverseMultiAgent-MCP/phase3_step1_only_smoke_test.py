"""
Phase 3 Step-1 Only Smoke Test

Runs only Step 1 (entity identification) for a single MuSiQue example
to isolate entity extraction and type validation.
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from agents.mixed_model_orchestrator import create_optimized_marag_pipeline
from agents.musique_document_loader import (
    load_musique_example_context_as_documents,
    _load_musique_from_github,
)


def load_single_musique_example(split: str, example_index: int):
    try:
        dataset = load_dataset("allenai/musique-v1")
        split_data = dataset[split]
        return split_data[example_index]
    except Exception:
        examples = _load_musique_from_github(split)
        return examples[example_index]


async def run_step1_only(split: str, example_index: int):
    example = load_single_musique_example(split, example_index)
    question = example.get("question", "").strip()
    if not question:
        raise ValueError("Loaded MuSiQue example missing 'question'")

    # Build a filtered example using only supporting facts for isolation
    supporting_facts = example.get("supporting_facts", []) or []
    paragraphs = example.get("paragraphs", []) or []
    supporting_set = set(supporting_facts) if isinstance(supporting_facts, list) else set()

    if supporting_set:
        filtered_paragraphs = []
        for p in paragraphs:
            if isinstance(p, dict):
                idx = p.get("idx")
                if idx in supporting_set:
                    filtered_paragraphs.append(p)
            else:
                # If paragraph is a string, fall back to positional filtering
                # (assumes supporting_facts indices align with order)
                pass
        # Fallback to positional filtering if needed
        if not filtered_paragraphs and isinstance(paragraphs, list):
            filtered_paragraphs = [
                p for i, p in enumerate(paragraphs) if i in supporting_set
            ]
        filtered_example = dict(example)
        filtered_example["paragraphs"] = filtered_paragraphs
        documents = load_musique_example_context_as_documents(filtered_example)
        print(f"✅ Loaded example {example_index} (supporting_facts only) with {len(documents)} documents")
    elif any(isinstance(p, dict) and p.get("is_supporting") for p in paragraphs):
        # Fallback: use paragraph-level is_supporting flag
        filtered_paragraphs = [p for p in paragraphs if isinstance(p, dict) and p.get("is_supporting")]
        filtered_example = dict(example)
        filtered_example["paragraphs"] = filtered_paragraphs
        documents = load_musique_example_context_as_documents(filtered_example)
        print(f"✅ Loaded example {example_index} (is_supporting only) with {len(documents)} documents")
    else:
        documents = load_musique_example_context_as_documents(example)
        print(f"✅ Loaded example {example_index} from {split} with {len(documents)} documents")
    print(f"Question: {question}")
    gold_answer = example.get("answer", "")
    if gold_answer:
        print(f"Gold answer: {gold_answer}")

    orchestrator = await create_optimized_marag_pipeline(documents=documents)

    # Initialize execution
    await orchestrator.state_manager.initialize_execution(
        execution_id="step1_debug",
        main_query=question,
        context=None
    )

    # Build plan and take only step 1
    plan_result = await orchestrator.planner.process({"query": question})
    plan = plan_result.metadata.get("plan", {})
    steps = plan.get("steps", [])
    if not steps:
        # Fallback: parse plan from content JSON
        try:
            import json as _json
            from agents.tokenization_utils import TokenizationUtils as _TU
            clean_plan = _TU.strip_markdown_json(plan_result.content)
            plan = _json.loads(clean_plan)
            steps = plan.get("steps", [])
        except Exception:
            steps = []
    if not steps:
        raise RuntimeError("Planner produced no steps")
    step1 = steps[0]

    print(f"\nRunning Step 1 only: {step1.get('description')}")

    result = await orchestrator._execute_single_step(
        step=step1,
        plan=plan,
        plan_goal=question,
        hop=1,
        current_step_index=0,
        total_steps=len(steps)
    )

    qa = result.get("qa_result", {})
    answer = qa.get("answer", "")
    print("\nStep 1 Answer:", answer)
    passages = result.get("extracted_passages", []) or []
    if passages:
        print("\nTop extracted passages (step 1):")
        for i, p in enumerate(passages[:3], start=1):
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            score = p.get("relevance", p.get("score", 0.0)) if isinstance(p, dict) else 0.0
            print(f"  [{i}] relevance={score:.3f} | {text[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Step-1 Only Smoke Test")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--example-index", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(run_step1_only(args.split, args.example_index))


if __name__ == "__main__":
    main()
