"""
Phase 3 Oracle Sanity Test

Injects only supporting paragraphs for a single MuSiQue example and runs
Extractor + QA to verify downstream logic without retrieval noise.
"""

import argparse
import asyncio
import os
import sys
import json

# Ensure MCP folder is first on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.musique_document_loader import _load_musique_from_github, load_musique_example_context_as_documents
from agents.extractor_agent import ExtractorAgent
from agents.qa_agent import QAAgent


def _filter_supporting(example: dict) -> dict:
    paragraphs = example.get("paragraphs", []) or []
    supporting = [p for p in paragraphs if isinstance(p, dict) and p.get("is_supporting")]
    cloned = dict(example)
    cloned["paragraphs"] = supporting
    return cloned


async def run_oracle(split: str, index: int, supporting_only: bool):
    examples = _load_musique_from_github(split)
    if index < 0 or index >= len(examples):
        raise ValueError(f"Index out of range: {index} (total={len(examples)})")

    example = examples[index]
    if supporting_only:
        example = _filter_supporting(example)

    question = example.get("question", "")
    gold = example.get("answer", "")

    documents = load_musique_example_context_as_documents(example)
    if not documents:
        print("⚠️ No documents available after filtering. Try --use-all.")
        return
    doc_dicts = [
        {
            "page_content": d.page_content,
            "metadata": d.metadata or {},
            "score": 1.0,
        }
        for d in documents
    ]

    extractor = ExtractorAgent()
    qa = QAAgent()

    extractor_input = {
        "query": question,
        "documents": doc_dicts,
        "min_relevance": 0.0,
        "max_documents": min(len(doc_dicts), 12),
    }
    extractor_resp = await extractor.process(extractor_input)
    extracted = json.loads(extractor_resp.content)
    passages = extracted.get("extracted_passages", [])

    qa_input = {
        "question": question,
        "context": passages,
        "step_context": {"description": question},
        "overall_query": question,
        "previous_answers": {},
        "hop": 1,
    }
    qa_resp = await qa.process(qa_input)
    qa_result = json.loads(qa_resp.content)

    print(f"\nQuestion: {question}")
    print(f"Gold answer: {gold}")
    print(f"QA answer: {qa_result.get('answer')}")
    print(f"Confidence: {qa_result.get('confidence')}")
    print(f"Supporting paragraphs: {len(doc_dicts)}")
    print(f"Extracted passages: {len(passages)}")


def main():
    parser = argparse.ArgumentParser(description="Oracle sanity test with supporting facts only.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--use-all", action="store_true", default=False,
                        help="Use all paragraphs instead of only supporting facts.")
    args = parser.parse_args()

    asyncio.run(run_oracle(args.split, args.index, not args.use_all))


if __name__ == "__main__":
    main()
