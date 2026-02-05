"""
MuSiQue Document Loader

This module provides functions to load and convert MuSiQue dataset context
into documents suitable for the retriever agent in the multi-agent pipeline.

MuSiQue Structure:
- Questions requiring 2-4 hops
- 20-30 context paragraphs per question (more challenging than HotpotQA's 10)
- Supporting facts indicating which paragraphs are needed
- Answers
"""

from typing import List, Dict, Any
from langchain.schema import Document
from datasets import load_dataset
import json
import os
import requests
from pathlib import Path


def load_musique_example_context_as_documents(
    example: Dict[str, Any],
    include_metadata: bool = True
) -> List[Document]:
    """
    Convert a single MuSiQue example's context paragraphs into Document objects.
    
    MuSiQue Intent:
    - Each question has 20-30 context paragraphs
    - Only 2-4 paragraphs are actually needed (supporting facts)
    - Tests retrieval over a larger pool than HotpotQA (10 docs)
    - Requires more hops (2-4) than HotpotQA (typically 2)
    
    Args:
        example: A single MuSiQue example with 'paragraphs' field
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of Document objects from this example's context paragraphs
    """
    documents = []
    
    # MuSiQue has 'paragraphs' field (list of paragraph dictionaries or strings)
    paragraphs = example.get('paragraphs', [])
    
    # Supporting facts indicate which paragraphs are needed
    supporting_facts = example.get('supporting_facts', [])
    
    # Create a document for each paragraph
    for para_idx, paragraph_item in enumerate(paragraphs):
        # Handle both string and dict formats
        if isinstance(paragraph_item, dict):
            # ✅ FIRST PRINCIPLES FIX: MuSiQue paragraphs use 'paragraph_text' field, not 'text'
            # Check 'paragraph_text' FIRST (the actual field name in MuSiQue JSON)
            # This prevents extracting document titles instead of actual paragraph content
            paragraph_text = paragraph_item.get('paragraph_text',
                          paragraph_item.get('text', 
                          paragraph_item.get('content', 
                          paragraph_item.get('paragraph',
                          paragraph_item.get('title', str(paragraph_item))))))
            
            # Get paragraph index from dict if available, otherwise use enumerate index
            para_idx_from_dict = paragraph_item.get('idx', para_idx)
            
            # Get supporting fact status from dict if available
            is_supporting_from_dict = paragraph_item.get('is_supporting', False)
        else:
            # If it's already a string, use it directly
            paragraph_text = str(paragraph_item)
            para_idx_from_dict = para_idx
            is_supporting_from_dict = False
        
        # Create metadata
        metadata = {
            "source": "musique",
            "paragraph_id": para_idx_from_dict,
            "example_id": example.get('id', 'unknown')
        }
        
        if include_metadata:
            # Check if this paragraph is a supporting fact
            # Use dict value if available, otherwise check the supporting_facts list
            if isinstance(paragraph_item, dict) and 'is_supporting' in paragraph_item:
                is_supporting = is_supporting_from_dict
            else:
                is_supporting = para_idx_from_dict in supporting_facts if isinstance(supporting_facts, list) else False
            
            metadata.update({
                "is_supporting_fact": is_supporting,
                "question": example.get('question', ''),
                "answer": example.get('answer', ''),
                "num_paragraphs": len(paragraphs),
                "num_supporting_facts": len(supporting_facts) if isinstance(supporting_facts, list) else 0
            })
        
        # Create Document object
        doc = Document(
            page_content=paragraph_text,  # Now guaranteed to be a string
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def _load_musique_from_github(dataset_split: str = "validation") -> List[Dict[str, Any]]:
    """
    Load MuSiQue dataset from local JSONL files.
    
    Args:
        dataset_split: Which split to load ('train' or 'validation')
        
    Returns:
        List of examples from the dataset
    """
    import os
    import json
    
    # Map split names to file names
    split_files = {
        "train": "musique_ans_v1.0_train.jsonl",
        "validation": "musique_ans_v1.0_dev.jsonl",
        "dev": "musique_ans_v1.0_dev.jsonl"
    }
    
    filename = split_files.get(dataset_split, split_files["validation"])
    
    # Check for local JSONL file in multiple locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up from agents/ to Clean directory
    cwd = os.getcwd()  # Current working directory
    
    # ✅ FIX: Also check parent directory and MCP sibling directory
    parent_dir = os.path.dirname(project_root)  # Parent of Clean (AlgoverseMultiAgent)
    mcp_dir = os.path.join(parent_dir, "AlgoverseMultiAgent-MCP")  # MCP sibling directory
    
    possible_paths = [
        # Check in Clean directory first
        os.path.join(project_root, "musique_data_v1.0", "data", filename),  # In Clean/musique_data_v1.0/data/
        os.path.join(project_root, "data", filename),  # In Clean/data/
        os.path.join(cwd, "musique_data_v1.0", "data", filename),  # In cwd/musique_data_v1.0/data/
        os.path.join(cwd, "data", filename),  # In cwd/data/
        os.path.join(project_root, filename),  # In Clean root
        os.path.join(cwd, filename),  # In current directory
        os.path.join(current_dir, "..", "data", filename),  # Parent/data/
        # ✅ FIX: Check MCP directory (where dataset actually is)
        os.path.join(mcp_dir, "musique_data_v1.0", "data", filename),  # In MCP/musique_data_v1.0/data/
        os.path.join(mcp_dir, "data", filename),  # In MCP/data/
        # ✅ FIX: Check parent directory
        os.path.join(parent_dir, "musique_data_v1.0", "data", filename),  # In parent/musique_data_v1.0/data/
        os.path.join(parent_dir, "data", filename),  # In parent/data/
    ]
    
    print(f"Looking for {filename} in:")
    jsonl_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"  - {abs_path} {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists and jsonl_path is None:
            jsonl_path = abs_path
    
    if not jsonl_path:
        raise FileNotFoundError(
            f"Could not find {filename} in any of these locations:\n" +
            "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths) +
            f"\n\nPlease extract musique_v1.0.zip and place the 'data' folder in the project directory, "
            f"or ensure the dataset exists in AlgoverseMultiAgent-MCP/musique_data_v1.0/data/"
        )
    
    print(f"\n✅ Found file: {jsonl_path}")
    print(f"Loading examples from {jsonl_path}...")
    
    # Read and parse JSONL file
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
    
    print(f"✅ Successfully loaded {len(examples)} examples")
    return examples

def load_musique_context_as_documents(
    dataset_split: str = "train", 
    num_examples: int = 1000,
    include_metadata: bool = True
) -> List[Document]:
    """
    Load MuSiQue dataset and convert context paragraphs into Document objects for retrieval.
    
    Tries to load from HuggingFace first, falls back to GitHub if needed.
    
    Args:
        dataset_split: Which split to load ('train' or 'validation')
        num_examples: Number of examples to process (for memory management)
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of Document objects ready for the retriever agent
    """
    print(f"Loading MuSiQue {dataset_split} split...")
    
    # Try HuggingFace first
    dataset = None
    split_data = None
    
    try:
        print("Attempting to load from HuggingFace: allenai/musique-v1")
        dataset = load_dataset("allenai/musique-v1")
        split_data = dataset[dataset_split]
        print(f"✅ Successfully loaded from HuggingFace")
    except Exception as e:
        print(f"❌ Failed to load from HuggingFace: {e}")
        print("Attempting to load from GitHub repository...")
        
        # Fallback to GitHub
        examples = _load_musique_from_github(dataset_split)
        
        # Convert to dataset-like format
        # Create a simple list that can be indexed like a dataset
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, idx):
                return self.data[idx]
            def __len__(self):
                return len(self.data)
            def select(self, indices):
                return SimpleDataset([self.data[i] for i in indices])
        
        split_data = SimpleDataset(examples)
    
    # Limit number of examples for memory management
    if num_examples > 0:
        split_data = split_data.select(range(min(num_examples, len(split_data))))
    
    documents = []
    processed_paragraphs = set()  # To avoid duplicates (if same paragraph appears in multiple examples)
    
    print(f"Processing {len(split_data)} examples...")
    
    for i, example in enumerate(split_data):
        if i % 100 == 0:
            print(f"Processed {i}/{len(split_data)} examples...")
            
        paragraphs = example.get('paragraphs', [])
        
        # ✅ FIRST PRINCIPLES FIX: Handle paragraphs as dictionaries (not strings)
        # MuSiQue paragraphs are dictionaries with 'paragraph_text', 'title', 'idx', etc.
        # Create a document for each paragraph
        for para_idx, paragraph_item in enumerate(paragraphs):
            # Handle both string and dict formats
            if isinstance(paragraph_item, dict):
                # ✅ FIX: Check 'paragraph_text' FIRST (the actual field name in MuSiQue JSON)
                paragraph_text = paragraph_item.get('paragraph_text',
                                  paragraph_item.get('text',
                                  paragraph_item.get('content',
                                  paragraph_item.get('paragraph',
                                  paragraph_item.get('title', str(paragraph_item))))))
                para_idx_from_dict = paragraph_item.get('idx', para_idx)
            else:
                # If it's already a string, use it directly
                paragraph_text = str(paragraph_item)
                para_idx_from_dict = para_idx
            
            # Create a unique ID for this paragraph
            para_id = f"{example.get('id', i)}_{para_idx_from_dict}"
            
            # Skip if we've already processed this exact paragraph
            if para_id in processed_paragraphs:
                continue
            processed_paragraphs.add(para_id)
            
            # Create metadata
            metadata = {
                "source": "musique",
                "paragraph_id": para_idx_from_dict,
                "paragraph_unique_id": para_id,
                "dataset_split": dataset_split,
                "example_id": example.get('id', 'unknown')
            }
            
            if include_metadata:
                supporting_facts = example.get('supporting_facts', [])
                is_supporting = para_idx_from_dict in supporting_facts if isinstance(supporting_facts, list) else False
                
                metadata.update({
                    "is_supporting_fact": is_supporting,
                    "question_type": example.get('type', 'unknown'),
                    "answer": example.get('answer', '')
                })
            
            # Create Document object
            doc = Document(
                page_content=paragraph_text,  # Now correctly extracts paragraph_text field
                metadata=metadata
            )
            documents.append(doc)
    
    print(f"Created {len(documents)} documents from {len(processed_paragraphs)} unique paragraphs")
    return documents


def verify_musique_structure(example: Dict[str, Any], documents: List[Document]) -> Dict[str, Any]:
    """
    Verify that MuSiQue example has correct structure: multiple paragraphs with supporting facts.
    
    Returns:
        Dict with verification results
    """
    paragraphs = example.get('paragraphs', [])
    supporting_facts = example.get('supporting_facts', [])
    
    # Count supporting facts
    if isinstance(supporting_facts, list):
        num_supporting = len(supporting_facts)
    else:
        num_supporting = 0
    
    # Get loaded document count
    num_docs = len(documents)
    num_paragraphs = len(paragraphs)
    
    # Check if supporting facts are in loaded docs
    supporting_in_loaded = 0
    if isinstance(supporting_facts, list):
        for sf_idx in supporting_facts:
            if sf_idx < num_docs:
                supporting_in_loaded += 1
    
    return {
        "total_paragraphs": num_paragraphs,
        "total_documents": num_docs,
        "supporting_facts": num_supporting,
        "supporting_facts_in_loaded": supporting_in_loaded,
        "is_valid": (num_docs == num_paragraphs and num_supporting > 0)
    }


if __name__ == "__main__":
    # Test the document loader
    print("Testing MuSiQue document loader...")
    
    # Load a small sample
    dataset = load_dataset("allenai/musique-v1")
    sample_example = dataset["validation"][0]
    
    docs = load_musique_example_context_as_documents(sample_example)
    print(f"\nLoaded {len(docs)} documents")
    
    # Show verification
    verification = verify_musique_structure(sample_example, docs)
    print(f"\nVerification: {verification}")
    
    # Show first document
    if docs:
        print(f"\nFirst document:")
        print(f"Paragraph ID: {docs[0].metadata.get('paragraph_id')}")
        print(f"Content preview: {docs[0].page_content[:200]}...")
        print(f"Metadata: {docs[0].metadata}")

