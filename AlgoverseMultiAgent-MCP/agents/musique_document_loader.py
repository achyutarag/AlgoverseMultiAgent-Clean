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
    
    # MuSiQue has 'paragraphs' field (list of paragraph texts)
    paragraphs = example.get('paragraphs', [])
    
    # Supporting facts indicate which paragraphs are needed
    supporting_facts = example.get('supporting_facts', [])
    
    # Create a document for each paragraph
    for para_idx, paragraph_text in enumerate(paragraphs):
        # Create metadata
        metadata = {
            "source": "musique",
            "paragraph_id": para_idx,
            "example_id": example.get('id', 'unknown')
        }
        
        if include_metadata:
            # Check if this paragraph is a supporting fact
            is_supporting = para_idx in supporting_facts if isinstance(supporting_facts, list) else False
            
            metadata.update({
                "is_supporting_fact": is_supporting,
                "question": example.get('question', ''),
                "answer": example.get('answer', ''),
                "num_paragraphs": len(paragraphs),
                "num_supporting_facts": len(supporting_facts) if isinstance(supporting_facts, list) else 0
            })
        
        # Create Document object
        doc = Document(
            page_content=paragraph_text,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def load_musique_context_as_documents(
    dataset_split: str = "train", 
    num_examples: int = 1000,
    include_metadata: bool = True
) -> List[Document]:
    """
    Load MuSiQue dataset and convert context paragraphs into Document objects for retrieval.
    
    Args:
        dataset_split: Which split to load ('train' or 'validation')
        num_examples: Number of examples to process (for memory management)
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of Document objects ready for the retriever agent
    """
    print(f"Loading MuSiQue {dataset_split} split...")
    # MuSiQue is typically available on HuggingFace as "allenai/musique"
    dataset = load_dataset("allenai/musique")
    
    # Get the specified split
    split_data = dataset[dataset_split]
    
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
        
        # Create a document for each paragraph
        for para_idx, paragraph_text in enumerate(paragraphs):
            # Create a unique ID for this paragraph
            para_id = f"{example.get('id', i)}_{para_idx}"
            
            # Skip if we've already processed this exact paragraph
            if para_id in processed_paragraphs:
                continue
            processed_paragraphs.add(para_id)
            
            # Create metadata
            metadata = {
                "source": "musique",
                "paragraph_id": para_idx,
                "paragraph_unique_id": para_id,
                "dataset_split": dataset_split,
                "example_id": example.get('id', 'unknown')
            }
            
            if include_metadata:
                supporting_facts = example.get('supporting_facts', [])
                is_supporting = para_idx in supporting_facts if isinstance(supporting_facts, list) else False
                
                metadata.update({
                    "is_supporting_fact": is_supporting,
                    "question_type": example.get('type', 'unknown'),
                    "answer": example.get('answer', '')
                })
            
            # Create Document object
            doc = Document(
                page_content=paragraph_text,
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
    dataset = load_dataset("allenai/musique")
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

