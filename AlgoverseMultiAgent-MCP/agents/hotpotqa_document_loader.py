"""
HotpotQA Document Loader

This module provides functions to load and convert HotpotQA dataset context
into documents suitable for the retriever agent in the multi-agent pipeline.
"""

from typing import List, Dict, Any
from langchain.schema import Document
from datasets import load_dataset


def load_hotpotqa_context_as_documents(
    dataset_split: str = "train", 
    num_examples: int = 1000,
    include_metadata: bool = True
) -> List[Document]:
    """
    Load HotpotQA dataset and convert context into Document objects for retrieval.
    
    Args:
        dataset_split: Which split to load ('train' or 'validation')
        num_examples: Number of examples to process (for memory management)
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of Document objects ready for the retriever agent
    """
    print(f"Loading HotpotQA {dataset_split} split...")
    dataset = load_dataset("hotpot_qa", "distractor")
    
    # Get the specified split
    split_data = dataset[dataset_split]
    
    # Limit number of examples for memory management
    if num_examples > 0:
        split_data = split_data.select(range(min(num_examples, len(split_data))))
    
    documents = []
    processed_articles = set()  # To avoid duplicates
    
    print(f"Processing {len(split_data)} examples...")
    
    for i, example in enumerate(split_data):
        if i % 100 == 0:
            print(f"Processed {i}/{len(split_data)} examples...")
            
        context = example['context']
        titles = context['title']
        sentences_lists = context['sentences']
        
        # Create a document for each article
        for title, sentences in zip(titles, sentences_lists):
            # Skip if we've already processed this article
            if title in processed_articles:
                continue
            processed_articles.add(title)
            
            # Combine all sentences for this article
            article_content = " ".join(sentences)
            
            # Create metadata
            metadata = {
                "source": "hotpotqa",
                "title": title,
                "article_id": f"hotpotqa_{title.replace(' ', '_').replace('/', '_')}",
                "dataset_split": dataset_split,
                "example_id": example['id']
            }
            
            if include_metadata:
                metadata.update({
                    "question_type": example.get('type', 'unknown'),
                    "difficulty_level": example.get('level', 'unknown'),
                    "supporting_facts": example.get('supporting_facts', {}),
                    "answer": example.get('answer', '')
                })
            
            # Create Document object
            doc = Document(
                page_content=article_content,
                metadata=metadata
            )
            documents.append(doc)
    
    print(f"Created {len(documents)} documents from {len(processed_articles)} unique articles")
    return documents


def load_hotpotqa_example_context_as_documents(
    example: Dict[str, Any],
    include_metadata: bool = True
) -> List[Document]:
    """
    Convert a single HotpotQA example's context into Document objects.
    
    Args:
        example: A single HotpotQA example
        include_metadata: Whether to include additional metadata
        
    Returns:
        List of Document objects from this example's context
    """
    documents = []
    context = example['context']
    titles = context['title']
    sentences_lists = context['sentences']
    
    # Create a document for each article in this example's context
    for title, sentences in zip(titles, sentences_lists):
        # Combine all sentences for this article
        article_content = " ".join(sentences)
        
        # Create metadata
        metadata = {
            "source": "hotpotqa",
            "title": title,
            "article_id": f"hotpotqa_{title.replace(' ', '_').replace('/', '_')}",
            "example_id": example['id']
        }
        
        if include_metadata:
            metadata.update({
                "question_type": example.get('type', 'unknown'),
                "difficulty_level": example.get('level', 'unknown'),
                "supporting_facts": example.get('supporting_facts', {}),
                "answer": example.get('answer', ''),
                "question": example.get('question', '')
            })
        
        # Create Document object
        doc = Document(
            page_content=article_content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def get_hotpotqa_sample_questions(
    dataset_split: str = "validation",
    num_questions: int = 5
) -> List[Dict[str, Any]]:
    """
    Get sample questions from HotpotQA dataset for testing.
    
    Args:
        dataset_split: Which split to load ('train' or 'validation')
        num_questions: Number of sample questions to return
        
    Returns:
        List of question dictionaries with context documents
    """
    print(f"Loading {num_questions} sample questions from HotpotQA {dataset_split}...")
    dataset = load_dataset("hotpot_qa", "distractor")
    split_data = dataset[dataset_split].select(range(num_questions))
    
    sample_questions = []
    for example in split_data:
        # Get context documents for this example
        context_docs = load_hotpotqa_example_context_as_documents(example)
        
        sample_questions.append({
            "question": example['question'],
            "answer": example['answer'],
            "question_type": example.get('type', 'unknown'),
            "difficulty_level": example.get('level', 'unknown'),
            "supporting_facts": example.get('supporting_facts', {}),
            "context_documents": context_docs
        })
    
    return sample_questions


if __name__ == "__main__":
    # Test the document loader
    print("Testing HotpotQA document loader...")
    
    # Load a small sample
    docs = load_hotpotqa_context_as_documents("validation", num_examples=10)
    print(f"\nLoaded {len(docs)} documents")
    
    # Show first document
    if docs:
        print(f"\nFirst document:")
        print(f"Title: {docs[0].metadata['title']}")
        print(f"Content preview: {docs[0].page_content[:200]}...")
        print(f"Metadata: {docs[0].metadata}")
    
    # Test sample questions
    print(f"\nTesting sample questions...")
    sample_questions = get_hotpotqa_sample_questions("validation", num_questions=2)
    for i, q in enumerate(sample_questions):
        print(f"\nQuestion {i+1}: {q['question']}")
        print(f"Answer: {q['answer']}")
        print(f"Context documents: {len(q['context_documents'])}")
