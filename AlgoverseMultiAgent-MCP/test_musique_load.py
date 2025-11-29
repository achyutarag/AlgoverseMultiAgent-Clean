import traceback
import json
from datasets import load_dataset

def _load_musique_from_github(dataset_split: str = "validation") -> list[dict]:
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
    project_root = current_dir  # test_musique_load.py is in project root
    cwd = os.getcwd()  # Current working directory
    
    possible_paths = [
        os.path.join(project_root, "musique_data_v1.0", "data", filename),  # In musique_data_v1.0/data/
        os.path.join(project_root, "data", filename),  # In project_root/data/
        os.path.join(cwd, "musique_data_v1.0", "data", filename),  # In cwd/musique_data_v1.0/data/
        os.path.join(cwd, "data", filename),  # In cwd/data/
        os.path.join(project_root, filename),  # In project root
        os.path.join(cwd, filename),  # In current directory
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
            f"\n\nPlease ensure the extracted 'data' folder is in the project directory."
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



# ============================================================================
# ACTUAL TEST CODE - This is what runs when you execute the script
# ============================================================================

if __name__ == "__main__":
    print("Testing MuSiQue dataset loading...")
    print("=" * 60)
    
    # Test 1: Try HuggingFace
    print("\n" + "=" * 60)
    print("TEST 1: HuggingFace Loading")
    print("=" * 60)
    
    test_paths = [
        "allenai/musique-v1",
        "allenai/musique",
        "StonyBrookNLP/musique",
    ]
    
    huggingface_success = False
    for path in test_paths:
        print(f"\nTrying: {path}")
        try:
            dataset = load_dataset(path)
            print(f"✅ SUCCESS with {path}")
            print(f"   Available splits: {list(dataset.keys())}")
            if "validation" in dataset:
                print(f"   Validation examples: {len(dataset['validation'])}")
                if len(dataset['validation']) > 0:
                    sample = dataset['validation'][0]
                    print(f"   Sample keys: {list(sample.keys())}")
            huggingface_success = True
            break
        except Exception as e:
            print(f"❌ FAILED with {path}")
            print(f"   Error: {type(e).__name__}: {str(e)}")
            if "401" in str(e) or "Unauthorized" in str(e):
                print("   → This suggests authentication is required")
            elif "404" in str(e) or "not found" in str(e).lower():
                print("   → This suggests the dataset doesn't exist at this path")
            traceback.print_exc()
            print()
    
    if not huggingface_success:
        print("\nHuggingFace loading failed for all tested paths.")
    
    # Test 2: Try Google Drive fallback
    print("\n" + "=" * 60)
    print("TEST 2: Google Drive Fallback Loading")
    print("=" * 60)
    
    try:
        print("\nTesting Google Drive download...")
        examples = _load_musique_from_github("validation")
        print(f"✅ SUCCESS: Loaded {len(examples)} examples from Google Drive")
        if examples:
            print(f"   First example keys: {list(examples[0].keys())}")
            print(f"   Sample question: {examples[0].get('question', 'N/A')[:100]}...")
            # Add this to check the answer field:
            answer = examples[0].get('answer', 'NOT FOUND')
            print(f"   Sample answer: {answer}")
            print(f"   Answer type: {type(answer)}")
            if isinstance(answer, list):
                print(f"   Answer is a list with {len(answer)} items")
            elif isinstance(answer, dict):
                print(f"   Answer is a dict with keys: {list(answer.keys())}")
    except Exception as e:
        print(f"❌ FAILED to load from Google Drive")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)