#!/usr/bin/env python3
"""
Test script for the tokenization utility to ensure reliability.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tokenization_utils import tokenization_utils

def test_tokenization_utility():
    """Test the tokenization utility with various inputs."""
    
    print("🧪 Testing Tokenization Utility")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "name": "Query Normalization",
            "input": "  what   are   the   environmental   benefits   of   solar   energy???  ",
            "function": "normalize_query",
            "expected_features": ["proper spacing", "single question mark", "capitalization"]
        },
        {
            "name": "LLM Input Preprocessing",
            "input": "This is a test   with   multiple   spaces   and   weird   characters!!!",
            "function": "preprocess_llm_input",
            "expected_features": ["normalized whitespace", "cleaned punctuation"]
        },
        {
            "name": "Answer Post-processing",
            "input": "```json\n{\"answer\": \"This is a test response\"}\n```",
            "function": "postprocess_answer",
            "output_type": "json",
            "expected_features": ["removed markdown", "valid JSON"]
        },
        {
            "name": "Document Text Cleaning",
            "input": "This   document   has   multiple   spaces   and   special   characters   @#$%^&*()",
            "function": "clean_text_consistently",
            "context": "document",
            "expected_features": ["cleaned text", "preserved punctuation"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 30)
        
        # Get input
        input_text = test_case["input"]
        print(f"Input: '{input_text}'")
        
        # Process based on function
        if test_case["function"] == "normalize_query":
            result = tokenization_utils.normalize_query(input_text)
        elif test_case["function"] == "preprocess_llm_input":
            result = tokenization_utils.preprocess_llm_input(input_text)
        elif test_case["function"] == "postprocess_answer":
            output_type = test_case.get("output_type", "text")
            result = tokenization_utils.postprocess_answer(input_text, output_type)
        elif test_case["function"] == "clean_text_consistently":
            context = test_case.get("context", "general")
            result = tokenization_utils.clean_text_consistently(input_text, context)
        
        print(f"Output: '{result}'")
        
        # Get statistics
        stats = tokenization_utils.get_text_statistics(result)
        print(f"Stats: {stats}")
        
        # Validate JSON if applicable
        if test_case.get("output_type") == "json":
            is_valid = tokenization_utils.validate_json_output(result)
            print(f"Valid JSON: {is_valid}")
            
            if is_valid:
                try:
                    import json
                    parsed = json.loads(result)
                    print(f"Parsed JSON: {parsed}")
                except:
                    pass
        
        print()
    
    # Test JSON extraction
    print("\n6. JSON Extraction Test")
    print("-" * 30)
    mixed_text = "Here is some text with JSON: {\"key\": \"value\", \"number\": 123} and more text."
    extracted = tokenization_utils.extract_json_from_text(mixed_text)
    print(f"Mixed text: '{mixed_text}'")
    print(f"Extracted JSON: {extracted}")
    
    print("\n✅ Tokenization utility test completed!")
    print("\nKey Benefits:")
    print("• Consistent text preprocessing across all agents")
    print("• Reliable query normalization for experimental metrics")
    print("• Proper LLM input/output cleaning")
    print("• JSON validation and extraction")
    print("• Centralized tokenization for reproducibility")

if __name__ == "__main__":
    test_tokenization_utility()

