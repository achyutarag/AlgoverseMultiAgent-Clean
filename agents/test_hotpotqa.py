"""
Debug script to test the planner agent and see what Gemini is generating.
This will help us understand the JSON parsing issue.
"""
import asyncio
import json
from .planner_agent import PlannerAgent
from .llm_wrapper import get_llm, LLMConfig
from .tokenization_utils import TokenizationUtils  # Move this here

async def test_planner_debug():
    """Test the planner agent and see the raw response."""
    print("🔍 Testing Planner Agent Debug...")
    print("=" * 50)
    
    try:
        # First, let's test the LLM directly to see the original response
        print("🔧 Testing LLM directly...")
        llm_config = LLMConfig(
            model_name="gemini-2.5-pro-preview-03-25",
            model_type="google_gemini"
        )
        llm = get_llm(llm_config)
        
        # Test with a simple question
        test_query = "Which magazine was started first Arthur's Magazine or First for Women?"
        
        print(f"📝 Test Query: {test_query}")
        print("\n🔄 Calling LLM directly...")
        
        # Call LLM directly to see original response
        original_response = await llm.generate(test_query)
        print(f"📄 ORIGINAL LLM RESPONSE: {original_response.text}")
        print(f"📄 ORIGINAL RESPONSE LENGTH: {len(original_response.text)}")
        
        # Now test the planner agent
        print("\n" + "=" * 50)
        print("🔧 Testing Planner Agent...")
        
        # Create planner agent
        planner = PlannerAgent()
        
        print(f"📝 Test Query: {test_query}")
        print("\n🔄 Processing with Planner Agent...")
        
        # Process the query
        result = await planner.process({'query': test_query})
        
        print(f"\n📊 Response Status: {result.metadata.get('status', 'unknown')}")
        print(f"📄 Raw Content Length: {len(result.content)}")
        print(f"📄 Raw Content Preview: {result.content[:200]}...")
        print(f"\n📄 FULL RAW RESPONSE:\n{result.content}\n")
        
        # Try to parse as JSON using the NEW logic
        try:
            # Use the new strip_markdown_json function
            clean_response = TokenizationUtils.strip_markdown_json(result.content)
            json_data = json.loads(clean_response)
            print("✅ JSON Parsing: SUCCESS")
            print(f"📋 JSON Keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
            print(f"📋 Number of Steps: {len(json_data.get('steps', []))}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parsing: FAILED")
            print(f"🔍 Error: {e}")
            print(f"🔍 Error Position: {e.pos}")
            print(f"🔍 Error Line: {e.lineno}")
            print(f"🔍 Error Column: {e.colno}")
            
            # Show the problematic area
            start = max(0, e.pos - 50)
            end = min(len(result.content), e.pos + 50)
            print(f"🔍 Problematic Area: ...{result.content[start:end]}...")
        
        # Check metadata
        print(f"\n📋 Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_planner_debug())