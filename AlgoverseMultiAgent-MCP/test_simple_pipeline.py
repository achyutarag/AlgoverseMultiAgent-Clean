import asyncio
import logging
from agents.mixed_model_orchestrator import run_optimized_marag_pipeline

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_pipeline():
    """Test the MA-RAG pipeline with a simple question."""
    print("\n" + "="*60)
    print("Testing MA-RAG Pipeline")
    print("="*60 + "\n")
    
    query = "Who was the man behind The Chipmunks?"
    print(f"Question: {query}\n")
    
    try:
        print("Calling run_optimized_marag_pipeline...")
        result = await run_optimized_marag_pipeline(query)
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nResult type: {type(result)}")
        
        if hasattr(result, 'content'):
            print(f"\n--- Content ---")
            print(result.content[:500] if len(result.content) > 500 else result.content)
            
        if hasattr(result, 'metadata'):
            print(f"\n--- Metadata ---")
            for key, value in result.metadata.items():
                print(f"{key}: {value}")
            
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR OCCURRED")
        print("="*60)
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}\n")
        
        print("Full traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
