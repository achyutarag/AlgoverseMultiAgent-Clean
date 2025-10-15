import asyncio
import json
from agents.planner_agent import PlannerAgent

async def test_planner():
    """Test just the planner to see what it generates."""
    print("\n" + "="*60)
    print("Testing Planner Agent Only")
    print("="*60 + "\n")
    
    query = "Who was the man behind The Chipmunks?"
    print(f"Question: {query}\n")
    
    try:
        planner = PlannerAgent()
        result = await planner.process({'query': query})
        
        print("\n" + "="*60)
        print("PLANNER RESULT")
        print("="*60)
        
        print(f"\nContent (first 500 chars):")
        print(result.content[:500] if len(result.content) > 500 else result.content)
        
        print(f"\n\nMetadata:")
        for key, value in result.metadata.items():
            if key == 'steps':
                print(f"\n{key}:")
                for i, step in enumerate(value):
                    print(f"  Step {i+1}: {step.get('id', 'NO ID')} - {step.get('description', 'NO DESC')[:50]}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_planner())
