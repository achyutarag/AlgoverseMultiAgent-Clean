"""
Runtime diagnostic to check if diffusion components are actually WORKING during execution,
not just initialized. This runs a real query and monitors what happens.
"""

import asyncio
import logging
from agents.mixed_model_orchestrator import create_optimized_marag_pipeline
from agents.musique_document_loader import load_musique_example_context_as_documents
from datasets import load_dataset

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see all details
    format='%(levelname)s - %(name)s - %(message)s'
)

# Track what happens during execution
execution_trace = {
    "queries": [],
    "stabilized_queries": [],
    "entropy_values": [],
    "diffusion_values": [],
    "retrieval_k_values": [],
    "retrieval_min_sim_values": [],
    "doc_counts": [],
    "regulator_changes": []
}

# Monkey-patch to capture what's happening
original_stabilize = None
original_entropy_retrieve = None

def capture_stabilize_and_retrieve(original_func):
    """Capture calls to stabilize_and_retrieve."""
    async def wrapper(proposed_query, hop, previous_answers, plan_goal=None, retriever_agent=None, current_step_index=None, total_steps=None):
        execution_trace["queries"].append(proposed_query)
        result = await original_func(proposed_query, hop, previous_answers, plan_goal, retriever_agent, current_step_index, total_steps)
        stabilized = result.get("stabilized_query", proposed_query)
        execution_trace["stabilized_queries"].append(stabilized)
        
        # Check if query changed
        if stabilized != proposed_query:
            execution_trace["regulator_changes"].append({
                "original": proposed_query,
                "stabilized": stabilized,
                "hop": hop
            })
        
        # Get entropy/diffusion from flow_snapshot
        flow_snapshot = result.get("flow_snapshot")
        if flow_snapshot:
            if isinstance(flow_snapshot, dict):
                entropy = flow_snapshot.get("entropy", 0.0)
                diffusion = flow_snapshot.get("diffusion_coefficient", 0.0)
            else:
                entropy = getattr(flow_snapshot, "entropy", 0.0)
                diffusion = getattr(flow_snapshot, "diffusion_coefficient", 0.0)
            execution_trace["entropy_values"].append(entropy)
            execution_trace["diffusion_values"].append(diffusion)
        
        return result
    return wrapper

def capture_retriever_process(original_func):
    """Capture retriever calls to see if entropy-aware adjustments happen."""
    async def wrapper(self, input_data):
        # Check if entropy-aware
        entropy_penalty = float(input_data.get('entropy_penalty', 0.0))
        diffusion_penalty = float(input_data.get('diffusion_penalty', 0.0))
        is_entropy_aware = (entropy_penalty > 0.0 or diffusion_penalty > 0.0 or 
                           input_data.get('regulator_constraints') or input_data.get('flow_snapshot'))
        
        base_k = int(input_data.get('k', self.top_k))
        base_min_sim = float(input_data.get('min_similarity', self.min_similarity))
        
        # ✅ FIX: original_func is already bound, don't pass self again
        result = await original_func(input_data)
        
        # Check what k was actually used (from logs or result)
        docs = result.metadata.get('documents', [])
        execution_trace["doc_counts"].append(len(docs))
        execution_trace["retrieval_k_values"].append(base_k)
        execution_trace["retrieval_min_sim_values"].append(base_min_sim)
        
        if is_entropy_aware:
            print(f"  🎯 Entropy-aware retrieval detected: H(t)={entropy_penalty:.3f}, D(t)={diffusion_penalty:.3f}")
        else:
            print(f"  ⚠️  Direct retrieval (no entropy/diffusion values)")
        
        return result
    return wrapper

async def test_runtime_diffusion():
    """Test if diffusion is actually working during execution."""
    print("\n" + "="*60)
    print("RUNTIME DIFFUSION DIAGNOSTIC")
    print("="*60)
    print("\nThis will run a real MuSiQue query and check if:")
    print("1. Queries are being stabilized by regulators")
    print("2. Entropy/diffusion values are being tracked")
    print("3. Adaptive retrieval is happening (k adjusted, similarity adjusted)")
    print("4. More documents retrieved due to entropy adjustments")
    
    try:
        # Load a MuSiQue example
        print("\n📚 Loading MuSiQue example...")
        from agents.musique_document_loader import _load_musique_from_github
        examples = _load_musique_from_github("validation")
        example = examples[0]
        
        question = example.get('question', '')
        print(f"Question: {question}")
        
        # Load documents
        documents = load_musique_example_context_as_documents(example)
        print(f"✅ Loaded {len(documents)} documents")
        
        # Create orchestrator
        print("\n🔧 Creating orchestrator...")
        orchestrator = await create_optimized_marag_pipeline()
        orchestrator.add_documents(documents)
        
        # Patch to capture execution
        from agents.state_manager import StateManager
        import types
        
        # Store original methods
        sm = orchestrator.state_manager
        original_stabilize = sm.stabilize_and_retrieve
        original_retrieve = orchestrator.retriever.process
        
        # Wrap methods to capture data
        # Create a wrapper that preserves the bound method signature
        wrapped_stabilize = capture_stabilize_and_retrieve(original_stabilize)
        sm.stabilize_and_retrieve = wrapped_stabilize
        orchestrator.retriever.process = types.MethodType(
            capture_retriever_process(original_retrieve),
            orchestrator.retriever
        )
        
        print("\n🚀 Running query through pipeline...")
        print("="*60)
        
        result = await orchestrator.execute_pipeline(question)
        
        print("\n" + "="*60)
        print("EXECUTION ANALYSIS")
        print("="*60)
        
        # Analyze what happened
        print(f"\n📊 Queries processed: {len(execution_trace['queries'])}")
        print(f"📊 Stabilized queries: {len(execution_trace['stabilized_queries'])}")
        
        # Check if queries were stabilized
        query_changes = execution_trace['regulator_changes']
        if query_changes:
            print(f"\n✅ REGULATORS ARE WORKING: {len(query_changes)} queries were stabilized")
            for i, change in enumerate(query_changes[:3], 1):  # Show first 3
                print(f"\n  Change {i} (Hop {change['hop']}):")
                print(f"    Original: {change['original'][:80]}...")
                print(f"    Stabilized: {change['stabilized'][:80]}...")
        else:
            print("\n⚠️  WARNING: No query stabilization detected!")
            print("   Regulators may not be modifying queries")
        
        # Check entropy tracking
        entropy_vals = execution_trace['entropy_values']
        if entropy_vals:
            avg_entropy = sum(entropy_vals) / len(entropy_vals)
            max_entropy = max(entropy_vals)
            print(f"\n📈 Entropy Tracking:")
            print(f"   Average H(t): {avg_entropy:.3f}")
            print(f"   Max H(t): {max_entropy:.3f}")
            print(f"   Values: {[f'{v:.3f}' for v in entropy_vals[:5]]}...")
            
            if max_entropy > 0.0:
                print("   ✅ Entropy is being tracked")
            else:
                print("   ⚠️  WARNING: All entropy values are 0.0!")
                print("      Entropy-aware retrieval won't activate")
        else:
            print("\n❌ CRITICAL: No entropy values tracked!")
            print("   Flow snapshots may not be created")
        
        # Check diffusion tracking
        diffusion_vals = execution_trace['diffusion_values']
        if diffusion_vals:
            avg_diffusion = sum(diffusion_vals) / len(diffusion_vals)
            print(f"\n📈 Diffusion Tracking:")
            print(f"   Average D(t): {avg_diffusion:.3f}")
            print(f"   Values: {[f'{v:.3f}' for v in diffusion_vals[:5]]}...")
        else:
            print("\n⚠️  No diffusion values tracked")
        
        # Check adaptive retrieval
        k_vals = execution_trace['retrieval_k_values']
        if k_vals:
            base_k = k_vals[0] if k_vals else 10
            max_k = max(k_vals)
            print(f"\n🔍 Retrieval Parameters:")
            print(f"   Base k: {base_k}")
            print(f"   Max k used: {max_k}")
            print(f"   k values: {k_vals[:5]}...")
            
            if max_k > base_k:
                print(f"   ✅ Adaptive k is working (increased from {base_k} to {max_k})")
            else:
                print(f"   ⚠️  k not adjusted (stayed at {base_k})")
                print("      Entropy-aware adjustments may not be activating")
        
        # Check document counts
        doc_counts = execution_trace['doc_counts']
        if doc_counts:
            avg_docs = sum(doc_counts) / len(doc_counts)
            max_docs = max(doc_counts)
            print(f"\n📄 Documents Retrieved:")
            print(f"   Average: {avg_docs:.1f}")
            print(f"   Max: {max_docs}")
            print(f"   Counts: {doc_counts[:5]}...")
        
        # Final verdict
        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)
        
        issues = []
        if not query_changes:
            issues.append("❌ Queries not being stabilized by regulators")
        if not entropy_vals or max(entropy_vals) == 0.0:
            issues.append("❌ Entropy tracking not working (all values 0.0)")
        if k_vals and max(k_vals) == k_vals[0]:
            issues.append("⚠️  Adaptive k not adjusting (entropy-aware retrieval may not activate)")
        
        if issues:
            print("\n🚨 ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            print("\n   These issues could explain why diffusion isn't helping!")
            print("   The pipeline may be running but diffusion features aren't activating.")
        else:
            print("\n✅ Diffusion components appear to be working during execution")
            print("   If results are still similar to Clean, the issue may be:")
            print("   - Regulators too conservative (not changing queries enough)")
            print("   - Entropy values too low (not triggering adaptive retrieval)")
            print("   - MuSiQue questions don't benefit from diffusion approach")
        
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during runtime test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_runtime_diffusion())