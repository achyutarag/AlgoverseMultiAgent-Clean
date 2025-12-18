"""
Diagnostic script to check if diffusion-aware components are actually being used
in the MCP pipeline during evaluation.

Run this to verify:
1. StateManager initializes diffusion components
2. Orchestrator uses them
3. stabilize_and_retrieve is called (not fallback)
4. No silent failures
"""

import logging
import sys
import traceback
from agents.state_manager import StateManager
from agents.mixed_model_orchestrator import create_optimized_marag_pipeline

# Set up logging to capture warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

def check_state_manager():
    """Check if StateManager initializes diffusion components."""
    print("\n" + "="*60)
    print("TEST 1: StateManager Diffusion Component Initialization")
    print("="*60)
    
    try:
        sm = StateManager()
        
        # Check components
        checks = {
            "Entropy Tracker": sm.entropy_tracker is not None,
            "Reasoning Flow": sm.reasoning_flow is not None,
            "Regulator Manager": sm.regulator_manager is not None,
        }
        
        print("\nComponent Status:")
        all_ok = True
        for component, status in checks.items():
            status_symbol = "✅" if status else "❌"
            print(f"  {status_symbol} {component}: {status}")
            if not status:
                all_ok = False
        
        if all_ok:
            print("\n✅ All diffusion components initialized successfully!")
            
            # Check if stabilize_and_retrieve method exists
            if hasattr(sm, 'stabilize_and_retrieve'):
                print("✅ stabilize_and_retrieve method exists")
            else:
                print("❌ stabilize_and_retrieve method NOT FOUND")
                all_ok = False
        else:
            print("\n❌ CRITICAL: Some diffusion components failed to initialize!")
            print("   The pipeline will fall back to direct retrieval (like Clean)")
            print("   Check for import errors above.")
        
        return all_ok, sm
        
    except Exception as e:
        print(f"\n❌ ERROR initializing StateManager: {e}")
        traceback.print_exc()
        return False, None

def check_orchestrator():
    """Check if orchestrator uses diffusion components."""
    print("\n" + "="*60)
    print("TEST 2: Orchestrator Diffusion Component Integration")
    print("="*60)
    
    try:
        import asyncio
        
        async def test():
            orchestrator = await create_optimized_marag_pipeline()
            
            # Check state_manager
            if orchestrator.state_manager is None:
                print("❌ Orchestrator has NO state_manager!")
                return False
            
            sm = orchestrator.state_manager
            
            # Check if diffusion components are present
            checks = {
                "State Manager exists": sm is not None,
                "Entropy Tracker": sm.entropy_tracker is not None,
                "Reasoning Flow": sm.reasoning_flow is not None,
                "Regulator Manager": sm.regulator_manager is not None,
            }
            
            print("\nOrchestrator Component Status:")
            all_ok = True
            for component, status in checks.items():
                status_symbol = "✅" if status else "❌"
                print(f"  {status_symbol} {component}: {status}")
                if not status:
                    all_ok = False
            
            if all_ok:
                print("\n✅ Orchestrator has all diffusion components!")
            else:
                print("\n❌ CRITICAL: Orchestrator missing diffusion components!")
                print("   Pipeline will use direct retrieval (fallback mode)")
            
            return all_ok
        
        return asyncio.run(test())
        
    except Exception as e:
        print(f"\n❌ ERROR checking orchestrator: {e}")
        traceback.print_exc()
        return False

def check_imports():
    """Check if all required diffusion components can be imported."""
    print("\n" + "="*60)
    print("TEST 3: Import Check for Diffusion Components")
    print("="*60)
    
    components = [
        ("entropy_tracker", "EntropyTracker"),
        ("reasoning_flow", "ReasoningFlowIndex"),
        ("regulators.regulator_manager", "RegulatorManager"),
        ("regulators.granularity_regulator", "GranularityRegulator"),
        ("regulators.entity_regulator", "EntityRegulator"),
        ("regulators.relation_regulator", "RelationRegulator"),
        ("regulators.evidence_regulator", "EvidenceRegulator"),
        ("regulators.confidence_regulator", "ConfidenceRegulator"),
        ("regulators.plan_regulator", "PlanRegulator"),
    ]
    
    all_ok = True
    for module_path, class_name in components:
        try:
            module = __import__(f"agents.{module_path}", fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {class_name} imported successfully")
        except ImportError as e:
            print(f"  ❌ {class_name} import FAILED: {e}")
            all_ok = False
        except AttributeError as e:
            print(f"  ❌ {class_name} not found in module: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ❌ {class_name} unexpected error: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All diffusion components can be imported!")
    else:
        print("\n❌ CRITICAL: Some components cannot be imported!")
        print("   This will cause silent fallback to direct retrieval")
    
    return all_ok

def check_retrieval_method():
    """Check if stabilize_and_retrieve will use diffusion or fallback."""
    print("\n" + "="*60)
    print("TEST 4: Retrieval Method Check")
    print("="*60)
    
    try:
        sm = StateManager()
        
        # Check the condition that determines fallback
        will_use_diffusion = (
            sm.reasoning_flow is not None and 
            sm.regulator_manager is not None
        )
        
        if will_use_diffusion:
            print("✅ stabilize_and_retrieve will use DIFFUSION-AWARE retrieval")
            print("   - Query stabilization via regulators")
            print("   - Entropy tracking")
            print("   - Reasoning flow updates")
            print("   - Anchor corrections")
        else:
            print("❌ CRITICAL: stabilize_and_retrieve will FALLBACK to direct retrieval")
            print("   - No query stabilization")
            print("   - No entropy tracking")
            print("   - No reasoning flow")
            print("   - Pipeline behaves like Clean (non-diffusion)")
        
        return will_use_diffusion
        
    except Exception as e:
        print(f"\n❌ ERROR checking retrieval method: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks."""
    print("\n" + "="*60)
    print("DIFFUSION PIPELINE DIAGNOSTIC CHECK")
    print("="*60)
    print("\nThis script checks if diffusion-aware components are actually")
    print("being used in the MCP pipeline, or if it's silently falling back")
    print("to direct retrieval (which would explain similar results to Clean).")
    
    results = {}
    
    # Test 1: Import check
    results['imports'] = check_imports()
    
    # Test 2: StateManager initialization
    results['state_manager'], sm = check_state_manager()
    
    # Test 3: Retrieval method
    if sm:
        results['retrieval'] = check_retrieval_method()
    else:
        results['retrieval'] = False
    
    # Test 4: Orchestrator integration
    results['orchestrator'] = check_orchestrator()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("   Diffusion components are properly initialized and will be used.")
        print("   If results are still similar to Clean, the issue is elsewhere.")
    else:
        print("❌ CRITICAL ISSUES FOUND")
        print("   Diffusion components are NOT being used!")
        print("   The pipeline is falling back to direct retrieval (like Clean).")
        print("   This explains why results are similar between MCP and Clean.")
        print("\n   ACTION REQUIRED:")
        print("   1. Check import errors above")
        print("   2. Verify all regulator files exist")
        print("   3. Check for circular import issues")
        print("   4. Verify entropy_tracker.py and reasoning_flow.py exist")
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)