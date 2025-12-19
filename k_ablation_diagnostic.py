"""
K Ablation Diagnostic: Static k-NN vs Adaptive k-NN

This script compares:
- Clean Pipeline (Baseline): static k=10, 15, 20
- Diffusion Pipeline (New): adaptive k

Location: Parent directory to access both Clean and MCP pipelines
"""

import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Get parent directory (where this script is located)
SCRIPT_DIR = Path(__file__).parent.absolute()
CLEAN_DIR = SCRIPT_DIR / "AlgoverseMultiAgent-Clean"
MCP_DIR = SCRIPT_DIR / "AlgoverseMultiAgent-MCP"

# Add both to path
sys.path.insert(0, str(CLEAN_DIR))
sys.path.insert(0, str(MCP_DIR))


def parse_results_csv(csv_path: str) -> Dict[str, float]:
    """Parse results from CSV file created by evaluate_datasets.py."""
    if not os.path.exists(csv_path):
        return {'em': 0.0, 'f1': 0.0, 'latency': 0.0, 'tokens': 0}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # The summary is in multiple rows after "Summary" header
            # Format:
            # Summary,,,,,,,,
            # Exact Match,,,0.5000,,,,,
            # F1 Score,,,0.6486,,,,,
            # Latency/Q (s),,,79.99,,,,,
            # Tokens/Q,,,47713,,,47713,32674,2411
            
            em = 0.0
            f1 = 0.0
            latency = 0.0
            tokens = 0
            
            for row in rows:
                question = row.get('Question', '').strip()
                
                # Look for summary rows
                if 'Exact Match' in question or question == 'Exact Match':
                    em_str = row.get('ExactMatch', '0.0')
                    em = float(em_str) if em_str and em_str.strip() else 0.0
                
                elif 'F1 Score' in question or question == 'F1 Score':
                    f1_str = row.get('F1Score', '0.0')
                    f1 = float(f1_str) if f1_str and f1_str.strip() else 0.0
                
                elif 'Latency/Q' in question or ('Latency' in question and 'Q' in question):
                    latency_str = row.get('Latency/Q (s)', row.get('Latency(s)', '0.0'))
                    latency = float(latency_str) if latency_str and latency_str.strip() else 0.0
                
                elif 'Tokens/Q' in question or ('Tokens' in question and 'Q' in question):
                    tokens_str = row.get('Tokens/Q', '0')
                    # Handle format like "47713,,,47713,32674,2411" - take first number
                    if tokens_str and tokens_str.strip():
                        tokens_parts = tokens_str.split(',')
                        tokens = int(tokens_parts[0]) if tokens_parts[0].strip() else 0
            
            # If we found values, return them
            if em > 0 or f1 > 0:
                return {
                    'em': em,
                    'f1': f1,
                    'latency': latency,
                    'tokens': tokens
                }
            
            # Fallback: try to find summary row with all values
            for row in reversed(rows):
                if row.get('Question', '').strip() in ['Summary', '']:
                    em_str = row.get('ExactMatch', '0.0')
                    f1_str = row.get('F1Score', '0.0')
                    latency_str = row.get('Latency/Q (s)', '0.0')
                    tokens_str = row.get('Tokens/Q', '0')
                    
                    if em_str or f1_str:
                        return {
                            'em': float(em_str) if em_str and em_str.strip() else 0.0,
                            'f1': float(f1_str) if f1_str and f1_str.strip() else 0.0,
                            'latency': float(latency_str) if latency_str and latency_str.strip() else 0.0,
                            'tokens': int(tokens_str.split(',')[0]) if tokens_str and tokens_str.strip() else 0
                        }
                    
    except Exception as e:
        print(f"Error parsing CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'em': 0.0, 'f1': 0.0, 'latency': 0.0, 'tokens': 0}
    
    return {'em': 0.0, 'f1': 0.0, 'latency': 0.0, 'tokens': 0}


async def run_clean_evaluation(k_value: int, num_examples: int = 20) -> Dict[str, Any]:
    """Run Clean pipeline evaluation with specific static k value."""
    print(f"\n{'='*60}")
    print(f"Testing CLEAN pipeline with static k={k_value}")
    print(f"{'='*60}")
    
    # Change to Clean directory
    original_cwd = os.getcwd()
    os.chdir(CLEAN_DIR)
    
    try:
        # Import Clean's evaluation (you'll need to adapt this to your Clean pipeline)
        # For now, we'll use a subprocess approach or direct import
        
        # Option 1: If Clean has evaluate_datasets.py
        clean_eval_path = CLEAN_DIR / "evaluate_datasets.py"
        if clean_eval_path.exists():
            # Set k value via environment variable or modify retriever
            os.environ['FORCE_STATIC_K'] = str(k_value)
            
            # Run evaluation (you'll need to adapt this)
            # For now, return placeholder
            print(f"⚠️  Clean evaluation not yet implemented")
            print(f"    Need to run: python evaluate_datasets.py --dataset musique")
            print(f"    With k={k_value} configured in retriever")
            
            return {
                'pipeline': 'Clean',
                'k_type': 'static',
                'k_value': k_value,
                'em': 0.0,
                'f1': 0.0,
                'latency': 0.0,
                'tokens': 0,
                'note': 'Manual evaluation needed'
            }
        else:
            print(f"⚠️  Clean evaluate_datasets.py not found")
            print(f"    You'll need to run Clean evaluations manually")
            return {
                'pipeline': 'Clean',
                'k_type': 'static',
                'k_value': k_value,
                'em': 0.0,
                'f1': 0.0,
                'latency': 0.0,
                'tokens': 0,
                'note': 'Manual evaluation needed'
            }
    finally:
        os.chdir(original_cwd)


async def run_diffusion_evaluation(num_examples: int = 20) -> Dict[str, Any]:
    """Run Diffusion pipeline evaluation with adaptive k."""
    print(f"\n{'='*60}")
    print(f"Testing DIFFUSION pipeline with adaptive k")
    print(f"{'='*60}")
    
    # Change to MCP directory
    original_cwd = os.getcwd()
    os.chdir(MCP_DIR)
    
    try:
        # Import MCP's evaluation
        from evaluate_datasets import evaluate_dataset
        
        # Run MuSiQue evaluation
        print("Running MuSiQue evaluation with adaptive k...")
        await evaluate_dataset(
            dataset_name="MuSiQue",
            dataset_name_full="allenai/musique-v1",
            dataset_config=None,
            num_examples=num_examples
        )
        
        # Find the latest results CSV
        results_dir = MCP_DIR / "results"
        if results_dir.exists():
            csv_files = sorted(results_dir.glob("musique_results_*.csv"), key=os.path.getmtime, reverse=True)
            if csv_files:
                latest_csv = csv_files[0]
                print(f"Parsing results from: {latest_csv}")
                results = parse_results_csv(str(latest_csv))
                
                return {
                    'pipeline': 'Diffusion',
                    'k_type': 'adaptive',
                    'k_value': 'adaptive',
                    'em': results['em'],
                    'f1': results['f1'],
                    'latency': results['latency'],
                    'tokens': results['tokens'],
                    'results_file': str(latest_csv)
                }
        
        print("⚠️  Results CSV not found")
        return {
            'pipeline': 'Diffusion',
            'k_type': 'adaptive',
            'k_value': 'adaptive',
            'em': 0.0,
            'f1': 0.0,
            'latency': 0.0,
            'tokens': 0
        }
    except Exception as e:
        print(f"Error running Diffusion evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pipeline': 'Diffusion',
            'k_type': 'adaptive',
            'k_value': 'adaptive',
            'em': 0.0,
            'f1': 0.0,
            'latency': 0.0,
            'tokens': 0,
            'error': str(e)
        }
    finally:
        os.chdir(original_cwd)


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print organized comparison table."""
    print("\n" + "="*80)
    print("K ABLATION DIAGNOSTIC RESULTS")
    print("="*80)
    
    # Separate Clean and Diffusion results
    clean_results = [r for r in results if r.get('pipeline') == 'Clean']
    diffusion_results = [r for r in results if r.get('pipeline') == 'Diffusion']
    
    # Print Clean results
    print("\n" + "-"*80)
    print("BASELINE: Clean Pipeline (Static k-NN)")
    print("-"*80)
    print(f"{'k Value':<12} {'Exact Match':<15} {'F1 Score':<15} {'Latency (s)':<15} {'Tokens':<10}")
    print("-"*80)
    
    if clean_results:
        for result in sorted(clean_results, key=lambda x: x.get('k_value', 0)):
            k_val = result.get('k_value', 'N/A')
            em = result.get('em', 0.0)
            f1 = result.get('f1', 0.0)
            latency = result.get('latency', 0.0)
            tokens = result.get('tokens', 0)
            note = result.get('note', '')
            
            if note:
                print(f"{k_val:<12} {em:<15.4f} {f1:<15.4f} {latency:<15.2f} {tokens:<10} ({note})")
            else:
                print(f"{k_val:<12} {em:<15.4f} {f1:<15.4f} {latency:<15.2f} {tokens:<10}")
    else:
        print("No Clean results available")
    
    # Print Diffusion results
    print("\n" + "-"*80)
    print("NEW: Diffusion Pipeline (Adaptive k-NN)")
    print("-"*80)
    print(f"{'k Type':<12} {'Exact Match':<15} {'F1 Score':<15} {'Latency (s)':<15} {'Tokens':<10}")
    print("-"*80)
    
    if diffusion_results:
        for result in diffusion_results:
            k_type = result.get('k_type', 'adaptive')
            em = result.get('em', 0.0)
            f1 = result.get('f1', 0.0)
            latency = result.get('latency', 0.0)
            tokens = result.get('tokens', 0)
            print(f"{k_type:<12} {em:<15.4f} {f1:<15.4f} {latency:<15.2f} {tokens:<10}")
    else:
        print("No Diffusion results available")
    
    # Print comparison
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    
    if clean_results and diffusion_results:
        # Filter out results with notes/errors
        valid_clean = [r for r in clean_results if r.get('em', 0) > 0 and 'note' not in r]
        valid_diffusion = [r for r in diffusion_results if r.get('em', 0) > 0]
        
        if valid_clean and valid_diffusion:
            best_clean_em = max(r['em'] for r in valid_clean)
            best_clean_f1 = max(r['f1'] for r in valid_clean)
            diffusion_em = valid_diffusion[0]['em']
            diffusion_f1 = valid_diffusion[0]['f1']
            
            em_improvement = ((diffusion_em - best_clean_em) / best_clean_em * 100) if best_clean_em > 0 else 0
            f1_improvement = ((diffusion_f1 - best_clean_f1) / best_clean_f1 * 100) if best_clean_f1 > 0 else 0
            
            print(f"\nBest Static k (Clean):     EM={best_clean_em:.4f}, F1={best_clean_f1:.4f}")
            print(f"Adaptive k (Diffusion):    EM={diffusion_em:.4f}, F1={diffusion_f1:.4f}")
            print(f"\nImprovement:               EM +{em_improvement:.1f}%, F1 +{f1_improvement:.1f}%")
            
            if diffusion_em > best_clean_em or diffusion_f1 > best_clean_f1:
                print("\n✅ ADAPTIVE RETRIEVAL OUTPERFORMS STATIC k-NN")
                print("   This demonstrates that adaptive k (based on uncertainty/diffusion)")
                print("   handles scattered documents better than fixed k values.")
            else:
                print("\n⚠️  Results are similar - may need more examples")
        else:
            print("\n⚠️  Need valid results from both pipelines to compare")


async def run_full_diagnostic(num_examples: int = 20, skip_clean: bool = False):
    """Run complete diagnostic comparing static vs adaptive k."""
    print("\n" + "="*80)
    print("K ABLATION DIAGNOSTIC: Static k-NN vs Adaptive k-NN")
    print("="*80)
    print("\nThis diagnostic demonstrates:")
    print("  1. Static k-NN cannot fix scatter by simply increasing k (Clean pipeline)")
    print("  2. Adaptive retrieval (diffusion-aware) can handle scattered documents (Diffusion pipeline)")
    print("\n" + "="*80)
    
    results = []
    
    # Test Clean with different static k values
    if not skip_clean:
        static_k_values = [10, 15, 20]
        
        print("\n" + "="*80)
        print("PHASE 1: Testing CLEAN pipeline (Baseline)")
        print("="*80)
        print("\nNOTE: Clean pipeline evaluation needs to be run manually")
        print("      with different k values configured in the retriever.")
        print("      This script will organize results once they're available.")
        
        for k in static_k_values:
            result = await run_clean_evaluation(k, num_examples)
            results.append(result)
    else:
        print("\nSkipping Clean pipeline tests (--skip-clean flag)")
    
    # Test Diffusion with adaptive k
    print("\n" + "="*80)
    print("PHASE 2: Testing DIFFUSION pipeline (New Method)")
    print("="*80)
    
    diffusion_result = await run_diffusion_evaluation(num_examples)
    results.append(diffusion_result)
    
    if diffusion_result.get('em', 0) > 0:
        print(f"✅ Diffusion adaptive: EM={diffusion_result['em']:.4f}, F1={diffusion_result['f1']:.4f}")
    else:
        print("⚠️  Diffusion evaluation may have failed - check logs above")
    
    # Print organized comparison
    print_comparison_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"k_ablation_diagnostic_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_examples': num_examples,
            'results': results,
            'summary': {
                'clean_results': [r for r in results if r.get('pipeline') == 'Clean'],
                'diffusion_results': [r for r in results if r.get('pipeline') == 'Diffusion']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="K Ablation Diagnostic: Compare static k-NN (Clean) vs adaptive k-NN (Diffusion)"
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=20, 
        help="Number of examples to test (default: 20)"
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip Clean pipeline tests (only test Diffusion)"
    )
    args = parser.parse_args()
    
    asyncio.run(run_full_diagnostic(
        num_examples=args.num_examples,
        skip_clean=args.skip_clean
    ))

