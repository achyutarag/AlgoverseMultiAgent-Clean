"""
Clean Pipeline Debugging System

Captures reasoning traces for the Clean (baseline) MA-RAG pipeline.
Generates comparable reports to debug_diffusion_pipeline.py for MCP.

Usage:
    python debug_clean_pipeline.py --dataset musique --num_examples 10
"""

import asyncio
import json
import csv
import logging
import os
import sys
import re
import string
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Evaluation functions (copied from evaluate_datasets.py)
def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


class CleanTrace:
    """Simple trace for Clean pipeline (no regulators/entropy)."""
    
    def __init__(self, question: str, ground_truth: str):
        self.question = question
        self.ground_truth = ground_truth
        self.hops: List[Dict[str, Any]] = []
        self.prediction: str = ""
        self.em: float = 0.0
        self.f1: float = 0.0
    
    def add_hop(self, hop_data: Dict[str, Any]):
        """Add a hop to the trace."""
        self.hops.append(hop_data)
    
    @property
    def total_hops(self) -> int:
        return len(self.hops)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'ground_truth': self.ground_truth,
            'prediction': self.prediction,
            'em': self.em,
            'f1': self.f1,
            'total_hops': self.total_hops,
            'hops': self.hops
        }


class CleanDebugger:
    """Captures and analyzes Clean pipeline reasoning traces."""
    
    def __init__(self):
        self.traces: List[CleanTrace] = []
        self.current_trace: Optional[CleanTrace] = None
    
    def start_trace(self, question: str, ground_truth: str):
        """Start tracing a new question."""
        self.current_trace = CleanTrace(question, ground_truth)
        logger.info(f"🔍 Starting trace for: {question[:60]}...")
    
    def capture_hop(
        self,
        hop: int,
        step_id: str,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        extracted_passages: List[str],
        qa_answer: str,
        qa_confidence: float,
        step_metadata: Optional[Dict[str, Any]] = None
    ):
        """Capture a complete hop trace."""
        if not self.current_trace:
            return
        
        hop_data = {
            'hop': hop,
            'step_id': step_id,
            'query': query,
            'retrieval': {
                'documents_retrieved': len(retrieved_docs),
                'doc_ids': [d.get('id', d.get('source', 'unknown')) for d in retrieved_docs[:5]]  # Top 5
            },
            'extraction': {
                'passages_extracted': len(extracted_passages),
                'passage_count': sum(len(p.split()) for p in extracted_passages)  # Word count
            },
            'qa': {
                'answer': qa_answer,
                'confidence': qa_confidence
            },
            'metadata': step_metadata or {}
        }
        
        self.current_trace.add_hop(hop_data)
    
    def finish_trace(self, prediction: str, em: float, f1: float):
        """Finish tracing a question."""
        if self.current_trace:
            self.current_trace.prediction = prediction
            self.current_trace.em = em
            self.current_trace.f1 = f1
            self.traces.append(self.current_trace)
            logger.info(f"✅ Finished trace: EM={em:.2f}, F1={f1:.2f}")
            self.current_trace = None
    
    def generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report."""
        report = {
            'summary': {
                'total_questions': len(self.traces),
                'avg_em': sum(t.em for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'avg_f1': sum(t.f1 for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'avg_hops': sum(t.total_hops for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'pipeline': 'Clean',
                'note': 'Baseline pipeline without regulators/entropy tracking'
            },
            'statistics': {
                'regulator_firings': {},  # Empty - Clean has no regulators
                'anchor_rejections': {},  # Empty - Clean has no anchors
                'convergence_checks': 0   # Empty - Clean has no convergence gate
            },
            'failure_modes': self._identify_failure_modes(),
            'traces': [t.to_dict() for t in self.traces]
        }
        
        # Calculate additional metrics for Google Sheets
        metrics = self._calculate_additional_metrics()
        report["summary"].update(metrics)
        
        return report
    
    def _calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional metrics for tracking: confidence mean/variance, etc."""
        confidences = []
        docs_per_hop = []
        early_terminations = 0
        notes = []
        
        for trace in self.traces:
            trace_early_term = False
            
            for hop in trace.hops:
                # Confidence
                conf = hop.get('qa', {}).get('confidence', 0.5)
                confidences.append(conf)
                
                # Docs per hop
                docs = hop.get('retrieval', {}).get('documents_retrieved', 0)
                docs_per_hop.append(docs)
            
            # Early termination: questions with < 2 hops or very low average confidence
            trace_confs = [hop.get('qa', {}).get('confidence', 0.5) for hop in trace.hops]
            if trace.total_hops < 2 or (trace_confs and statistics.mean(trace_confs) < 0.4):
                early_terminations += 1
            
            # Qualitative notes
            if trace.em == 0.0 and trace.f1 < 0.3:
                notes.append(f"Low performance: {trace.question[:50]}... (F1={trace.f1:.2f})")
            elif trace.total_hops > 4:
                notes.append(f"Long reasoning: {trace.question[:50]}... ({trace.total_hops} hops)")
        
        # Calculate metrics
        conf_mean = statistics.mean(confidences) if confidences else 0.5
        conf_var = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        
        early_term_rate = early_terminations / len(self.traces) if self.traces else 0.0
        avg_docs_per_hop = statistics.mean(docs_per_hop) if docs_per_hop else 0.0
        
        return {
            'confidence_mean': conf_mean,
            'confidence_variance': conf_var,
            'granularity_error_rate': None,  # N/A for Clean pipeline
            'avg_docs_per_hop': avg_docs_per_hop,
            'early_termination_rate': early_term_rate,
            'notes': notes[:5]  # Top 5 notes
        }
    
    def _identify_failure_modes(self) -> Dict[str, Any]:
        """Identify common failure modes from traces."""
        failures = {
            'low_confidence_issues': [],
            'short_horizon_issues': [],
            'retrieval_issues': [],
            'extraction_issues': []
        }
        
        for trace in self.traces:
            question = trace.question
            f1 = trace.f1
            em = trace.em
            
            # Short horizon (too few steps)
            if trace.total_hops < 2:
                failures['short_horizon_issues'].append({
                    'question': question,
                    'hops': trace.total_hops,
                    'f1': f1,
                    'em': em
                })
            
            # Low confidence across hops
            low_conf_count = 0
            for hop in trace.hops:
                conf = hop.get('qa', {}).get('confidence', 0.5)
                if conf < 0.5:
                    low_conf_count += 1
            
            if low_conf_count > 0:
                failures['low_confidence_issues'].append({
                    'question': question,
                    'low_conf_hops': low_conf_count,
                    'total_hops': trace.total_hops,
                    'f1': f1,
                    'em': em
                })
            
            # Retrieval issues (no documents retrieved)
            for hop in trace.hops:
                docs_retrieved = hop.get('retrieval', {}).get('documents_retrieved', 0)
                if docs_retrieved == 0:
                    failures['retrieval_issues'].append({
                        'question': question,
                        'hop': hop.get('hop', 0),
                        'f1': f1,
                        'em': em
                    })
                    break
        
        return failures
    
    def print_failure_analysis(self):
        """Print failure mode analysis."""
        failures = self._identify_failure_modes()
        
        print("\n" + "="*80)
        print("FAILURE MODE ANALYSIS")
        print("="*80)
        
        if failures['short_horizon_issues']:
            print(f"\n⚠️  Short Horizon Issues: {len(failures['short_horizon_issues'])}")
            for issue in failures['short_horizon_issues'][:3]:
                print(f"   - {issue['question'][:60]}... (hops={issue['hops']}, F1={issue['f1']:.2f})")
        
        if failures['low_confidence_issues']:
            print(f"\n⚠️  Low Confidence Issues: {len(failures['low_confidence_issues'])}")
            for issue in failures['low_confidence_issues'][:3]:
                print(f"   - {issue['question'][:60]}... (F1={issue['f1']:.2f})")
        
        if failures['retrieval_issues']:
            print(f"\n⚠️  Retrieval Issues: {len(failures['retrieval_issues'])}")
            for issue in failures['retrieval_issues'][:3]:
                print(f"   - {issue['question'][:60]}... (hop={issue['hop']}, F1={issue['f1']:.2f})")
        
        if not any(failures.values()):
            print("\n✅ No major failure modes detected")


async def debug_evaluate_dataset(dataset_name: str, num_examples: int = 10):
    """Run evaluation with debugging traces for Clean pipeline."""
    print("\n" + "="*80)
    print("CLEAN PIPELINE DEBUGGING MODE")
    print("="*80)
    print(f"\nEvaluating {dataset_name} with {num_examples} examples")
    print("Capturing reasoning traces...")
    
    # Load dataset
    from datasets import load_dataset
    from agents.musique_document_loader import _load_musique_from_github, load_musique_example_context_as_documents
    
    start_idx = 0
    end_idx = num_examples
    
    try:
        if dataset_name.lower() == "musique":
            dataset_name_full = "allenai/musique-v1"  # ✅ FIX: Correct HuggingFace dataset name
            dataset_config = None
        else:
            dataset_name_full = dataset_name
            dataset_config = None
        
        ds = load_dataset(dataset_name_full, dataset_config)
        examples = ds["validation"]
    except Exception as e:
        if dataset_name.lower() == "musique":
            print(f"⚠️  HuggingFace load failed: {e}")
            print("Attempting to load from GitHub repository...")
            try:
                raw = _load_musique_from_github("validation")
                
                class SimpleDataset:
                    def __init__(self, data):
                        self.data = data
                    
                    def select(self, idxs):
                        return [self.data[i] for i in idxs]
                
                examples = SimpleDataset(raw)
            except Exception as e2:
                raise RuntimeError(f"Failed to load MuSiQue from GitHub: {e2}")
        else:
            raise
    
    # Apply slicing
    if hasattr(examples, "select"):
        eval_examples = examples.select(range(start_idx, end_idx))
    else:
        eval_examples = examples[start_idx:end_idx]
    
    num_examples = len(eval_examples)
    
    # Import Clean pipeline (local pipeline.py in Clean directory)
    from pipeline import MARAGPipeline
    
    debugger = CleanDebugger()
    
    # Evaluate each example
    for i, example in enumerate(eval_examples, 1):
        question = example.get('question', '')
        ground_truth = example.get('answer', '')
        
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(eval_examples)}: {question[:60]}...")
        print(f"{'='*80}")
        
        # Create new pipeline instance for each example
        pipeline = MARAGPipeline()
        
        # Start trace
        debugger.start_trace(question, ground_truth)
        
        try:
            # Load documents
            documents = load_musique_example_context_as_documents(example)
            pipeline.add_documents(documents)  # ✅ Use pipeline method (matches MCP pattern)
            
            # Execute query
            result = await pipeline.query(question, max_steps=5)
            
            # Extract prediction
            prediction = result.final_answer if hasattr(result, 'final_answer') else str(result)
            
            # Capture hops from reasoning_trajectory
            if hasattr(result, 'reasoning_trajectory') and result.reasoning_trajectory:
                for hop_idx, step_data in enumerate(result.reasoning_trajectory, 1):
                    step_id = step_data.get('step_id', f'step_{hop_idx}')
                    query = step_data.get('query', question)
                    retrieved_docs = step_data.get('retrieved_documents', [])
                    extracted_passages = step_data.get('extracted_passages', [])
                    qa_answer = step_data.get('answer', '')
                    qa_confidence = step_data.get('confidence', 0.5)
                    
                    debugger.capture_hop(
                        hop=hop_idx,
                        step_id=step_id,
                        query=query,
                        retrieved_docs=retrieved_docs,
                        extracted_passages=extracted_passages,
                        qa_answer=qa_answer,
                        qa_confidence=qa_confidence,
                        step_metadata=step_data.get('metadata', {})
                    )
            else:
                # Fallback: create single hop from result
                retrieved_docs = result.sources if hasattr(result, 'sources') else []
                debugger.capture_hop(
                    hop=1,
                    step_id='single_step',
                    query=question,
                    retrieved_docs=retrieved_docs,
                    extracted_passages=[],
                    qa_answer=prediction,
                    qa_confidence=0.5,
                    step_metadata={}
                )
            
            # Calculate scores
            em = exact_match_score(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)
            
            # Finish trace
            debugger.finish_trace(prediction, em, f1)
            
            print(f"\nResult: EM={em:.2f}, F1={f1:.2f}")
            print(f"Prediction: {prediction[:100]}...")
            print(f"Ground Truth: {ground_truth}")
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
            debugger.finish_trace("ERROR", 0.0, 0.0)
    
    # Generate and save report
    report = debugger.generate_debug_report()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    report_file = results_dir / f"clean_debug_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Export to CSV for Google Sheets
    csv_file = results_dir / f"clean_results_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Question', 'Prediction', 'Ground_Truth', 'ExactMatch', 'F1Score', 'Hops'])
        
        # Data rows
        for trace in report.get('traces', []):
            question = trace.get('question', '').replace('\n', ' ').strip()
            prediction = trace.get('prediction', '').replace('\n', ' ').strip()
            ground_truth = trace.get('ground_truth', '').replace('\n', ' ').strip()
            em = trace.get('em', 0.0)
            f1 = trace.get('f1', 0.0)
            hops = trace.get('total_hops', 0)
            
            writer.writerow([question, prediction, ground_truth, f"{em:.4f}", f"{f1:.4f}", hops])
        
        # Summary row
        writer.writerow([])
        writer.writerow(['Summary', '', '', '', '', ''])
        writer.writerow(['Exact Match', '', '', f"{report['summary']['avg_em']:.4f}", '', ''])
        writer.writerow(['F1 Score', '', '', f"{report['summary']['avg_f1']:.4f}", '', ''])
        writer.writerow(['Avg Hops', '', '', f"{report['summary']['avg_hops']:.2f}", '', ''])
        writer.writerow(['Examples', '', '', f"{report['summary']['total_questions']}", '', ''])
    
    print("\n" + "="*80)
    print("DEBUG REPORT SUMMARY")
    print("="*80)
    print(f"\nTotal Questions: {report['summary']['total_questions']}")
    print(f"Average EM: {report['summary']['avg_em']:.4f}")
    print(f"Average F1: {report['summary']['avg_f1']:.4f}")
    print(f"Average Hops: {report['summary']['avg_hops']:.2f}")
    print(f"Pipeline: {report['summary']['pipeline']}")
    
    # Print additional metrics for Google Sheets
    print("\n" + "="*80)
    print("ADDITIONAL METRICS (for Google Sheets)")
    print("="*80)
    print(f"Confidence Mean: {report['summary'].get('confidence_mean', 0.0):.4f}")
    print(f"Confidence Variance: {report['summary'].get('confidence_variance', 0.0):.4f}")
    gran_error = report['summary'].get('granularity_error_rate')
    if gran_error is not None:
        print(f"Granularity Error Rate: {gran_error:.4f} ({gran_error*100:.1f}%)")
    else:
        print(f"Granularity Error Rate: N/A (Clean pipeline)")
    print(f"Avg Docs per Hop: {report['summary'].get('avg_docs_per_hop', 0.0):.2f}")
    print(f"Early Termination Rate: {report['summary'].get('early_termination_rate', 0.0):.4f} ({report['summary'].get('early_termination_rate', 0.0)*100:.1f}%)")
    print(f"\nQualitative Notes:")
    notes = report['summary'].get('notes', [])
    if notes:
        for note in notes:
            print(f"  - {note}")
    else:
        print("  - No significant notes")
    
    # Print failure analysis
    debugger.print_failure_analysis()
    
    print(f"\n📄 Full debug report saved to: {report_file}")
    print(f"📊 CSV results saved to: {csv_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Debug Clean Pipeline")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset to evaluate")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to debug")
    args = parser.parse_args()
    
    asyncio.run(debug_evaluate_dataset(args.dataset, args.num_examples))

