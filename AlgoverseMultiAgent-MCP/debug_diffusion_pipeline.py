"""
Diffusion Pipeline Debugging System

Captures comprehensive reasoning traces to identify failure modes:
1. Regulator decisions (too conservative/aggressive?)
2. Granularity regulator firing
3. Adaptive retrieval weighting
4. Drift logic and entropy thresholds
5. Convergence issues
6. Query rewriting issues

Usage:
    python debug_diffusion_pipeline.py --dataset musique --num_examples 5
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_datasets import (
    exact_match_score,
    f1_score,
    normalize_text
)

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more details
    format='%(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ReasoningTrace:
    """Captures full reasoning trace for a single question."""
    
    def __init__(self, question: str, ground_truth: str):
        self.question = question
        self.ground_truth = ground_truth
        self.prediction = None
        self.em = 0.0
        self.f1 = 0.0
        
        # Per-hop traces
        self.hops: List[Dict[str, Any]] = []
        
        # Summary metrics
        self.total_hops = 0
        self.total_retrievals = 0
        self.total_anchors_created = 0
        self.total_anchors_rejected = 0
        self.early_terminated = False
        self.termination_reason = None
        
    def add_hop(self, hop_data: Dict[str, Any]):
        """Add a hop trace."""
        self.hops.append(hop_data)
        self.total_hops = len(self.hops)
        self.total_retrievals += hop_data.get('retrieval', {}).get('retrieval_count', 0)
        self.total_anchors_created += len(hop_data.get('anchors', {}).get('created', []))
        self.total_anchors_rejected += len(hop_data.get('anchors', {}).get('rejected', []))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'question': self.question,
            'ground_truth': self.ground_truth,
            'prediction': self.prediction,
            'em': self.em,
            'f1': self.f1,
            'total_hops': self.total_hops,
            'total_retrievals': self.total_retrievals,
            'total_anchors_created': self.total_anchors_created,
            'total_anchors_rejected': self.total_anchors_rejected,
            'early_terminated': self.early_terminated,
            'termination_reason': self.termination_reason,
            'hops': self.hops
        }


class DiffusionDebugger:
    """Captures and analyzes diffusion pipeline reasoning traces."""
    
    def __init__(self):
        self.traces: List[ReasoningTrace] = []
        self.current_trace: Optional[ReasoningTrace] = None
        
        # Statistics
        self.stats = {
            'regulator_firings': defaultdict(int),
            'anchor_rejections': defaultdict(int),
            'convergence_checks': [],
            'query_transformations': []
        }
    
    def start_trace(self, question: str, ground_truth: str):
        """Start tracing a new question."""
        self.current_trace = ReasoningTrace(question, ground_truth)
        logger.info(f"🔍 Starting trace for: {question[:60]}...")
    
    def capture_hop(
        self,
        hop: int,
        step_id: str,
        proposed_query: str,
        stabilized_query: str,
        regulator_decisions: List[Dict[str, Any]],
        flow_snapshot: Dict[str, Any],
        retrieval_params: Dict[str, Any],
        retrieval_results: Dict[str, Any],
        anchor_decisions: List[Dict[str, Any]],
        qa_answer: str,
        qa_confidence: float,
        convergence_check: Optional[Dict[str, Any]] = None,
        debug_metadata: Optional[Dict[str, Any]] = None
    ):
        """Capture a complete hop trace."""
        if not self.current_trace:
            return
        
        hop_data = {
            'hop': hop,
            'step_id': step_id,
            'queries': {
                'proposed': proposed_query,
                'stabilized': stabilized_query,
                'transformation': self._analyze_query_transformation(proposed_query, stabilized_query)
            },
            # ✅ NEW: Persist per-hop debug metadata (e.g., query repair events)
            'debug_metadata': debug_metadata or {},
            'regulators': {
                'decisions': regulator_decisions,
                'summary': self._summarize_regulator_decisions(regulator_decisions)
            },
            'flow_state': {
                'entropy': flow_snapshot.get('entropy', 0.0),
                'diffusion': flow_snapshot.get('diffusion_coefficient', 0.0),
                'confidence': flow_snapshot.get('confidence', 0.0),
                'beliefs': flow_snapshot.get('beliefs', {}),
                'anchors_count': len(flow_snapshot.get('anchors', [])),
                'plan_alignment': flow_snapshot.get('plan_alignment', 0.0),
                'drift': flow_snapshot.get('drift_from_previous', 0.0)
            },
            'retrieval': {
                'k_requested': retrieval_params.get('k', 0),
                # ✅ FIX: k_actual should reflect the retriever's actual search k (not len(docs))
                'k_actual': retrieval_results.get('k_actual', retrieval_results.get('k_used', 0)),
                'min_similarity': retrieval_params.get('min_similarity', 0.0),
                'uncertainty': retrieval_params.get('total_uncertainty', 0.0),
                'entropy_penalty': retrieval_params.get('entropy_penalty', 0.0),
                'diffusion_penalty': retrieval_params.get('diffusion_penalty', 0.0),
                'documents_retrieved': retrieval_results.get('doc_count', 0),
                'top_similarity': retrieval_results.get('top_similarity', 0.0),
                'avg_similarity': retrieval_results.get('avg_similarity', 0.0),
                'adaptive_boost': retrieval_results.get('adaptive_boost', False),
                'retrieval_count': 1
            },
            'anchors': {
                'created': [a for a in anchor_decisions if a.get('action') == 'created'],
                'rejected': [a for a in anchor_decisions if a.get('action') == 'rejected'],
                'reasons': [a.get('reason') for a in anchor_decisions if a.get('action') == 'rejected']
            },
            'qa': {
                'answer': qa_answer,
                'confidence': qa_confidence
            },
            'convergence': convergence_check
        }
        
        self.current_trace.add_hop(hop_data)
        
        # Update statistics
        for reg_decision in regulator_decisions:
            reg_name = reg_decision.get('regulator', 'unknown')
            self.stats['regulator_firings'][reg_name] += 1
        
        for anchor in anchor_decisions:
            if anchor.get('action') == 'rejected':
                reason = anchor.get('reason', 'unknown')
                self.stats['anchor_rejections'][reason] += 1
        
        if convergence_check:
            self.stats['convergence_checks'].append(convergence_check)
    
    def _analyze_query_transformation(self, original: str, stabilized: str) -> Dict[str, Any]:
        """Analyze how the query was transformed."""
        original_words = set(original.lower().split())
        stabilized_words = set(stabilized.lower().split())
        
        added = stabilized_words - original_words
        removed = original_words - stabilized_words
        
        return {
            'added_terms': list(added),
            'removed_terms': list(removed),
            'transformation_ratio': len(added) / len(original_words) if original_words else 0.0,
            'is_significant': len(added) > 3 or len(removed) > 2,
            'terms_added_count': len(added),
            'terms_removed_count': len(removed)
        }
    
    def _summarize_regulator_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize regulator decisions for this hop."""
        if not decisions:
            return {
                'total_regulators': 0,
                'regulators_fired': [],
                'total_weight': 0.0,
                'avg_weight': 0.0,
                'granularity_fired': False,
                'entity_fired': False,
                'evidence_fired': False
            }
        
        summary = {
            'total_regulators': len(decisions),
            'regulators_fired': [d.get('regulator', 'unknown') for d in decisions],
            'total_weight': sum(d.get('weight', 0.0) for d in decisions),
            'avg_weight': sum(d.get('weight', 0.0) for d in decisions) / len(decisions),
            'granularity_fired': any('Granularity' in str(d.get('regulator', '')) for d in decisions),
            'entity_fired': any('Entity' in str(d.get('regulator', '')) for d in decisions),
            'evidence_fired': any('Evidence' in str(d.get('regulator', '')) for d in decisions),
            'relation_fired': any('Relation' in str(d.get('regulator', '')) for d in decisions),
            'confidence_fired': any('Confidence' in str(d.get('regulator', '')) for d in decisions),
            'plan_fired': any('Plan' in str(d.get('regulator', '')) for d in decisions)
        }
        return summary
    
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
        return {
            'summary': {
                'total_questions': len(self.traces),
                'avg_em': sum(t.em for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'avg_f1': sum(t.f1 for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'avg_hops': sum(t.total_hops for t in self.traces) / len(self.traces) if self.traces else 0.0,
                'total_anchors_created': sum(t.total_anchors_created for t in self.traces),
                'total_anchors_rejected': sum(t.total_anchors_rejected for t in self.traces)
            },
            'statistics': {
                'regulator_firings': dict(self.stats['regulator_firings']),
                'anchor_rejections': dict(self.stats['anchor_rejections']),
                'convergence_checks': len(self.stats['convergence_checks'])
            },
            'failure_modes': self._identify_failure_modes(),
            'traces': [t.to_dict() for t in self.traces]
        }
    
    def _identify_failure_modes(self) -> Dict[str, Any]:
        """Identify common failure modes from traces."""
        failures = {
            'low_entropy_issues': [],
            'high_diffusion_issues': [],
            'anchor_rejection_issues': [],
            'convergence_issues': [],
            'query_transformation_issues': [],
            'retrieval_issues': [],
            'regulator_issues': []
        }
        
        for trace in self.traces:
            if trace.em == 0.0:  # Failed question
                for hop in trace.hops:
                    hop_num = hop.get('hop', 0)
                    
                    # Check entropy
                    entropy = hop.get('flow_state', {}).get('entropy', 0.0)
                    if entropy < 0.1:
                        failures['low_entropy_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'entropy': entropy,
                            'note': 'Entropy too low - may not trigger adaptive retrieval'
                        })
                    
                    # Check diffusion
                    diffusion = hop.get('flow_state', {}).get('diffusion', 0.0)
                    if diffusion > 0.1:
                        failures['high_diffusion_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'diffusion': diffusion,
                            'note': 'High diffusion - query may be unstable'
                        })
                    
                    # Check anchor rejections
                    rejected = hop.get('anchors', {}).get('rejected', [])
                    if rejected:
                        failures['anchor_rejection_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'rejections': len(rejected),
                            'reasons': [r.get('reason', 'unknown') for r in rejected]
                        })
                    
                    # Check query transformation
                    transform = hop.get('queries', {}).get('transformation', {})
                    if transform.get('is_significant'):
                        failures['query_transformation_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'added_terms': transform.get('added_terms', []),
                            'transformation_ratio': transform.get('transformation_ratio', 0.0),
                            'note': 'Query significantly transformed - may indicate regulator issues'
                        })
                    
                    # Check retrieval
                    retrieval = hop.get('retrieval', {})
                    docs_retrieved = retrieval.get('documents_retrieved', 0)
                    if docs_retrieved < 5:
                        failures['retrieval_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'docs_retrieved': docs_retrieved,
                            'k_actual': retrieval.get('k_actual', 0),
                            'note': 'Too few documents retrieved'
                        })
                    
                    # Check regulators
                    reg_summary = hop.get('regulators', {}).get('summary', {})
                    if not reg_summary.get('granularity_fired'):
                        failures['regulator_issues'].append({
                            'question': trace.question[:60],
                            'hop': hop_num,
                            'issue': 'Granularity regulator not fired',
                            'note': 'Initial condition regulator should always fire'
                        })
        
        return failures
    
    def print_failure_analysis(self):
        """Print analysis of failure modes."""
        report = self.generate_debug_report()
        failures = report['failure_modes']
        
        print("\n" + "="*80)
        print("FAILURE MODE ANALYSIS")
        print("="*80)
        
        # Low entropy issues
        if failures['low_entropy_issues']:
            print(f"\n⚠️  LOW ENTROPY ISSUES: {len(failures['low_entropy_issues'])} instances")
            print("   Entropy < 0.1 may prevent adaptive retrieval from activating")
            for issue in failures['low_entropy_issues'][:5]:
                print(f"   - Hop {issue['hop']}: H(t)={issue['entropy']:.3f} - {issue['note']}")
        
        # High diffusion issues
        if failures['high_diffusion_issues']:
            print(f"\n⚠️  HIGH DIFFUSION ISSUES: {len(failures['high_diffusion_issues'])} instances")
            print("   High diffusion (D(t) > 0.1) indicates query instability")
            for issue in failures['high_diffusion_issues'][:5]:
                print(f"   - Hop {issue['hop']}: D(t)={issue['diffusion']:.3f} - {issue['note']}")
        
        # Anchor rejection issues
        if failures['anchor_rejection_issues']:
            print(f"\n⚠️  ANCHOR REJECTION ISSUES: {len(failures['anchor_rejection_issues'])} instances")
            print("   Anchors being rejected may indicate quality threshold too strict")
            for issue in failures['anchor_rejection_issues'][:5]:
                print(f"   - Hop {issue['hop']}: {issue['rejections']} rejections - Reasons: {issue['reasons']}")
        
        # Query transformation issues
        if failures['query_transformation_issues']:
            print(f"\n⚠️  QUERY TRANSFORMATION ISSUES: {len(failures['query_transformation_issues'])} instances")
            print("   Significant query transformations may indicate regulator over-aggression")
            for issue in failures['query_transformation_issues'][:5]:
                print(f"   - Hop {issue['hop']}: Added {len(issue['added_terms'])} terms - {issue['note']}")
        
        # Retrieval issues
        if failures['retrieval_issues']:
            print(f"\n⚠️  RETRIEVAL ISSUES: {len(failures['retrieval_issues'])} instances")
            print("   Too few documents retrieved may indicate k too low or similarity threshold too high")
            for issue in failures['retrieval_issues'][:5]:
                print(f"   - Hop {issue['hop']}: {issue['docs_retrieved']} docs (k={issue['k_actual']}) - {issue['note']}")
        
        # Regulator issues
        if failures['regulator_issues']:
            print(f"\n⚠️  REGULATOR ISSUES: {len(failures['regulator_issues'])} instances")
            for issue in failures['regulator_issues'][:5]:
                print(f"   - Hop {issue['hop']}: {issue['issue']} - {issue['note']}")
        
        if not any(failures.values()):
            print("\n✅ No obvious failure modes detected in failed questions")


# Global debugger instance
debugger = DiffusionDebugger()


# Global capture state
_capture_state = {
    'current_hop': 0,
    'current_step_id': '',
    'proposed_query': '',
    'stabilized_query': '',
    'regulator_decisions': [],
    'flow_snapshot': {},
    'retrieval_params': {},
    'retrieval_results': {},
    'anchor_decisions': [],
    'qa_answer': '',
    'qa_confidence': 0.0,
    # ✅ NEW: Per-call debug metadata from stabilize_and_retrieve (e.g., query repair)
    'debug_metadata': {}
}


def patch_state_manager_for_debugging(state_manager):
    """Patch state_manager to capture stabilize_and_retrieve calls."""
    from agents.state_manager.retrieval import stabilize_and_retrieve
    import types
    
    original_stabilize = state_manager.stabilize_and_retrieve
    
    async def debug_stabilize_and_retrieve(proposed_query, hop, previous_answers, plan_goal=None, retriever_agent=None, current_step_index=None, total_steps=None):
        """Wrapped stabilize_and_retrieve that captures traces."""
        # Store proposed query
        _capture_state['proposed_query'] = proposed_query
        _capture_state['current_hop'] = hop
        
        # Call original
        result = await original_stabilize(proposed_query, hop, previous_answers, plan_goal, retriever_agent, current_step_index, total_steps)
        
        # Capture stabilized query
        _capture_state['stabilized_query'] = result.get("stabilized_query", proposed_query)

        # ✅ NEW: Capture query repair / other debug metadata from StateManager
        dm = result.get("debug_metadata", {}) if isinstance(result, dict) else {}
        _capture_state['debug_metadata'] = dm if isinstance(dm, dict) else {}
        
        # Capture constraints (regulator decisions)
        constraints = result.get("constraints", [])
        regulator_decisions = []
        for constraint in constraints:
            if hasattr(constraint, 'dict'):
                constraint_dict = constraint.dict()
            elif isinstance(constraint, dict):
                constraint_dict = constraint
            else:
                continue
            
            regulator_decisions.append({
                'regulator': constraint_dict.get('regulator_name', 'unknown'),
                'type': constraint_dict.get('constraint_type', 'unknown'),
                'weight': constraint_dict.get('weight', 0.0),
                'parameters': constraint_dict.get('parameters', {})
            })
        _capture_state['regulator_decisions'] = regulator_decisions
        
        # Capture flow snapshot
        flow_snapshot = result.get("flow_snapshot", {})
        if hasattr(flow_snapshot, 'dict'):
            flow_snapshot = flow_snapshot.dict()
        elif not isinstance(flow_snapshot, dict):
            flow_snapshot = {}
        _capture_state['flow_snapshot'] = flow_snapshot
        
        # Capture retrieval info
        docs = result.get("documents", [])
        _capture_state['retrieval_results'] = {
            'k_used': len(docs),
            'doc_count': len(docs),
            'top_similarity': 0.0,  # Will be filled from retriever if available
            'avg_similarity': 0.0,
            'adaptive_boost': result.get("adaptive_boost", False)
        }
        
        return result
    
    state_manager.stabilize_and_retrieve = debug_stabilize_and_retrieve
    
    return state_manager


def patch_retriever_for_debugging(retriever):
    """Patch retriever to capture retrieval parameters."""
    import types
    
    original_process = retriever.process
    
    async def debug_process(input_data):
        """Wrapped process that captures retrieval parameters."""
        # Extract retrieval parameters
        query = input_data.get("query", "")
        k = input_data.get("k", retriever.top_k)
        min_sim = input_data.get("min_similarity", retriever.min_similarity)
        entropy_penalty = input_data.get("entropy_penalty", 0.0)
        diffusion_penalty = input_data.get("diffusion_penalty", 0.0)
        
        # Calculate total uncertainty
        total_uncertainty = entropy_penalty + diffusion_penalty
        
        _capture_state['retrieval_params'] = {
            'k': k,
            'min_similarity': min_sim,
            'total_uncertainty': total_uncertainty,
            'entropy_penalty': entropy_penalty,
            'diffusion_penalty': diffusion_penalty
        }
        
        # Call original
        result = await original_process(input_data)
        
        # Capture retrieval results
        docs = result.metadata.get("documents", [])

        # ✅ NEW: Capture retriever-reported actual parameters (if available)
        rp = result.metadata.get("retrieval_parameters", {}) or {}
        try:
            _capture_state['retrieval_results']['k_actual'] = int(rp.get("k", k))
        except Exception:
            _capture_state['retrieval_results']['k_actual'] = k
        try:
            _capture_state['retrieval_results']['min_similarity_actual'] = float(rp.get("min_similarity", min_sim))
        except Exception:
            _capture_state['retrieval_results']['min_similarity_actual'] = min_sim

        if docs:
            similarities = [doc.get("similarity", 0.0) for doc in docs if "similarity" in doc]
            if similarities:
                _capture_state['retrieval_results']['top_similarity'] = max(similarities)
                _capture_state['retrieval_results']['avg_similarity'] = sum(similarities) / len(similarities)
        
        _capture_state['retrieval_results']['k_used'] = len(docs)
        _capture_state['retrieval_results']['doc_count'] = len(docs)
        
        return result
    
    retriever.process = debug_process
    
    return retriever


def patch_regulator_manager_for_debugging(regulator_manager):
    """Patch regulator_manager to capture anchor decisions."""
    import types
    
    original_apply_all = regulator_manager.apply_all
    
    def debug_apply_all(proposed_query, reasoning_state, previous_answers, plan_goal=None):
        """Wrapped apply_all that captures regulator decisions."""
        result = original_apply_all(proposed_query, reasoning_state, previous_answers, plan_goal)
        # Regulator decisions are already captured in stabilize_and_retrieve
        return result
    
    regulator_manager.apply_all = debug_apply_all
    
    return regulator_manager


def patch_orchestrator_for_debugging(orchestrator):
    """Patch orchestrator to capture debugging traces."""
    import types
    
    # Patch state_manager
    if hasattr(orchestrator, 'state_manager') and orchestrator.state_manager:
        orchestrator.state_manager = patch_state_manager_for_debugging(orchestrator.state_manager)
    
    # Patch retriever
    if hasattr(orchestrator, 'retriever') and orchestrator.retriever:
        orchestrator.retriever = patch_retriever_for_debugging(orchestrator.retriever)
    
    # Patch regulator_manager if accessible
    if hasattr(orchestrator, 'state_manager') and orchestrator.state_manager:
        if hasattr(orchestrator.state_manager, 'regulator_manager') and orchestrator.state_manager.regulator_manager:
            orchestrator.state_manager.regulator_manager = patch_regulator_manager_for_debugging(
                orchestrator.state_manager.regulator_manager
            )
    
    original_execute_single_step = orchestrator._execute_single_step
    
    async def debug_execute_single_step(self, step, plan, hop=1, plan_goal=None, current_step_index=None, total_steps=None):
        """Wrapped execute_single_step that captures traces."""
        _capture_state['current_step_id'] = step.get("id", "unknown")
        
        # Execute original method
        # Note: original_execute_single_step is already a bound method, so don't pass 'self'
        result = await original_execute_single_step(step, plan, hop, plan_goal, current_step_index, total_steps)
        
        # Extract QA answer from step result
        # _execute_single_step returns a dict with "qa_result" containing "answer" and "confidence"
        qa_result = result.get("qa_result", {})
        qa_answer = qa_result.get("answer", "") if isinstance(qa_result, dict) else ""
        qa_confidence = qa_result.get("confidence", 0.0) if isinstance(qa_result, dict) else 0.0

        # ✅ NEW: Surface orchestrator step-level debug metadata into hop-level debug metadata
        step_debug_metadata = result.get("debug_metadata", {}) if isinstance(result, dict) else {}
        if isinstance(step_debug_metadata, dict) and step_debug_metadata:
            existing = _capture_state.get("debug_metadata", {})
            if not isinstance(existing, dict):
                existing = {}
            existing.setdefault("orchestrator", {})
            if isinstance(existing.get("orchestrator"), dict):
                existing["orchestrator"].update(step_debug_metadata)
            else:
                existing["orchestrator"] = step_debug_metadata
            _capture_state["debug_metadata"] = existing
        
        # Capture anchor decisions (from flow_snapshot if available)
        anchor_decisions = []
        flow_snapshot = _capture_state.get('flow_snapshot', {})
        if flow_snapshot:
            anchors = flow_snapshot.get('anchors', [])
            entity_anchors = flow_snapshot.get('entity_anchors', {})
            
            # Track created anchors
            for anchor in anchors:
                if isinstance(anchor, dict):
                    anchor_decisions.append({
                        'action': 'created',
                        'entity': anchor.get('value', 'unknown'),
                        'strength': anchor.get('strength', 0.0),
                        'hop': anchor.get('hop_created', hop)
                    })
                elif hasattr(anchor, 'value'):
                    anchor_decisions.append({
                        'action': 'created',
                        'entity': str(anchor.value),
                        'strength': getattr(anchor, 'strength', 0.0),
                        'hop': getattr(anchor, 'hop_created', hop)
                    })
            
            # Track entity anchors from dict
            for entity, anchor_data in entity_anchors.items():
                if isinstance(anchor_data, dict):
                    anchor_decisions.append({
                        'action': 'created',
                        'entity': str(entity),
                        'strength': anchor_data.get('strength', 0.0),
                        'hop': anchor_data.get('hop', hop)
                    })
        
        # Capture hop trace
        try:
            debugger.capture_hop(
                hop=hop,
                step_id=_capture_state['current_step_id'],
                proposed_query=_capture_state.get('proposed_query', step.get("description", "")),
                stabilized_query=_capture_state.get('stabilized_query', step.get("description", "")),
                regulator_decisions=_capture_state.get('regulator_decisions', []),
                flow_snapshot=_capture_state.get('flow_snapshot', {}),
                retrieval_params=_capture_state.get('retrieval_params', {}),
                retrieval_results=_capture_state.get('retrieval_results', {}),
                anchor_decisions=anchor_decisions,
                qa_answer=qa_answer,
                qa_confidence=qa_confidence,
                debug_metadata=_capture_state.get('debug_metadata', {})
            )
        except Exception as e:
            logger.warning(f"Failed to capture debug trace for hop {hop}: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear capture state for next hop
        _capture_state['proposed_query'] = ''
        _capture_state['stabilized_query'] = ''
        _capture_state['regulator_decisions'] = []
        _capture_state['flow_snapshot'] = {}
        _capture_state['retrieval_params'] = {}
        _capture_state['retrieval_results'] = {}
        _capture_state['anchor_decisions'] = []
        _capture_state['debug_metadata'] = {}
        
        return result
    
    orchestrator._execute_single_step = types.MethodType(debug_execute_single_step, orchestrator)
    
    return orchestrator


async def debug_evaluate_dataset(dataset_name: str, num_examples: int = 5):
    """Run evaluation with debugging traces."""
    print("\n" + "="*80)
    print("DIFFUSION PIPELINE DEBUGGING MODE")
    print("="*80)
    print(f"\nEvaluating {dataset_name} with {num_examples} examples")
    print("Capturing comprehensive reasoning traces...")
    
    # Load dataset
    from agents.musique_document_loader import _load_musique_from_github, load_musique_example_context_as_documents
    
    examples = _load_musique_from_github("validation")
    eval_examples = examples[:num_examples]
    
    # Create orchestrator
    from agents.mixed_model_orchestrator import create_optimized_marag_pipeline
    
    orchestrator = await create_optimized_marag_pipeline()
    
    # Patch for debugging
    orchestrator = patch_orchestrator_for_debugging(orchestrator)
    
    # Evaluate each example
    for i, example in enumerate(eval_examples, 1):
        question = example.get('question', '')
        ground_truth = example.get('answer', '')
        
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(eval_examples)}: {question[:60]}...")
        print(f"{'='*80}")
        
        # Start trace
        debugger.start_trace(question, ground_truth)
        
        try:
            # Load documents
            documents = load_musique_example_context_as_documents(example)
            orchestrator.add_documents(documents)
            
            # Execute pipeline
            result = await orchestrator.execute_pipeline(question)
            
            # Fix: Extract answer from PipelineResult object
            if hasattr(result, 'final_answer'):
                prediction = result.final_answer
            else:
                prediction = 'Unknown'
            
            # Calculate scores
            em = exact_match_score(prediction, ground_truth)
            f1 = f1_score(prediction, ground_truth)
            
            # Finish trace
            debugger.finish_trace(prediction, em, f1)
            
            print(f"\nResult: EM={em:.2f}, F1={f1:.2f}")
            print(f"Prediction: {prediction}")
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
    
    report_file = results_dir / f"diffusion_debug_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("DEBUG REPORT SUMMARY")
    print("="*80)
    print(f"\nTotal Questions: {report['summary']['total_questions']}")
    print(f"Average EM: {report['summary']['avg_em']:.4f}")
    print(f"Average F1: {report['summary']['avg_f1']:.4f}")
    print(f"Average Hops: {report['summary']['avg_hops']:.2f}")
    print(f"Anchors Created: {report['summary']['total_anchors_created']}")
    print(f"Anchors Rejected: {report['summary']['total_anchors_rejected']}")
    
    # Print failure analysis
    debugger.print_failure_analysis()
    
    print(f"\n📄 Full debug report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Debug Diffusion Pipeline")
    parser.add_argument("--dataset", type=str, default="musique", help="Dataset to evaluate")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to debug")
    args = parser.parse_args()
    
    asyncio.run(debug_evaluate_dataset(args.dataset, args.num_examples))

