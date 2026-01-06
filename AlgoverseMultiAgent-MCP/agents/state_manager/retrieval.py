# agents/state_manager/retrieval.py
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def _cap_late_hop_terms(
    proposed_query: str,
    stabilized_query: str,
    max_added: int = 3,
) -> str:
    """
    For late hops, cap how many new tokens are added to avoid query ballooning.
    Keeps the original proposed query plus up to max_added new tokens.
    """
    if max_added <= 0:
        return proposed_query

    import re

    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", str(s).lower())

    base_tokens = _tokens(proposed_query)
    stab_tokens = _tokens(stabilized_query)
    added = [t for t in stab_tokens if t not in base_tokens]

    if not added:
        return stabilized_query

    keep_added = added[:max_added]
    # Reconstruct: original proposed query + capped added tokens
    return f"{proposed_query} {' '.join(keep_added)}".strip()

async def stabilize_and_retrieve(
    self,
    proposed_query: str,
    hop: int,
    previous_answers: Dict[str, Any],
    plan_goal: Optional[str] = None,
    retriever_agent=None,
    current_step_index: Optional[int] = None,
    total_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    Stabilize reasoning flow then perform entropy-aware retrieval.
    
    This is the central method that:
    1. Updates reasoning flow state
    2. Applies regulators to stabilize query
    3. Checks if early termination is valid (convergence conditions)
    4. Performs entropy-aware retrieval
    
    ✅ FIRST PRINCIPLES: Early termination is a CONVERGENCE condition, not just entropy.
    Requires: low entropy + evidence coverage + plan completion + hierarchical level + answer stability.
    
    Args:
        proposed_query: Raw query from Step Definer
        hop: Current hop number
        previous_answers: Answers from previous steps
        plan_goal: Overall plan goal/question
        retriever_agent: Retriever agent instance
        current_step_index: Current step index (0-based) in the plan
        total_steps: Total number of steps in the plan
        
    Returns:
        Dictionary with documents or early termination answer
    """
    if not self.reasoning_flow or not self.regulator_manager:
        # Fallback to direct retrieval if components not available
        logger.warning("Diffusion-aware components not available, using direct retrieval")
        if retriever_agent:
            fallback_k = 5
            try:
                fallback_k = max(getattr(retriever_agent, "top_k", 5), 5)
            except Exception:
                fallback_k = 5
            rq = proposed_query
            try:
                try:
                    from agents.regulators.granularity_regulator import GranularityRegulator
                except ImportError:
                    from ..regulators.granularity_regulator import GranularityRegulator
                req_domain, req_level, keywords = GranularityRegulator.infer_required(plan_goal, proposed_query)
                if keywords:
                    kw = keywords[0]
                    if rq and kw.lower() not in rq.lower():
                        rq = f"{kw} {rq}".strip()
                        rq = " ".join(rq.split())
            except Exception:
                pass
            result = await retriever_agent.process({"query": rq, "k": fallback_k, "min_similarity": 0.0})
            docs = result.metadata.get("documents", []) or []
            if not docs:
                try:
                    result.metadata["empty_retrieval"] = True
                    result.metadata["k_requested"] = fallback_k
                    result.metadata["k_actual"] = 0
                except Exception:
                    pass
                docs = [{"id": "no_docs", "page_content": "NO RETRIEVAL CONTEXT - documents unavailable", "metadata": {"sentinel": True}}]
            return {
                "documents": docs,
                "stabilized_query": rq,
                "constraints": []
            }
        return {"error": "retriever_agent required"}
    
    # 1. Update reasoning flow state
    flow_snapshot = self._update_flow_state(hop, previous_answers, plan_goal)

    # ✅ INVARIANT GUARDRAIL: proposed_query must be non-empty before regulators/retrieval
    # Variant A (no orchestrator changes):
    # Fallback preference: last stabilized query (previous hop) -> plan_goal.
    query_repair_event = None
    if not proposed_query or not str(proposed_query).strip():
        last_stabilized = getattr(self, "_last_stabilized_query", None)
        fallback_query = last_stabilized if last_stabilized and str(last_stabilized).strip() else plan_goal

        # Absolute last resort: avoid silent collapse into empty query (k=0 retrieval, regulators not firing)
        if not fallback_query or not str(fallback_query).strip():
            fallback_query = "search for relevant information"

        query_repair_event = {
            "query_repaired": True,
            "reason": "empty_proposed_query",
            "fallback_used": (
                "last_stabilized_query"
                if last_stabilized and str(last_stabilized).strip()
                else ("plan_goal" if plan_goal else "default_stub")
            ),
            "original_proposed_query": proposed_query,
            "repaired_query": str(fallback_query),
            "hop": hop,
        }
        proposed_query = str(fallback_query)

    # NOTE: We intentionally do NOT mutate the query pre-regulators based on hierarchy mismatch here.
    # That can interfere with existing stabilization expectations. Instead, we apply a minimal
    # failure-triggered granularity re-expansion in the retrieval failure path (docs=0 retry) below.
    
    # Determine per-step first-hop flag (approximate: hop==1 in current implementation)
    is_step_first_hop = (hop == 1)
    
    # 2. Apply regulators to stabilize query
    # ====================================================================
    # DIFFUSION PROCESS: Initial Condition (u(x,0))
    # ====================================================================
    # GranularityRegulator is applied FIRST as the initial condition,
    # setting the correct hierarchical level before retrieval begins.
    # This prevents hierarchical leakage and level-mismatch cascades.
    # Other regulators then apply iteratively to guide convergence.
    # ====================================================================
    stabilized_query, constraints = self.regulator_manager.apply_all(
        proposed_query=proposed_query,
        reasoning_state=flow_snapshot.dict() if flow_snapshot else {},
        previous_answers=previous_answers,
        plan_goal=plan_goal
    )

    # 🔒 Invariant: Granularity must be present on every retrieval call.
    # We infer required level and add a synthetic constraint if missing.
    try:
        try:
            from agents.regulators.granularity_regulator import GranularityRegulator
        except ImportError:
            from ..regulators.granularity_regulator import GranularityRegulator
        req_domain, req_level, keywords = GranularityRegulator.infer_required(plan_goal, proposed_query or stabilized_query)
    except Exception as e:
        logger.warning(f"Granularity inference failed; proceeding without granularity. err={e}")
        req_domain, req_level, keywords = (None, None, [])
    try:
        has_granularity = any(
            (c.dict().get("regulator_name") if hasattr(c, "dict") else c.get("regulator_name", ""))
            == "Granularity"
            for c in (constraints or [])
        )
    except Exception:
        has_granularity = False
    granularity_injected = False
    # Ensure query carries the level keyword (idempotent)
    if req_domain and req_level and keywords:
        kw = keywords[0]
        if stabilized_query and kw.lower() not in stabilized_query.lower():
            stabilized_query = f"{kw} {stabilized_query}".strip()
            stabilized_query = " ".join(stabilized_query.split())
            granularity_injected = True
    if (not req_domain or not req_level) and flow_snapshot and hasattr(flow_snapshot, "__dict__"):
        try:
            req_domain = getattr(flow_snapshot, "required_domain", None) or getattr(flow_snapshot, "granularity_domain", None) or req_domain
            req_level = getattr(flow_snapshot, "required_level", None) or getattr(flow_snapshot, "granularity_level", None) or req_level
        except Exception:
            pass

    if not has_granularity and req_domain and req_level:
        constraints = list(constraints or [])
        constraints.append({
            "regulator_name": "Granularity",
            "parameters": {
                "required_domain": req_domain,
                "required_level": req_level
            }
        })
        granularity_injected = True or granularity_injected
        if flow_snapshot and hasattr(flow_snapshot, "__dict__"):
            try:
                flow_snapshot.granularity_injected = True
                flow_snapshot.granularity_level = req_level
            except Exception:
                pass

    # Track last stabilized query for the next hop's invariant fallback.
    # (This is the "last stabilized query" contract asked for in Variant A.)
    if stabilized_query and str(stabilized_query).strip():
        setattr(self, "_last_stabilized_query", stabilized_query)
    elif proposed_query and str(proposed_query).strip():
        # Belt-and-suspenders: if regulators somehow returned empty, fall back to repaired query.
        stabilized_query = proposed_query
        setattr(self, "_last_stabilized_query", stabilized_query)

    # ✅ Late-hop guard: cap added terms to avoid ballooning
    if hop >= 3:
        capped = _cap_late_hop_terms(proposed_query, stabilized_query, max_added=3)
        if capped != stabilized_query:
            stabilized_query = capped
        setattr(self, "_last_stabilized_query", stabilized_query)
    
    # ✅ FIRST PRINCIPLES FIX: Early termination should NOT happen BEFORE retrieval
    # ====================================================================
    # CRITICAL: Early termination before retrieval causes the current step to never execute,
    # returning the previous step's answer instead of finding the current step's answer.
    # 
    # Example bug pattern: Multi-hop questions where step N finds an intermediate entity,
    # and step N+1 should find a property/relationship of that entity.
    # - Step N finds: intermediate entity (e.g., a company, person, location)
    # - Step N+1 should find: related entity (e.g., founder, headquarters, parent company)
    # - But early termination returns step N's answer, skipping step N+1 entirely
    # 
    # Early termination should ONLY happen AFTER the current step has executed
    # and found its answer. It should check if the CURRENT step's answer satisfies
    # convergence conditions, not the previous step's answer.
    # ====================================================================
    # REMOVED: Early termination check before retrieval
    # Early termination will be checked AFTER QA agent produces the current step's answer
    
    # 3. Entropy-aware retrieval
    if not retriever_agent:
        raise ValueError("retriever_agent required for entropy-aware retrieval")
    
    result = await self._entropy_aware_retrieve(
        stabilized_query=stabilized_query,
        flow_snapshot=flow_snapshot,
        constraints=constraints,
        retriever_agent=retriever_agent,
        plan_goal=plan_goal,
        previous_answers=previous_answers,
        hop=hop,
        is_step_first_hop=(hop == 1),
    )

    # Attach debug metadata so debug reports can verify query repair events.
    if query_repair_event:
        if isinstance(result, dict):
            result.setdefault("debug_metadata", {})
            result["debug_metadata"]["query_repair_event"] = query_repair_event

    # Attach debug metadata
    if isinstance(result, dict):
        result.setdefault("debug_metadata", {})
        result["debug_metadata"]["granularity_injected"] = granularity_injected
        result["debug_metadata"]["granularity_level"] = req_level if granularity_injected else None
    return result

async def _entropy_aware_retrieve(
    self,
    stabilized_query: str,
    flow_snapshot: Optional[Any],
    constraints: List,
    retriever_agent,
    plan_goal: Optional[str] = None,
    previous_answers: Optional[Dict[str, Any]] = None,
    hop: int = 1,
    is_step_first_hop: bool = False,
) -> Dict[str, Any]:
    """
    Perform entropy-aware retrieval with regulator constraints.
    
    Args:
        stabilized_query: Query after regulator stabilization
        flow_snapshot: Unified flow snapshot
        constraints: List of regulator constraints
        retriever_agent: Retriever agent instance
        
    Returns:
        Dictionary with retrieved documents and metadata
    """
    # Build retrieval input with constraints
    # ✅ FIX: Explicitly pass k and min_similarity from retriever defaults
    # ✅ FIRST PRINCIPLES: Pass hierarchical context for parent entity extraction
    flow_snapshot_dict = flow_snapshot.dict() if flow_snapshot and hasattr(flow_snapshot, 'dict') else {}
    
    # Normalize flag
    is_step_first_hop = bool(is_step_first_hop)
    
    # ✅ FIX: Extract entropy/diffusion with proper fallback handling
    entropy_val = 0.0
    diffusion_val = 0.0
    
    if flow_snapshot:
        if isinstance(flow_snapshot, dict):
            entropy_val = flow_snapshot.get('entropy', 0.0)
            diffusion_val = flow_snapshot.get('diffusion_coefficient', 0.0)
        else:
            entropy_val = getattr(flow_snapshot, 'entropy', 0.0)
            diffusion_val = getattr(flow_snapshot, 'diffusion_coefficient', 0.0)
    
    # ✅ FIX: Use dict version as fallback if object values are 0.0
    if (entropy_val == 0.0 or diffusion_val == 0.0) and flow_snapshot_dict:
        if entropy_val == 0.0:
            entropy_val = flow_snapshot_dict.get('entropy', 0.0)
        if diffusion_val == 0.0:
            diffusion_val = flow_snapshot_dict.get('diffusion_coefficient', 0.0)

    # ✅ EXTRA GUARD: If still zero entropy, try entropy_tracker current state (avoids stagnation)
    if entropy_val == 0.0 and hasattr(self, "entropy_tracker") and self.entropy_tracker:
        try:
            es = self.entropy_tracker.get_current_state()
            if es and getattr(es, "entropy", 0.0) > 0.0:
                entropy_val = es.entropy
        except Exception:
            pass
    
    # Extract hierarchical level requirement from GranularityRegulator constraint
    required_domain = None
    required_level = None
    for constraint in constraints:
        constraint_dict = constraint.dict() if hasattr(constraint, 'dict') else (constraint if isinstance(constraint, dict) else {})
        if constraint_dict.get('regulator_name') == 'Granularity':
            params = constraint_dict.get('parameters', {})
            required_domain = params.get('required_domain')
            required_level = params.get('required_level')
            break
    granularity_present = bool(required_domain and required_level)
    
    # Add hierarchical context to flow snapshot
    if required_domain and required_level:
        flow_snapshot_dict['required_domain'] = required_domain
        flow_snapshot_dict['required_level'] = required_level

    # ✅ CONTROL-LAYER FIX: Entropy floor for adaptive retrieval
    # If entropy collapses (H(t) < ε) but state is under-supported, force exploration by enforcing k >= k_min.
    # This prevents premature convergence from shutting down retrieval (k=0 / effectively no search).
    EPS_ENTROPY_FLOOR = 0.12
    # Raise the floor to avoid k=0 collapse when entropy is low
    K_MIN = 10
    MIN_BELIEF_COUNT = 2
    MIN_EVIDENCE_TERMS = 2
    # Epistemic support-aware widening (context floor)
    SUPPORT_BELIEF_FLOOR = 1
    SUPPORT_EVIDENCE_TERM_FLOOR = 2
    SUPPORT_K_FLOOR = 12
    SUPPORT_MIN_SIM_FLOOR = 0.25

    beliefs = flow_snapshot_dict.get("beliefs", {}) or {}
    evidence_terms = flow_snapshot_dict.get("evidence_terms", []) or []
    belief_count = len(beliefs)
    evidence_terms_count = len(evidence_terms)
    evidence_density_low = evidence_terms_count < MIN_EVIDENCE_TERMS
    epistemic_support_low = (
        belief_count < SUPPORT_BELIEF_FLOOR or evidence_terms_count < SUPPORT_EVIDENCE_TERM_FLOOR
    )

    # Surface support signals on the flow snapshot for downstream consumers
    flow_snapshot_dict["belief_count"] = belief_count
    flow_snapshot_dict["evidence_terms_count"] = evidence_terms_count
    flow_snapshot_dict["epistemic_support_low"] = epistemic_support_low
    
    retrieval_input = {
        "query": stabilized_query,
        "k": getattr(retriever_agent, 'top_k', 15),  # Use retriever's top_k, default to 15 for scattered docs
        "min_similarity": getattr(retriever_agent, 'min_similarity', 0.2),  # Use retriever's min_similarity
        "regulator_constraints": [c.dict() if hasattr(c, 'dict') else c for c in constraints],
        "flow_snapshot": flow_snapshot_dict,
        "entropy_penalty": entropy_val,  # ✅ FIX: Use extracted value
        "diffusion_penalty": diffusion_val  # ✅ FIX: Use extracted value
    }

    # Early-hop recall bump when entropy collapsed
    try:
        hop_num = int(getattr(flow_snapshot, "hop", 1) if flow_snapshot else 1)
    except Exception:
        hop_num = 1
    # If hop was explicitly passed, prefer it
    try:
        if hop:
            hop_num = int(hop)
    except Exception:
        pass
    if entropy_val == 0.0 and hop_num <= 2:
        old_k = retrieval_input["k"]
        old_min_sim = retrieval_input["min_similarity"]
        retrieval_input["k"] = max(int(retrieval_input.get("k") or 0), 15)
        retrieval_input["min_similarity"] = max(0.1, old_min_sim * 0.8)
        logger.debug(
            f"🚀 Early-hop recall bump: hop={hop_num}, entropy=0.0 → "
            f"k {old_k}->{retrieval_input['k']}, min_sim {old_min_sim:.3f}->{retrieval_input['min_similarity']:.3f}"
        )

    # Late-hop min-k floor to avoid empty retrievals
    if hop_num >= 3:
        try:
            base_k = int(retrieval_input.get("k", 0) or 0)
        except Exception:
            base_k = 0
        retrieval_input["k"] = max(base_k, 5)

    # Epistemic support-aware widening on first hop of a step (acts as an observability floor)
    if epistemic_support_low and is_step_first_hop:
        try:
            base_k = int(retrieval_input.get("k") or 0)
        except Exception:
            base_k = 0
        retrieval_input["k"] = max(base_k, SUPPORT_K_FLOOR)
        try:
            base_ms = float(retrieval_input.get("min_similarity") or 1.0)
        except Exception:
            base_ms = 1.0
        retrieval_input["min_similarity"] = min(base_ms, SUPPORT_MIN_SIM_FLOOR)
        retrieval_input["support_widened"] = True
        logger.debug(
            f"[support_widen] hop={hop_num}, k->{retrieval_input['k']}, "
            f"min_sim->{retrieval_input['min_similarity']:.3f}, "
            f"belief_count={belief_count}, evidence_terms={evidence_terms_count}"
        )

    # Force exploration only when entropy is "collapsed" AND state/evidence is weak.
    if entropy_val < EPS_ENTROPY_FLOOR and (belief_count < MIN_BELIEF_COUNT or evidence_density_low):
        old_k = retrieval_input["k"]
        retrieval_input["k"] = max(int(retrieval_input.get("k") or 0), K_MIN)
        logger.debug(
            f"🧯 Entropy floor triggered: H={entropy_val:.3f} < {EPS_ENTROPY_FLOOR} with "
            f"belief_count={belief_count}, evidence_terms={len(evidence_terms)} → forcing k {old_k}→{retrieval_input['k']}"
        )
    
    # Absolute guard: clamp k on every call
    try:
        k_val = int(retrieval_input.get("k") or 0)
    except Exception:
        k_val = 0
    retrieval_input["k"] = max(k_val, K_MIN if 'K_MIN' in locals() else 5)

    # Call retriever with constraints (log requested k/min_sim)
    logger.debug(
        f"[RetrievalCall] hop={hop_num}, req_k={retrieval_input.get('k')}, "
        f"min_sim={retrieval_input.get('min_similarity'):.3f} "
        f"entropy={entropy_val:.3f}"
    )
    result = await retriever_agent.process(retrieval_input)
    docs = result.metadata.get("documents", []) or []

    # ✅ CONTROL-LAYER FIX: If retrieval returns 0 documents, force exploration + explicit granularity and retry once.
    # This bypasses entropy-based shutdown and prevents k=0 retrieval collapse.
    if len(docs) == 0:
        stabilized_query_retry = stabilized_query
        try:
            try:
                from agents.regulators.granularity_regulator import GranularityRegulator
            except ImportError:
                from ..regulators.granularity_regulator import GranularityRegulator

            granularity_reg = GranularityRegulator()
            req_domain, req_level, keywords = GranularityRegulator.infer_required(plan_goal, stabilized_query)
            if req_domain and req_level and keywords:
                kw = keywords[0]
                if kw.lower() not in stabilized_query.lower():
                    stabilized_query_retry = f"{kw} {stabilized_query}".strip()
        except Exception:
            stabilized_query_retry = stabilized_query

        retry_input = dict(retrieval_input)
        retry_input["query"] = stabilized_query_retry
        try:
            k_retry = int(retry_input.get("k") or 0)
        except Exception:
            k_retry = 0
        retry_input["k"] = max(k_retry, K_MIN if 'K_MIN' in locals() else 5)
        # Soften min_similarity slightly on retry
        try:
            old_ms = float(retry_input.get("min_similarity") or 0.2)
        except Exception:
            old_ms = 0.2
        retry_input["min_similarity"] = max(0.05, old_ms * 0.9)

        logger.debug(
            f"🔁 Retrieval failure-retry: initial docs=0 → retry with k={retry_input['k']} "
            f"and query='{stabilized_query_retry[:120]}'"
        )

        result = await retriever_agent.process(retry_input)
        docs = result.metadata.get("documents", []) or []
        stabilized_query = stabilized_query_retry
        # Mark granularity injection on retry path
        try:
            result_metadata = result.metadata
            result_metadata["granularity_injected_retry"] = True
            result_metadata["granularity_level_retry"] = req_level if req_level else None
        except Exception:
            pass
    
    # ✅ CONTROL-LAYER FIX: If still no docs after one retry, perform a low-sim fallback with stripped constraints.
    if len(docs) == 0:
        fallback_query = stabilized_query
        fallback_keywords = []
        try:
            try:
                from agents.regulators.granularity_regulator import GranularityRegulator
            except ImportError:
                from ..regulators.granularity_regulator import GranularityRegulator
            fallback_req_domain, fallback_req_level, fallback_keywords = GranularityRegulator.infer_required(plan_goal, stabilized_query)
            if fallback_req_domain and fallback_req_level and fallback_keywords:
                kw = fallback_keywords[0]
                if kw.lower() not in fallback_query.lower():
                    fallback_query = f"{kw} {fallback_query}".strip()
                    fallback_query = " ".join(fallback_query.split())
        except Exception:
            fallback_query = stabilized_query

        fallback_input = dict(retrieval_input)
        fallback_input.update({
            "query": fallback_query,
            "k": max(int(retrieval_input.get("k") or 0), K_MIN if 'K_MIN' in locals() else 5),
            "min_similarity": 0.0,
            "regulator_constraints": [],
        })
        result = await retriever_agent.process(fallback_input)
        docs = result.metadata.get("documents", []) or []
        stabilized_query = fallback_query
        try:
            result.metadata["fallback_low_sim"] = True
            result.metadata["k_requested"] = fallback_input.get("k")
            result.metadata["k_actual"] = len(docs)
        except Exception:
            pass

    # ✅ CONTROL-LAYER FIX: If still empty, inject a sentinel doc so downstream never sees k_actual=0.
    if len(docs) == 0:
        sentinel_doc = {
            "id": "no_docs",
            "page_content": "NO RETRIEVAL CONTEXT - documents unavailable",
            "metadata": {
                "sentinel": True,
                "granularity_injected": granularity_present,
                "granularity_level": required_level,
            },
        }
        docs = [sentinel_doc]
        try:
            result.metadata["empty_retrieval"] = True
            # ensure requested_k reflects the enforced minimum so debug reports don't show k=0
            try:
                req_k = int(retrieval_input.get("k") or 0)
            except Exception:
                req_k = 0
            result.metadata["k_requested"] = max(req_k, K_MIN if 'K_MIN' in locals() else 5)
            result.metadata["k_actual"] = 1
            result.metadata["granularity_injected"] = granularity_present
            result.metadata["granularity_level"] = required_level
        except Exception:
            pass
    
    logger.debug(
        f"Entropy-aware retrieval: query='{stabilized_query}', "
        f"k={retrieval_input.get('k')}, min_similarity={retrieval_input.get('min_similarity')}, "
        f"H(t)={(flow_snapshot.entropy if flow_snapshot else 0.0):.3f}, "
        f"D(t)={(flow_snapshot.diffusion_coefficient if flow_snapshot else 0.0):.3f}, "
        f"documents={len(docs)}"
    )
    
    support_meta = {
        "belief_count": belief_count,
        "evidence_terms_count": evidence_terms_count,
        "epistemic_support_low": epistemic_support_low,
        "support_widened": retrieval_input.get("support_widened", False),
        "k_requested": retrieval_input.get("k"),
        "k_actual": len(docs),
        "docs_filtered_count": result.metadata.get("filtered_count") if hasattr(result, "metadata") else None,
        "low_similarity_floor": result.metadata.get("low_similarity_floor") if hasattr(result, "metadata") else None,
        "fallback_low_sim": result.metadata.get("fallback_low_sim") if hasattr(result, "metadata") else None,
    }

    return {
        "documents": docs,
        "stabilized_query": stabilized_query,
        "constraints": constraints,
        "flow_snapshot": flow_snapshot.dict() if flow_snapshot and hasattr(flow_snapshot, 'dict') else None,
        "support_signals": support_meta,
    }

def _check_termination_validity(
    self,
    flow_snapshot: Any,
    plan_goal: Optional[str] = None,
    hop: int = 1,
    previous_answers: Optional[Dict[str, Any]] = None,
    constraints: Optional[List] = None,
    current_step_index: Optional[int] = None,
    total_steps: Optional[int] = None,
    current_query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check if reasoning can terminate early based on CONVERGENCE conditions.
    
    ✅ FIRST PRINCIPLES: Stopping is a convergence condition, not just an entropy condition.
    
    Convergence requires ALL of:
    1. Low entropy + evidence coverage high + plan steps completed + 
       granularity target satisfied + retrieval no longer surfacing new information
    2. Answer stable across cycles (last 2-3 answers are consistent)
    3. No new retrieval improves the evidence score (implicitly checked via answer stability)
    4. Required hierarchical level is met
    5. ✅ DIFFUSION-AWARE: Answer satisfies the query's boundary condition (not from input space)
    
    Args:
        flow_snapshot: Current flow snapshot
        plan_goal: Plan goal to check against
        hop: Current hop number
        previous_answers: Answers from previous steps
        constraints: Regulator constraints (for hierarchical level check)
        current_step_index: Current step index (0-based) in the plan
        total_steps: Total number of steps in the plan
        current_query: Current query being processed (boundary condition to verify against)
        
    Returns:
        Dict with "can_terminate" (bool) and "reason" (str) explaining why/why not
    """
    if not flow_snapshot or not self.entropy_tracker:
        return {"can_terminate": False, "reason": "Missing flow snapshot or entropy tracker"}
    
    entropy_state = self.entropy_tracker.get_current_state()
    if not entropy_state:
        return {"can_terminate": False, "reason": "No entropy state available"}
    
    # ====================================================================
    # CONVERGENCE CONDITION 1: Plan steps must be completed (or on last step)
    # ====================================================================
    # ✅ CRITICAL: Don't terminate early if there are more steps to execute
    if current_step_index is not None and total_steps is not None:
        # Only allow termination if we're on the last step (or all steps are effectively done)
        if current_step_index < total_steps - 1:
            return {
                "can_terminate": False,
                "reason": f"Not on last step: step {current_step_index + 1}/{total_steps} - must complete all steps"
            }
    
    # ====================================================================
    # CONVERGENCE CONDITION 2: Low entropy + high confidence + low drift
    # ✅ FIX 2: Adaptive thresholds based on query complexity
    # ====================================================================
    # Calculate adaptive thresholds based on query complexity:
    # - Simple queries (1-2 steps): Stricter thresholds (higher confidence required)
    # - Complex queries (3+ steps): More lenient thresholds (allow earlier convergence)
    # - Multi-hop queries: More lenient to prevent over-processing
    # ====================================================================
    query_complexity = self._calculate_query_complexity(
        total_steps=total_steps,
        hop=hop,
        plan_goal=plan_goal
    )
    
    # Adaptive entropy threshold: Lower for simple queries, higher for complex
    # Simple: < 0.4, Medium: < 0.5, Complex: < 0.6
    entropy_threshold = 0.5  # Default
    if query_complexity == "simple":
        entropy_threshold = 0.4
    elif query_complexity == "complex":
        entropy_threshold = 0.6
    
    # Adaptive confidence threshold: Higher for simple queries, lower for complex
    # Simple: >= 0.85, Medium: >= 0.75, Complex: >= 0.7
    confidence_threshold = 0.75  # Default (relaxed from 0.8)
    if query_complexity == "simple":
        confidence_threshold = 0.85
    elif query_complexity == "complex":
        confidence_threshold = 0.7
    
    # Adaptive drift threshold: Lower for simple queries, higher for complex
    # Simple: < 0.25, Medium: < 0.3, Complex: < 0.4
    drift_threshold = 0.3  # Default
    if query_complexity == "simple":
        drift_threshold = 0.25
    elif query_complexity == "complex":
        drift_threshold = 0.4
    
    # Check entropy
    if entropy_state.entropy >= entropy_threshold:
        return {
            "can_terminate": False, 
            "reason": f"Entropy too high: {entropy_state.entropy:.3f} >= {entropy_threshold:.3f} (adaptive threshold for {query_complexity} query)"
        }
    
    # Check confidence
    if entropy_state.confidence < confidence_threshold:
        return {
            "can_terminate": False, 
            "reason": f"Confidence too low: {entropy_state.confidence:.3f} < {confidence_threshold:.3f} (adaptive threshold for {query_complexity} query)"
        }
    
    # Check drift
    if entropy_state.drift_from_previous >= drift_threshold:
        return {
            "can_terminate": False, 
            "reason": f"Drift too high: {entropy_state.drift_from_previous:.3f} >= {drift_threshold:.3f} (adaptive threshold for {query_complexity} query)"
        }
    
    # ====================================================================
    # CONVERGENCE CONDITION 3: Answer stable across cycles
    # ====================================================================
    if previous_answers and len(previous_answers) >= 2:
        # Check if last 2-3 answers are consistent
        answer_values = []
        for answer_data in list(previous_answers.values())[-3:]:
            if isinstance(answer_data, dict):
                answer_text = answer_data.get("answer", "").lower().strip()
            else:
                answer_text = str(answer_data).lower().strip()
            if answer_text and answer_text not in ["unknown", "none", "n/a", ""]:
                answer_values.append(answer_text)
        
        # Answers must be consistent (all same or very similar)
        if len(answer_values) >= 2:
            unique_answers = set(answer_values)
            if len(unique_answers) > 1:
                return {
                    "can_terminate": False,
                    "reason": f"Answer not stable across cycles - answers differ: {list(unique_answers)[:3]}"
                }
    
    # ====================================================================
    # CONVERGENCE CONDITION 4: Required hierarchical level is met
    # ====================================================================
    if constraints:
        try:
            from agents.regulators.granularity_regulator import GranularityRegulator
        except ImportError:
            from ..regulators.granularity_regulator import GranularityRegulator
        granularity_reg = GranularityRegulator()
        
        # Extract required hierarchical level from GranularityRegulator constraint
        required_domain = None
        required_level = None
        for constraint in constraints:
            constraint_dict = constraint.dict() if hasattr(constraint, 'dict') else (constraint if isinstance(constraint, dict) else {})
            if constraint_dict.get('regulator_name') == 'Granularity':
                params = constraint_dict.get('parameters', {})
                required_domain = params.get('required_domain')
                required_level = params.get('required_level')
                break
        
        # If hierarchical level is required, check if last answer meets it
        if required_domain and required_level and previous_answers:
            last_answer_data = list(previous_answers.values())[-1]
            last_answer = last_answer_data.get("answer", "") if isinstance(last_answer_data, dict) else str(last_answer_data)
            
            if last_answer and last_answer.lower() not in ["unknown", "none", "n/a", ""]:
                # Check if answer is at required hierarchical level
                answer_domain, answer_level, _ = granularity_reg.classify_entity_level(last_answer)
                if answer_domain == required_domain and answer_level:
                    answer_level_num = granularity_reg.get_level_number(answer_domain, answer_level)
                    required_level_num = granularity_reg.get_level_number(required_domain, required_level)
                    
                    # Answer must be at required level (not higher or lower)
                    if answer_level_num != required_level_num:
                        return {
                            "can_terminate": False,
                            "reason": f"Hierarchical level not met: answer is {answer_level} (level {answer_level_num}), "
                                     f"required is {required_level} (level {required_level_num})"
                        }
    
    # ====================================================================
    # CONVERGENCE CONDITION 5: Evidence coverage check
    # ====================================================================
    # Check if we have sufficient evidence (not just low entropy)
    if previous_answers:
        # Count non-"unknown" answers as evidence
        valid_answers = sum(
            1 for answer_data in previous_answers.values()
            if isinstance(answer_data, dict) and answer_data.get("answer", "").lower().strip() not in ["unknown", "none", "n/a", ""]
            or (not isinstance(answer_data, dict) and str(answer_data).lower().strip() not in ["unknown", "none", "n/a", ""])
        )
        
        # Need at least some valid answers (evidence coverage)
        if valid_answers == 0:
            return {"can_terminate": False, "reason": "No valid answers found - insufficient evidence coverage"}
    
    # ====================================================================
    # CONVERGENCE CONDITION 6: Answer satisfies query boundary condition (DIFFUSION-AWARE)
    # ====================================================================
    # ✅ DIFFUSION PERSPECTIVE: The query defines a boundary condition u(x, t=0) = f(x).
    # The answer must be in the solution space that satisfies this boundary condition,
    # not in the input space (entities mentioned in the query).
    # 
    # In diffusion terms: if the query asks "What is X's Y?", then:
    # - X is in the input/initial condition space
    # - Y is the property/relationship being queried
    # - The answer must be in the solution space (a value for Y), not X itself
    #
    # This prevents convergence on answers that are semantically the input entity
    # rather than a valid solution to the query's boundary condition.
    #
    # ✅ FIRST PRINCIPLES FIX: This now checks the CURRENT step's answer (the last one in previous_answers),
    # not the previous step's answer. This is called AFTER the current step has executed and found its answer.
    if current_query and previous_answers:
        last_answer_data = list(previous_answers.values())[-1]
        last_answer = last_answer_data.get("answer", "") if isinstance(last_answer_data, dict) else str(last_answer_data)
        
        if last_answer and last_answer.lower() not in ["unknown", "none", "n/a", ""]:
            # Check if CURRENT step's answer satisfies the CURRENT step's query boundary condition
            # This is the correct check: validate the answer we just found, not a previous answer
            relevance_result = self._check_answer_satisfies_boundary_condition(last_answer, current_query)
            if not relevance_result["satisfies"]:
                return {
                    "can_terminate": False,
                    "reason": (
                        f"Current step answer '{last_answer}' does not satisfy query boundary condition: "
                        f"{relevance_result.get('reason', 'answer is in input space, not solution space')}"
                    )
                }
    
    # All convergence conditions met
    return {
        "can_terminate": True,
        "reason": (
            f"Convergence conditions met: entropy={entropy_state.entropy:.3f} (low), "
            f"confidence={entropy_state.confidence:.3f} (high), "
            f"drift={entropy_state.drift_from_previous:.3f} (low), "
            f"answer stable, hierarchical level satisfied, plan steps completed, "
            f"answer satisfies query boundary condition"
        )
    }

def _check_answer_satisfies_boundary_condition(self, answer: str, query: str) -> Dict[str, Any]:
    """
    Check if an answer satisfies the query's boundary condition from a diffusion perspective.
    
    ✅ DIFFUSION PERSPECTIVE: In a diffusion process, the boundary condition u(x, 0) = f(x)
    defines what we're solving for. The answer must be in the solution space that satisfies
    this boundary condition, not in the input/initial condition space.
    
    Semantic Structure Analysis:
    - Query defines: subject (input entity) + predicate (what we're asking) + expected answer type
    - Answer must be semantically distinct from the subject/input entities
    - Answer must be in the solution space (the type of thing being asked for)
    
    This is generalizable because it analyzes semantic structure, not specific patterns.
    
    Args:
        answer: The answer to check
        query: The current query (boundary condition)
        
    Returns:
        Dict with "satisfies" (bool) and "reason" (str) explaining why/why not
    """
    if not answer or not query:
        return {"satisfies": True, "reason": "Cannot verify - missing answer or query"}
    
    answer_lower = answer.lower().strip()
    query_lower = query.lower().strip()
    
    # Extract semantic structure: subject entities (input space) and query type
    import re
    
    # Extract all potential subject entities from the query
    # These are entities in the "input space" - the answer should NOT be one of these
    subject_entities = self._extract_subject_entities(query_lower)
    
    # Check if answer matches any subject entity (input space violation)
    for subject_entity in subject_entities:
        # Normalize for comparison (handle variations)
        subject_normalized = subject_entity.lower().strip()
        answer_normalized = answer_lower.strip()
        
        # Exact match or answer is contained in subject (e.g., "Steve" matches "Steve Hillage")
        if answer_normalized == subject_normalized:
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is in input space (matches subject entity '{subject_entity}') "
                    f"rather than solution space. Query asks for information about '{subject_entity}', "
                    f"not '{subject_entity}' itself."
                )
            }
        
        # Check if answer is a substring of subject (e.g., "Steve" in "Steve Hillage")
        # This catches cases where answer is just part of the subject entity
        if len(answer_normalized) > 2 and answer_normalized in subject_normalized:
            # But allow if answer is longer/more specific (e.g., "Steve Hillage" is valid answer to "Who is Steve?")
            if len(answer_normalized) < len(subject_normalized) * 0.7:  # Answer is significantly shorter
                return {
                    "satisfies": False,
                    "reason": (
                        f"Answer '{answer}' appears to be part of subject entity '{subject_entity}' "
                        f"(input space), not a solution to the query."
                    )
                }
    
    # Check for relationship/attribute queries where answer should be semantically distinct
    # Pattern: "X's Y" or "Y of X" where X is subject, Y is what we're asking for
    possessive_pattern = r"(\w+(?:\s+\w+)*)'s\s+(\w+)"
    match = re.search(possessive_pattern, query_lower)
    if match:
        subject_entity = match.group(1).strip()
        attribute_or_relation = match.group(2).strip()
        
        # If answer matches subject, it's wrong (input space, not solution space)
        if answer_lower == subject_entity.lower():
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is the subject entity '{subject_entity}' (input space). "
                    f"Query asks for '{attribute_or_relation}' of '{subject_entity}', "
                    f"so answer must be in solution space (a value for '{attribute_or_relation}'), "
                    f"not the subject itself."
                )
            }
    
    # Check for "who is X's Y?" type queries
    who_pattern = r"who\s+is\s+(\w+(?:\s+\w+)*)'s\s+(\w+)"
    match = re.search(who_pattern, query_lower)
    if match:
        subject_entity = match.group(1).strip()
        relation = match.group(2).strip()
        
        if answer_lower == subject_entity.lower():
            return {
                "satisfies": False,
                "reason": (
                    f"Answer '{answer}' is the subject '{subject_entity}' (input space). "
                    f"Query asks 'who is {subject_entity}'s {relation}?', "
                    f"so answer must be a different entity (the {relation}), not the subject."
                )
            }
    
    # If none of the violations detected, assume answer satisfies boundary condition
    return {
        "satisfies": True,
        "reason": "Answer is semantically distinct from query subject entities and appears to be in solution space"
    }

def _extract_subject_entities(self, query: str) -> List[str]:
    """
    Extract subject entities (input space) from a query using semantic analysis.
    
    This is generalizable because it uses linguistic patterns to identify
    entities that are part of the query's input/initial condition, not the solution.
    
    Args:
        query: The query to analyze
        
    Returns:
        List of subject entity strings (normalized)
    """
    import re
    entities = []
    
    # Pattern 1: Possessive constructions "X's Y" - X is subject
    possessive_pattern = r"(\w+(?:\s+\w+)*)'s\s+\w+"
    matches = re.findall(possessive_pattern, query)
    entities.extend([m.strip() for m in matches])
    
    # Pattern 2: "Y of X" constructions - X is subject
    of_pattern = r"\w+\s+of\s+(\w+(?:\s+\w+)*)"
    matches = re.findall(of_pattern, query)
    entities.extend([m.strip() for m in matches])
    
    # Pattern 3: "who/what/which X" - X might be subject if followed by verb
    # But we're more conservative here - only extract if it's clearly a subject
    
    # Pattern 4: Named entities (capitalized sequences) that appear early in query
    # These are often subject entities
    capitalized_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    matches = re.findall(capitalized_pattern, query)
    # Only take first few as they're likely subjects
    entities.extend([m.strip() for m in matches[:2]])
    
    # Remove duplicates and normalize
    unique_entities = []
    seen = set()
    for entity in entities:
        entity_lower = entity.lower().strip()
        if entity_lower and entity_lower not in seen and len(entity_lower) > 2:
            unique_entities.append(entity)
            seen.add(entity_lower)
    
    return unique_entities

def _calculate_query_complexity(
    self,
    total_steps: Optional[int] = None,
    hop: int = 1,
    plan_goal: Optional[str] = None
) -> str:
    """
    Calculate query complexity to determine adaptive convergence thresholds.
    
    ✅ FIX 2: Adaptive thresholds based on query complexity
    - Simple queries (1-2 steps): Stricter thresholds (higher confidence required)
    - Medium queries (3-4 steps): Standard thresholds
    - Complex queries (5+ steps): More lenient thresholds (allow earlier convergence)
    
    Args:
        total_steps: Total number of steps in the plan
        hop: Current hop number
        plan_goal: Plan goal/question (for additional complexity hints)
        
    Returns:
        Complexity level: "simple", "medium", or "complex"
    """
    # Primary indicator: number of steps
    if total_steps is not None:
        if total_steps <= 2:
            return "simple"
        elif total_steps <= 4:
            return "medium"
        else:
            return "complex"
    
    # Secondary indicator: current hop number (if we're deep, it's complex)
    if hop >= 5:
        return "complex"
    elif hop <= 2:
        return "simple"
    
    # Tertiary indicator: plan goal complexity (heuristic)
    if plan_goal:
        plan_lower = plan_goal.lower()
        # Multi-hop indicators
        complex_indicators = ["compare", "difference", "relationship", "both", "all", "multiple"]
        if any(indicator in plan_lower for indicator in complex_indicators):
            return "complex"
        
        # Simple indicators
        simple_indicators = ["what is", "who is", "when", "where"]
        if any(indicator in plan_lower for indicator in simple_indicators):
            return "simple"
    
    # Default to medium
    return "medium"

