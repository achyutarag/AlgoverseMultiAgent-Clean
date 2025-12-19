# Diffusion Pipeline Debugging Guide

## Overview

The `debug_diffusion_pipeline.py` script captures comprehensive reasoning traces to identify failure modes in the diffusion-aware MA-RAG pipeline. This helps debug issues **before** tuning diffusion coefficients.

## What Traces Are Captured

### Per-Hop Traces

For each reasoning hop, the debugger captures:

1. **Query Transformations**
   - Proposed query (from step definer)
   - Stabilized query (after regulators)
   - Added/removed terms
   - Transformation ratio

2. **Regulator Decisions**
   - Which regulators fired (Granularity, Entity, Evidence, Relation, Confidence, Plan)
   - Constraint weights
   - Constraint types
   - Summary statistics

3. **Flow State**
   - Entropy H(t)
   - Diffusion coefficient D(t)
   - Confidence
   - Beliefs (probability distribution)
   - Anchor count
   - Plan alignment
   - Drift from previous hop

4. **Retrieval Parameters**
   - Requested k vs actual k
   - Min similarity threshold
   - Total uncertainty (entropy + diffusion)
   - Entropy penalty
   - Diffusion penalty
   - Documents retrieved
   - Top/average similarity scores
   - Adaptive boost status

5. **Anchor Decisions**
   - Created anchors (entity, strength, hop)
   - Rejected anchors (with reasons)
   - Rejection reasons (low confidence, high entropy, invalid term, etc.)

6. **QA Results**
   - Answer
   - Confidence

7. **Convergence Checks**
   - Early termination decisions
   - Convergence conditions met

## Failure Mode Detection

The debugger automatically identifies:

### 1. Low Entropy Issues
- **Symptom**: Entropy < 0.1
- **Impact**: May prevent adaptive retrieval from activating
- **Action**: Check if entropy tracker is working correctly

### 2. High Diffusion Issues
- **Symptom**: Diffusion coefficient > 0.1
- **Impact**: Query instability, reasoning drift
- **Action**: Check regulator weights, query stabilization

### 3. Anchor Rejection Issues
- **Symptom**: Many anchors rejected
- **Impact**: Missing key evidence, reasoning gaps
- **Action**: Review quality thresholds (confidence >= 0.7, entropy <= 0.3)

### 4. Query Transformation Issues
- **Symptom**: Significant query changes (>3 terms added/removed)
- **Impact**: Regulators may be too aggressive
- **Action**: Review regulator weights and constraint merging

### 5. Retrieval Issues
- **Symptom**: Too few documents retrieved (<5)
- **Impact**: Insufficient context for QA
- **Action**: Check k values, similarity thresholds, adaptive retrieval

### 6. Regulator Issues
- **Symptom**: Granularity regulator not firing
- **Impact**: Missing initial condition, hierarchical leakage
- **Action**: Check regulator initialization and application order

## Usage

### Basic Usage

```bash
cd AlgoverseMultiAgent-MCP
python debug_diffusion_pipeline.py --dataset musique --num_examples 5
```

### Arguments

- `--dataset`: Dataset to evaluate (default: `musique`)
- `--num_examples`: Number of examples to debug (default: `5`)

### Output

The script generates:

1. **Console Output**
   - Per-question progress
   - Summary statistics
   - Failure mode analysis

2. **JSON Report** (`results/diffusion_debug_report_YYYYMMDD_HHMMSS.json`)
   - Complete traces for all questions
   - Summary statistics
   - Failure mode details

## Analyzing Results

### Key Metrics to Check

1. **Average Entropy**: Should be > 0.1 for adaptive retrieval to activate
2. **Average Diffusion**: Should be < 0.1 for stable reasoning
3. **Anchor Creation Rate**: Should create anchors for high-quality evidence
4. **Regulator Firing Rate**: Granularity should fire on every hop
5. **Retrieval Coverage**: Should retrieve 10-20 documents per hop

### Common Issues and Fixes

#### Issue: Entropy Always 0.0
- **Cause**: Entropy tracker not updating
- **Fix**: Check `entropy_tracker.update()` calls

#### Issue: Diffusion Penalty Always 0.0
- **Cause**: Flow snapshot not passing diffusion coefficient
- **Fix**: Check `flow_snapshot.diffusion_coefficient` extraction

#### Issue: Adaptive k Not Adjusting
- **Cause**: Low uncertainty, min_boost not applied
- **Fix**: Check `min_boost` in `retriever_agent.py`

#### Issue: Granularity Regulator Not Firing
- **Cause**: Regulator not initialized or not in regulator list
- **Fix**: Check `regulator_manager` initialization

#### Issue: Too Many Anchor Rejections
- **Cause**: Quality thresholds too strict
- **Fix**: Adjust confidence/entropy thresholds in `flow_update.py`

## Next Steps After Debugging

Once you've identified the failure modes:

1. **Fix the root causes** (entropy tracking, regulator weights, etc.)
2. **Re-run the debugger** to verify fixes
3. **Then tune diffusion coefficients** if needed

## Example Output

```
================================================================================
FAILURE MODE ANALYSIS
================================================================================

⚠️  LOW ENTROPY ISSUES: 3 instances
   Entropy < 0.1 may prevent adaptive retrieval from activating
   - Hop 1: H(t)=0.050 - Entropy too low - may not trigger adaptive retrieval
   - Hop 2: H(t)=0.080 - Entropy too low - may not trigger adaptive retrieval

⚠️  RETRIEVAL ISSUES: 2 instances
   Too few documents retrieved may indicate k too low or similarity threshold too high
   - Hop 1: 3 docs (k=10) - Too few documents retrieved
   - Hop 2: 4 docs (k=10) - Too few documents retrieved
```

## Integration with Evaluation

The debugger can be run alongside or instead of regular evaluation:

```bash
# Regular evaluation
python evaluate_datasets.py --dataset musique --num_examples 20

# Debug evaluation (fewer examples, more detail)
python debug_diffusion_pipeline.py --dataset musique --num_examples 5
```

## Tips

1. **Start with 3-5 examples** to get quick feedback
2. **Focus on failed questions** (EM=0.0) first
3. **Check entropy/diffusion evolution** across hops
4. **Compare regulator decisions** between successful and failed questions
5. **Look for patterns** in failure modes across questions

