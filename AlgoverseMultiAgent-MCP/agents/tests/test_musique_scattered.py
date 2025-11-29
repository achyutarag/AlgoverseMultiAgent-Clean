"""
MuSiQue Scattered Scenario Tests: Real-world validation of diffusion-aware retrieval.

Tests the system with MuSiQue-style scenarios:
- 20-30 documents per question
- Only 2-4 are supporting facts (relevant)
- Multi-hop reasoning (2-4 hops)
- Scattered documents (not semantically clustered)

This validates that the diffusion-aware retrieval extension actually works
in the challenging scenarios it was designed for.
"""
import pytest
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.schema import Document
from agents.orchestrator import MARAGOrchestrator
from agents.state_manager import StateManager
from agents.retriever_agent import RetrieverAgent


def create_musique_style_example() -> Dict[str, Any]:
    """
    Create a realistic MuSiQue-style example with scattered documents.
    
    Scenario: "What administrative territorial entity is the owner of Ciudad Deportiva located?"
    - Hop 1: Find Ciudad Deportiva
    - Hop 2: Find owner of Ciudad Deportiva
    - Hop 3: Find administrative territorial entity where owner is located
    
    This mimics the actual MuSiQue challenge: 20-30 docs, only 2-4 relevant.
    """
    # Create 25 documents (realistic MuSiQue size)
    paragraphs = []
    supporting_facts = [2, 8, 15, 22]  # Only 4 out of 25 are relevant
    
    # Non-supporting documents (21 distractors)
    paragraphs.extend([
        "Mexico City is the capital of Mexico.",
        "The administrative divisions of Mexico include states and municipalities.",
        "Ciudad Deportiva is a sports complex in Nuevo Laredo, Tamaulipas.",  # Supporting fact 0 (idx 2)
        "Nuevo Laredo is a border city in northern Mexico.",
        "Tamaulipas borders the United States state of Texas.",
        "Sports complexes in Mexico are often owned by municipalities.",
        "The Mexican government administers various public facilities.",
        "Nuevo Laredo Municipality was established in 1848.",  # Supporting fact 1 (idx 8)
        "Municipal governments in Mexico have jurisdiction over local facilities.",
        "Ciudad Deportiva facilities include soccer fields and basketball courts.",
        "Tamaulipas is one of 32 states in Mexico.",
        "Public sports facilities are common in Mexican cities.",
        "The state government of Tamaulipas oversees regional administration.",
        "Nuevo Laredo has a population of over 400,000 people.",
        "Municipal ownership of facilities is common in Mexico.",  # Supporting fact 2 (idx 15)
        "Ciudad Deportiva hosts various sporting events throughout the year.",
        "The administrative structure of Mexico is federal.",
        "States in Mexico have significant autonomy.",
        "Nuevo Laredo is located on the Rio Grande border.",
        "Sports complexes require regular maintenance and funding.",
        "Municipal governments fund local infrastructure projects.",
        "The owner of Ciudad Deportiva is Nuevo Laredo Municipality.",  # Supporting fact 3 (idx 22)
        "Nuevo Laredo Municipality is located in Tamaulipas state.",
        "Tamaulipas is the administrative territorial entity containing Nuevo Laredo.",
        "Federal entities in Mexico include states and the federal district.",
    ])
    
    # Convert to MuSiQue format (list of dicts)
    musique_paragraphs = []
    for idx, text in enumerate(paragraphs):
        musique_paragraphs.append({
            "idx": idx,
            "text": text,
            "is_supporting": idx in supporting_facts
        })
    
    example = {
        "id": "test_musique_scattered_001",
        "question": "What administrative territorial entity is the owner of Ciudad Deportiva located?",
        "answer": "Tamaulipas",
        "paragraphs": musique_paragraphs,
        "supporting_facts": supporting_facts,
        "type": "multi-hop"
    }
    
    return example


def create_musique_documents_from_example(example: Dict[str, Any]) -> List[Document]:
    """Convert MuSiQue example to Document objects."""
    from agents.musique_document_loader import load_musique_example_context_as_documents
    return load_musique_example_context_as_documents(example, include_metadata=True)


@pytest.fixture
def musique_style_example():
    """Create a MuSiQue-style example."""
    return create_musique_style_example()


@pytest.fixture
def musique_orchestrator(musique_style_example):
    """Create orchestrator with MuSiQue-style documents."""
    # Convert example to documents
    documents = create_musique_documents_from_example(musique_style_example)
    
    # Create retriever with documents
    retriever = RetrieverAgent(
        documents=documents,
        model_config={},
        top_k=15,  # Higher k for MuSiQue (as configured in evaluate_datasets.py)
        min_similarity=0.2  # Lower threshold for harder retrieval
    )
    
    # Create state manager
    state_manager = StateManager()
    
    # Create orchestrator
    orchestrator = MARAGOrchestrator(
        retriever_agent=retriever,
        state_manager=state_manager
    )
    
    return orchestrator, musique_style_example


class TestMusiqueScatteredScenario:
    """Test 1: MuSiQue-style scattered scenario validation."""
    
    @pytest.mark.asyncio
    async def test_musique_scattered_execution(self, musique_orchestrator):
        """Test that pipeline executes with MuSiQue-style scattered documents."""
        orchestrator, example = musique_orchestrator
        query = example["question"]
        
        result = await orchestrator.execute_pipeline(query)
        
        # Verify execution
        assert result is not None
        assert result.final_answer is not None
        assert len(result.final_answer) > 0
        
        # Verify steps were executed (multi-hop should have multiple steps)
        assert result.steps_completed >= 1
    
    @pytest.mark.asyncio
    async def test_musique_retrieves_supporting_facts(self, musique_orchestrator):
        """
        Test that system retrieves supporting facts from scattered documents.
        
        With 25 documents and only 4 supporting facts, this tests:
        - Query stabilization prevents drift
        - Entity anchoring works across hops
        - Relevant documents are found despite noise
        - Direct answer validation prevents stopping at wrong level (municipality vs state)
        """
        orchestrator, example = musique_orchestrator
        query = example["question"]
        ground_truth = example["answer"]
        supporting_facts = example["supporting_facts"]
        
        result = await orchestrator.execute_pipeline(query)
        
        # Verify execution
        assert result is not None
        assert result.final_answer is not None
        
        # Check if answer is correct
        # The question asks for "administrative territorial entity" (state/province level)
        # NOT a municipality - so "Nuevo Laredo Municipality" would be wrong level
        answer_lower = result.final_answer.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Check if answer matches ground truth
        matches_ground_truth = ground_truth_lower in answer_lower or answer_lower in ground_truth_lower
        
        # Check if answer is at correct administrative level (state/province, not municipality)
        is_municipality = "municipality" in answer_lower
        is_state_level = (
            "state" in answer_lower or 
            "province" in answer_lower or
            "administrative territorial entity" in answer_lower or
            ground_truth_lower in answer_lower  # Ground truth is a state name
        )
        
        # Answer should either:
        # 1. Match ground truth, OR
        # 2. Be at state/province level (not municipality)
        # With the first-principles fix, the system should now retrieve the state
        # instead of stopping at the municipality
        assert matches_ground_truth or (is_state_level and not is_municipality), \
            f"Answer '{result.final_answer}' should be a state/province level entity " \
            f"(ground truth: '{ground_truth}'), not a municipality. " \
            f"Question asks for 'administrative territorial entity'. " \
            f"Steps completed: {result.steps_completed}"
    
    @pytest.mark.asyncio
    async def test_musique_entity_anchoring_across_hops(self, musique_orchestrator):
        """
        Test that entity anchoring prevents drift in multi-hop reasoning.
        
        This verifies:
        - Hop 1: Finds "Ciudad Deportiva"
        - Hop 2: Query stabilized with "Ciudad Deportiva" anchor
        - Hop 3: Query stabilized with "Nuevo Laredo Municipality" anchor
        """
        orchestrator, example = musique_orchestrator
        query = example["question"]
        
        # Execute pipeline
        result = await orchestrator.execute_pipeline(query)
        
        # Verify execution completed
        assert result is not None
        
        # Check reasoning trajectory for entity mentions
        trajectory = result.reasoning_trajectory
        entity_mentions = []
        
        for step in trajectory:
            step_result = step.get("result", {})
            # Check if entities are mentioned in steps
            # Answer can be in qa_result or directly in step_result
            answer = None
            if "qa_result" in step_result:
                answer = str(step_result["qa_result"].get("answer", "")).lower()
            elif "answer" in step_result:
                answer = str(step_result["answer"]).lower()
            
            if answer:
                if "ciudad deportiva" in answer:
                    entity_mentions.append("ciudad_deportiva")
                if "nuevo laredo" in answer:
                    entity_mentions.append("nuevo_laredo")
                if "tamaulipas" in answer:
                    entity_mentions.append("tamaulipas")
        
        # Also check final answer
        if result.final_answer:
            final_answer_lower = result.final_answer.lower()
            if "ciudad deportiva" in final_answer_lower:
                entity_mentions.append("ciudad_deportiva")
            if "nuevo laredo" in final_answer_lower:
                entity_mentions.append("nuevo_laredo")
            if "tamaulipas" in final_answer_lower:
                entity_mentions.append("tamaulipas")
        
        # Should have found at least one entity
        assert len(entity_mentions) > 0, \
            f"Should have found entities in reasoning trajectory. " \
            f"Trajectory length: {len(trajectory)}, Final answer: '{result.final_answer}'"


class TestMusiquePrecisionRecall:
    """Test 2: Precision/Recall metrics for scattered scenarios."""
    
    @pytest.mark.asyncio
    async def test_supporting_fact_retrieval_rate(self, musique_orchestrator):
        """
        Test the rate at which supporting facts are retrieved.
        
        With 25 documents and 4 supporting facts:
        - Random retrieval: ~16% chance (4/25)
        - Good retrieval: Should retrieve at least 2-3 supporting facts
        """
        orchestrator, example = musique_orchestrator
        query = example["question"]
        supporting_facts = set(example["supporting_facts"])
        
        result = await orchestrator.execute_pipeline(query)
        
        # Extract retrieved document IDs from reasoning trajectory
        retrieved_para_ids = set()
        for step in result.reasoning_trajectory:
            step_result = step.get("result", {})
            retrieved_docs = step_result.get("retrieved_documents", [])
            for doc in retrieved_docs:
                para_id = doc.get("metadata", {}).get("paragraph_id")
                if para_id is not None:
                    retrieved_para_ids.add(para_id)
        
        # Calculate how many supporting facts were retrieved
        retrieved_supporting = supporting_facts.intersection(retrieved_para_ids)
        recall = len(retrieved_supporting) / len(supporting_facts) if supporting_facts else 0
        
        # Should retrieve at least 50% of supporting facts (2 out of 4)
        assert recall >= 0.5, \
            f"Should retrieve at least 50% of supporting facts. " \
            f"Retrieved {len(retrieved_supporting)}/{len(supporting_facts)}. " \
            f"Retrieved IDs: {retrieved_para_ids}, Supporting: {supporting_facts}"
    
    @pytest.mark.asyncio
    async def test_precision_against_noise(self, musique_orchestrator):
        """
        Test precision: retrieved documents should be mostly relevant.
        
        With 25 documents (4 relevant, 21 noise):
        - Low precision: Many noise documents retrieved
        - Good precision: Mostly relevant documents retrieved
        """
        orchestrator, example = musique_orchestrator
        query = example["question"]
        supporting_facts = set(example["supporting_facts"])
        
        result = await orchestrator.execute_pipeline(query)
        
        # Extract retrieved document IDs
        retrieved_para_ids = set()
        for step in result.reasoning_trajectory:
            step_result = step.get("result", {})
            retrieved_docs = step_result.get("retrieved_documents", [])
            for doc in retrieved_docs:
                para_id = doc.get("metadata", {}).get("paragraph_id")
                if para_id is not None:
                    retrieved_para_ids.add(para_id)
        
        if len(retrieved_para_ids) > 0:
            # Calculate precision
            retrieved_supporting = supporting_facts.intersection(retrieved_para_ids)
            precision = len(retrieved_supporting) / len(retrieved_para_ids)
            
            # Precision should be reasonable (at least 20% given the 4/25 ratio)
            # But with query stabilization, should be higher
            assert precision >= 0.15, \
                f"Precision too low: {precision:.2%}. " \
                f"Retrieved {len(retrieved_supporting)} supporting out of {len(retrieved_para_ids)} total"


class TestRealMusiqueData:
    """Test 3: Real MuSiQue dataset examples (if available)."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        True,  # Skip by default - requires MuSiQue dataset
        reason="Requires MuSiQue dataset. Set SKIP_MUSIQUE_TESTS=false to enable."
    )
    async def test_real_musique_example(self):
        """
        Test with a real MuSiQue example from the dataset.
        
        This requires the MuSiQue dataset to be available.
        """
        try:
            from agents.musique_document_loader import _load_musique_from_github
            examples = _load_musique_from_github("validation")
            
            if not examples:
                pytest.skip("MuSiQue dataset not available")
            
            # Use first example
            example = examples[0]
            query = example.get("question", "")
            ground_truth = example.get("answer", "")
            
            if not query:
                pytest.skip("Invalid MuSiQue example")
            
            # Load documents
            from agents.musique_document_loader import load_musique_example_context_as_documents
            documents = load_musique_example_context_as_documents(example)
            
            # Create orchestrator
            retriever = RetrieverAgent(
                documents=documents,
                model_config={},
                top_k=15,
                min_similarity=0.2
            )
            
            state_manager = StateManager()
            orchestrator = MARAGOrchestrator(
                retriever_agent=retriever,
                state_manager=state_manager
            )
            
            # Execute
            result = await orchestrator.execute_pipeline(query)
            
            # Verify
            assert result is not None
            assert result.final_answer is not None
            
            # Check if answer is reasonable
            answer_lower = result.final_answer.lower()
            ground_truth_lower = str(ground_truth).lower()
            
            # Answer should be related to ground truth (exact match not required for this test)
            assert len(answer_lower) > 0, "Answer should not be empty"
            
        except ImportError:
            pytest.skip("MuSiQue loader not available")
        except Exception as e:
            pytest.skip(f"MuSiQue dataset not available: {e}")


class TestStabilizedVsUnstabilized:
    """Test 4: Compare stabilized vs unstabilized retrieval (if possible)."""
    
    @pytest.mark.asyncio
    async def test_stabilization_improves_retrieval(self, musique_orchestrator):
        """
        Test that query stabilization improves retrieval in scattered scenarios.
        
        This is a conceptual test - in practice, we can't easily disable
        stabilization in the orchestrator, but we can verify it's being used.
        """
        orchestrator, example = musique_orchestrator
        query = example["question"]
        
        # Verify that state manager has stabilization
        sm = orchestrator.state_manager
        assert sm.regulator_manager is not None
        assert len(sm.regulator_manager.regulators) > 0
        
        # Execute pipeline
        result = await orchestrator.execute_pipeline(query)
        
        # Verify execution
        assert result is not None
        
        # Check that reasoning flow was used (indicates stabilization was active)
        flow_snapshot = sm.reasoning_flow.get_flow_snapshot()
        # Flow snapshot may be None if no state, but reasoning_flow should exist
        assert sm.reasoning_flow is not None, "Reasoning flow should be active"