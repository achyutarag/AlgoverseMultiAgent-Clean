# agents/regulators/__init__.py
from .base_regulator import BaseRegulator, RegulatorConstraint
from .granularity_regulator import GranularityRegulator
# ✅ EXPERIMENT 1b: Adding back EntityRegulator
from .entity_regulator import EntityRegulator
# ✅ EXPERIMENT 1b: Removed EvidenceRegulator, RelationRegulator, ConfidenceRegulator - no benefit
# ✅ EXPERIMENT 2: PlanRegulator confirmed as CORE component - restores stability/trajectory control
from .plan_regulator import PlanRegulator
from .regulator_manager import RegulatorManager

__all__ = [
    "BaseRegulator",
    "RegulatorConstraint",
    "GranularityRegulator",
    # ✅ EXPERIMENT 1b: Adding back EntityRegulator
    "EntityRegulator",
    # ✅ EXPERIMENT 1b: Removed EvidenceRegulator, RelationRegulator, ConfidenceRegulator - no benefit
    # ✅ EXPERIMENT 2: PlanRegulator confirmed as CORE component - restores stability/trajectory control
    "PlanRegulator",
    "RegulatorManager"
]

