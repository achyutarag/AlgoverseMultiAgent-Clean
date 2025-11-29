# agents/state_manager/__init__.py
"""
State Manager module for diffusion-aware retrieval control.

This module provides the central controller for the MA-RAG pipeline,
managing state across hops and performing entropy-aware retrieval.
"""

from .core import StateManager, ExecutionState

__all__ = ["StateManager", "ExecutionState"]

# Global state manager instance for backward compatibility
state_manager = StateManager()

