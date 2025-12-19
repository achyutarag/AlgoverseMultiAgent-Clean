"""
Repo-root bootstrap to force the AlgoverseMultiAgent-MCP package to win import resolution.

Problem:
This repo contains TWO different Python packages named `agents/`:
  1) repo-root `agents/` (legacy)
  2) `AlgoverseMultiAgent-MCP/agents/` (MCP diffusion pipeline)

Python resolves imports by `sys.path` order. If you run scripts from the repo root,
`import agents...` will often bind to the legacy package, bypassing MCP invariants.

Solution:
Call `activate_mcp()` at process startup to prepend `AlgoverseMultiAgent-MCP/` to sys.path,
so `import agents...` resolves to MCP `agents/` deterministically.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def activate_mcp(repo_root: Optional[str] = None) -> str:
    """
    Prepend the MCP project directory to sys.path so MCP `agents/` is imported.

    Returns:
        The absolute path to the MCP directory that was prepended.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.abspath(__file__))

    mcp_dir = os.path.join(repo_root, "AlgoverseMultiAgent-MCP")
    mcp_dir = os.path.abspath(mcp_dir)

    if not os.path.isdir(mcp_dir):
        raise FileNotFoundError(f"AlgoverseMultiAgent-MCP directory not found at: {mcp_dir}")

    # Remove any existing occurrences to avoid duplicates, then prepend.
    sys.path = [p for p in sys.path if os.path.abspath(p) != mcp_dir]
    sys.path.insert(0, mcp_dir)
    return mcp_dir


def assert_using_mcp_agents() -> None:
    """
    Assert that `import agents` resolves to MCP's package, not repo-root legacy.
    """
    import agents  # noqa: F401

    agents_file = getattr(agents, "__file__", "") or ""
    if "AlgoverseMultiAgent-MCP" not in agents_file.replace("\\", "/"):
        raise RuntimeError(
            "Import resolution error: `agents` is not coming from AlgoverseMultiAgent-MCP.\n"
            f"Resolved agents.__file__ = {agents_file}\n"
            "Fix: call mcp_bootstrap.activate_mcp() before importing `agents.*`."
        )


