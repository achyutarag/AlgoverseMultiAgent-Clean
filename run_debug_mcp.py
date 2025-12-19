"""
Convenience entrypoint: run the MCP diffusion debugger from the repo root.

Usage:
  python run_debug_mcp.py --dataset musique --num_examples 5
"""

from __future__ import annotations

import argparse
import os
import runpy

from mcp_bootstrap import activate_mcp, assert_using_mcp_agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP diffusion debugger (repo-root wrapper)")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    mcp_dir = activate_mcp()
    assert_using_mcp_agents()

    # Run the MCP script as __main__ with correct argv semantics.
    debug_script = os.path.join(mcp_dir, "debug_diffusion_pipeline.py")
    import sys
    sys.argv = ["debug_diffusion_pipeline.py", "--dataset", args.dataset, "--num_examples", str(args.num_examples)]
    runpy.run_path(debug_script, run_name="__main__")


if __name__ == "__main__":
    main()


