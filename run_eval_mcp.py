"""
Convenience entrypoint: run MCP evaluation from the repo root with MCP-first imports.

Usage:
  python run_eval_mcp.py --dataset musique --num_examples 20
"""

from __future__ import annotations

import argparse
import os
import runpy

from mcp_bootstrap import activate_mcp, assert_using_mcp_agents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP evaluation (repo-root wrapper)")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    mcp_dir = activate_mcp()
    assert_using_mcp_agents()

    eval_script = os.path.join(mcp_dir, "evaluate_datasets.py")
    import sys
    sys.argv = ["evaluate_datasets.py", "--dataset", args.dataset, "--num_examples", str(args.num_examples)]
    runpy.run_path(eval_script, run_name="__main__")


if __name__ == "__main__":
    main()


