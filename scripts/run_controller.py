#!/usr/bin/env python3
"""CLI runner for task controller checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task_controller import TaskController, TaskViolationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Task controller git gate")
    parser.add_argument("--task-file", default=".agent/current_task.json")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--diff-range", default=None)
    parser.add_argument(
        "--mode",
        choices=["pre-commit", "ci", "auto"],
        default="auto",
        help="Controller mode: fast pre-commit checks or strict CI checks",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "pre-commit" if args.staged else "ci"

    controller = TaskController()
    try:
        controller.enforce(
            task_file=Path(args.task_file),
            staged=args.staged,
            diff_range=args.diff_range,
            mode=mode,
        )
    except TaskViolationError as exc:
        print(f"❌ Task controller blocked execution: {exc}")
        return 1

    print("✅ Task controller passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
