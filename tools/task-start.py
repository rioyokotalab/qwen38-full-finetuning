#!/usr/bin/env python3
"""Refresh a clean main from origin, then select its first task."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, capture: bool = False) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if capture else ""


def main() -> None:
    if sys.argv[1:] not in ([], ["--context"]):
        raise SystemExit("TASK_LEDGER status=failed reason=usage")
    try:
        if run("branch", "--show-current", capture=True) != "main":
            raise RuntimeError
        if run("status", "--porcelain", capture=True):
            raise RuntimeError
        run("fetch", "--quiet", "origin", "main")
        run("merge", "--ff-only", "--quiet", "origin/main")
    except (OSError, subprocess.SubprocessError, RuntimeError):
        raise SystemExit("TASK_LEDGER status=failed reason=checkout-stale")
    command = [sys.executable, str(ROOT / "tools/task-select.py"), "next-task"]
    if sys.argv[1:]:
        command.append("--context")
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
