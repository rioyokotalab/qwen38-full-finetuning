#!/usr/bin/env python3
"""Select one fixed-number project task without loading unrelated records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(reason: str) -> None:
    raise SystemExit(f"TASK_LEDGER status=failed reason={reason}")


def load() -> tuple[dict, list[dict[str, str]]]:
    try:
        cfg = json.loads((ROOT / "docs/tasks/config.json").read_text())
        board = (ROOT / "TODO.md").read_text()
    except (OSError, ValueError):
        fail("ledger-invalid")
    if cfg.get("repository") != "qwen38-full-finetuning" or cfg.get("prefix") != "QFT":
        fail("config-invalid")
    pattern = re.compile(
        r"^\| (QFT-[0-9]{3,}) \| (-?[0-9]+) \| `docs/tasks/\1\.md` \| ([^|]+) \|$"
    )
    rows = []
    for line in board.splitlines():
        if match := pattern.fullmatch(line):
            rows.append({"task": match[1], "priority": match[2], "summary": match[3].strip()})
    if [int(row["priority"]) for row in rows] != sorted(int(row["priority"]) for row in rows):
        fail("board-order")
    return cfg, rows


def record(task: str, limit: int) -> str:
    path = ROOT / "docs/tasks" / f"{task}.md"
    try:
        text = path.read_text()
    except OSError:
        fail(f"record-missing:{task}")
    if f"task: {task}" not in text or "status: open" not in text:
        fail(f"record-identity:{task}")
    if len(text.split()) > limit:
        fail(f"record-oversized:{task}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    select = sub.add_parser("next-task")
    select.add_argument("--task")
    select.add_argument("--context", action="store_true")
    args = parser.parse_args()
    cfg, rows = load()
    if args.command == "validate":
        expected = f"Next free ID: QFT-{int(cfg['next_id']):03d}."
        if expected not in (ROOT / "TODO.md").read_text():
            fail("next-id-mismatch")
        for row in rows:
            record(row["task"], int(cfg["max_active_record_words"]))
        print(f"TASK_LEDGER status=pass repository={cfg['repository']} tasks={len(rows)}")
        return
    candidates = [row for row in rows if not args.task or row["task"] == args.task]
    if not candidates:
        if args.task:
            fail("task-not-executable")
        print("TASK_SELECTION status=idle")
        return
    chosen = candidates[0]
    text = record(chosen["task"], int(cfg["max_active_record_words"]))
    print(
        f"TASK_SELECTION status=selected task={chosen['task']} "
        f"priority={chosen['priority']} record=docs/tasks/{chosen['task']}.md"
    )
    if args.context:
        print(text, end="" if text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
