from pathlib import Path
import json
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_ledger_is_valid_and_idle(self):
        validate = subprocess.run(
            [sys.executable, "tools/task-select.py", "validate"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        selected = subprocess.run(
            [sys.executable, "tools/task-select.py", "next-task"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(selected.returncode, 0)
        self.assertEqual(selected.stdout.strip(), "TASK_SELECTION status=idle")

    def test_model_is_pinned_and_training_is_full_parameter(self):
        lock = dict(
            line.split("=", 1)
            for line in (ROOT / "model.lock").read_text().splitlines()
            if line and not line.startswith("#")
        )
        self.assertEqual(len(lock["MODEL_REVISION"]), 40)
        source = (ROOT / "train_full_sft.py").read_text()
        self.assertIn('parameter.requires_grad_(True)', source)
        self.assertIn('total != trainable', source)
        self.assertIn('"activation_checkpointing": False', source)
        for forbidden in ("/data1/", "rkp00015", "rku00075", "peft import", "bitsandbytes"):
            self.assertNotIn(forbidden, source)

    def test_cluster_example_is_nonsecret_and_smoke_data_is_valid(self):
        config = (ROOT / "config/cluster.env.example").read_text()
        self.assertIn("CHANGE_ME", config)
        self.assertNotIn("/data1/", config)
        rows = [
            json.loads(line)
            for line in (ROOT / "data/smoke.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["messages"] for row in rows))


if __name__ == "__main__":
    unittest.main()
