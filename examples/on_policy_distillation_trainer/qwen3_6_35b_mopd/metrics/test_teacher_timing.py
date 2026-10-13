"""CPU contract tests for the timing wrapper; GPU integration needs a new run."""

import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class TimingTests(unittest.TestCase):
    def test_completed_execution_rank_and_startup_boundaries(self):
        calls = []

        class Worker:
            rank = 0

            def execute_model(self, value):
                calls.append("execute")
                return value

        fake_worker = ModuleType("vllm.v1.worker.gpu_worker")
        fake_worker.Worker = Worker
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")))
        spec = importlib.util.spec_from_file_location("timing_under_test", Path(__file__).with_name("teacher_timing.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"torch": fake_torch, "vllm.v1.worker.gpu_worker": fake_worker}):
            spec.loader.exec_module(module)
        with TemporaryDirectory() as directory, patch.dict(os.environ, MOPD_OUTPUT=directory):
            root = Path(directory)
            worker = module.CaesarTeacherWorker()
            self.assertEqual(worker.execute_model(7), 7)
            self.assertEqual(calls, ["execute"])
            (root / "teacher-timing-step.json").write_text(json.dumps({"step": 3, "batch_id": 2}))
            calls.clear()
            with patch.object(module.time, "perf_counter", side_effect=[10, 12]):
                self.assertEqual(worker.execute_model(8), 8)
            self.assertEqual(calls, ["sync", "execute", "sync"])
            trace = root / "teacher-execution-caesar_cipher.jsonl"
            record = json.loads(trace.read_text())
            self.assertEqual(record, {"step": 3, "batch_id": 2, "teacher": "caesar_cipher", "active_s": 2})
            calls.clear()
            worker.rank = 1
            worker.execute_model(9)
            self.assertEqual(calls, ["execute"])
            self.assertEqual(len(trace.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
