"""Example-local vLLM worker instrumentation for synchronous, single-replica teachers.

Time completed execute_model calls, including CPU dispatch and TP communication.
This is synchronized execution wall time, not a sum of GPU kernel durations.
"""

import json
import os
import time
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker


class TimedTeacherWorker(Worker):
    domain = None

    def execute_model(self, *args, **kwargs):
        # TP rank zero represents this one replica. Other ranks execute normally.
        if self.rank != 0:
            return super().execute_model(*args, **kwargs)
        root = Path(os.environ["MOPD_OUTPUT"])
        marker = root / "teacher-timing-step.json"
        if not marker.exists():  # Initialization/profiling is outside scoring.
            return super().execute_model(*args, **kwargs)
        batch = json.loads(marker.read_text())
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = super().execute_model(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = {**batch, "teacher": self.domain, "active_s": elapsed}
        # Flush before returning the completed model result. At batch completion
        # the trainer can therefore read all records without racing a writer.
        with (root / f"teacher-execution-{self.domain}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return result


class CaesarTeacherWorker(TimedTeacherWorker):
    domain = "caesar_cipher"


class GeometryTeacherWorker(TimedTeacherWorker):
    domain = "simple_geometry"
