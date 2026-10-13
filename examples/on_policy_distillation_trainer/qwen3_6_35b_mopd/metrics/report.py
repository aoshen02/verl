# Copyright 2026 The verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export the native scalar log as step-aligned CSV, without inventing missing observations."""

import argparse
import csv
import json
from pathlib import Path


def write_table(path, rows, leading_keys):
    keys = leading_keys + sorted({key for row in rows for key in row} - set(leading_keys))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    steps = {}
    with (args.run / "metrics.jsonl").open() as f:
        for line in f:
            event = json.loads(line)
            row = steps.setdefault(event["step"], {"step": event["step"]})
            row.update({key: value for key, value in event["data"].items() if isinstance(value, (int, float))})
    if not steps:
        raise ValueError("No measurements found")
    rows = [steps[step] for step in sorted(steps)]
    write_table(args.run / "metrics.csv", rows, ["step"])
    evaluations = [
        {key: value for key, value in row.items() if key == "step" or key.startswith("eval/")}
        for row in rows
        if any(key.startswith("eval/") for key in row)
    ]
    write_table(args.run / "evaluation.csv", evaluations, ["step"])
    training = [
        {
            key: value
            for key, value in row.items()
            if key in {"step", "perf/mfu/actor", "actor/grad_norm"} or key.startswith(("trainer/", "timing_s/"))
        }
        for row in rows
        if "trainer/step_s" in row
    ]
    write_table(args.run / "trainer.csv", training, ["step"])
    rollouts, teachers = [], []
    batch_path = args.run / "batches.jsonl"
    if batch_path.exists():
        for line in batch_path.open():
            event = json.loads(line)
            # A running trainer can finish another batch after the scalar
            # snapshot above; keep every exported table on that same snapshot.
            if event["step"] not in steps:
                continue
            for batch in event["batches"]:
                rollouts.append({"step": event["step"], **{k: v for k, v in batch.items() if k != "teachers"}})
                for teacher, values in batch["teachers"].items():
                    teachers.append(
                        {"step": event["step"], "batch_id": batch["batch_id"], "teacher": teacher, **values}
                    )
    write_table(args.run / "rollout_batches.csv", rollouts, ["step", "batch_id"])
    write_table(args.run / "teacher_batches.csv", teachers, ["step", "batch_id", "teacher"])


if __name__ == "__main__":
    main()
