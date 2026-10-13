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
"""Small, CPU-only checks for accounting errors that would invalidate a benchmark."""

import unittest

from .aggregation import TrainTotals, add_teacher_execution, summarize_batches


class MetricTests(unittest.TestCase):
    def fixtures(self):
        batches = [{"batch_id": 0, "start": 0.0, "uids": ["a", "b", "c"]}]
        requests = {
            "a": {
                "key": "a_0_0",
                "status": "ready",
                "generated_at": 2.0,
                "end": 5.0,
                "input_tokens": 10,
                "output_tokens": 100,
                "loss_tokens": 80,
                "generation_latency_s": 2.0,
                "e2e_latency_s": 5.0,
                "teacher": "caesar_cipher",
                "teacher_start": 2.0,
                "teacher_end": 5.0,
                "teacher_latency_s": 3.0,
            },
            "b": {
                "key": "b_0_0",
                "status": "ready",
                "generated_at": 10.0,
                "end": 12.0,
                "input_tokens": 20,
                "output_tokens": 300,
                "loss_tokens": 300,
                "generation_latency_s": 10.0,
                "e2e_latency_s": 12.0,
                "teacher": "caesar_cipher",
                "teacher_start": 4.0,
                "teacher_end": 12.0,
                "teacher_latency_s": 8.0,
            },
            "c": {"key": "c_0_0", "status": "failure", "end": 3.0},
        }
        return batches, requests

    def test_makespan_filtering_and_teacher_overlap(self):
        batches, requests = self.fixtures()
        scalars, records = summarize_batches(
            batches,
            requests,
            ["a_0_0"],
            8,
            {"caesar_cipher": 4},
            lambda lengths: sum(lengths) * 6e12,
            100,
        )
        row = records[0]
        self.assertEqual(row["output_tok_s_gpu"], 400 / 10 / 8)
        self.assertEqual(row["effective_output_tok_s_gpu"], 80 / 10 / 8)
        self.assertEqual(row["filtered_ratio"], 0.5)
        self.assertAlmostEqual(row["failure_ratio"], 1 / 3)
        self.assertEqual(row["unknown_output_requests"], 1)
        self.assertEqual(scalars["rollout/request/generation_latency_s_p50"], 6)
        teacher = row["teachers"]["caesar_cipher"]
        self.assertEqual(teacher["window_s"], 10)  # overlapping request durations must not be summed
        self.assertEqual(teacher["scored_tok_s_gpu"], 430 / 10 / 4)
        self.assertAlmostEqual(teacher["mfu_estimate"], 430 * 2 / (10 * 4 * 100))

    def test_unfinished_batch_is_not_reported_as_complete(self):
        batches, requests = self.fixtures()
        requests["c"]["status"] = "pending"
        with self.assertRaises(ValueError):
            summarize_batches(batches, requests, [], 8, {}, lambda lengths: 1, 100)

    def test_time_weighted_run_average(self):
        totals = TrainTotals()
        totals.update(1, 1000, 1e12, 8, 100)
        result = totals.update(9, 1000, 1e12, 8, 100)
        self.assertEqual(result["trainer/train_tok_s_gpu_run"], 2000 / 10 / 8)
        self.assertEqual(result["trainer/train_s_mean"], 5)
        self.assertEqual(result["trainer/train_mfu_estimate_run"], 2 / (10 * 8 * 100))

    def test_execution_accounting_and_step_isolation(self):
        records = [{"batch_id": 2, "teachers": {"caesar_cipher": {"window_s": 10}}}]
        events = [
            {"step": step, "batch_id": 2, "teacher": "caesar_cipher", "active_s": duration}
            for step, duration in [(1, 100), (2, 1.5), (2, 2.5)]
        ]
        scalars = add_teacher_execution(records, events, 2)
        teacher = records[0]["teachers"]["caesar_cipher"]
        self.assertEqual(teacher["active_s"], 4)
        self.assertEqual(teacher["wait_s"], 6)
        self.assertEqual(teacher["execution_calls"], 2)
        self.assertEqual(scalars["teacher/caesar_cipher/batch/active_s_p50"], 4)
        with self.assertRaises(ValueError):
            add_teacher_execution(records, [], 2)
        with self.assertRaises(ValueError):
            add_teacher_execution(records, events, 1)


if __name__ == "__main__":
    unittest.main()
