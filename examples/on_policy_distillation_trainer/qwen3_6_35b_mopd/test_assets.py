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
"""Check tensor fidelity, scoring semantics and the inherited Hydra configuration."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate_teachers import prepare_prompts
from .prepare_models import convert, verify_conversion
from .reward import compute_score


class AssetTests(unittest.TestCase):
    def test_prompt_tokens_normalize_transformers_outputs_and_check_real_length(self):
        from transformers import BatchEncoding

        class Tokenizer:
            def __init__(self, result):
                self.result = result

            def apply_chat_template(self, *args, **kwargs):
                return self.result

        rows = [{"prompt": [{"role": "user", "content": "test"}]}]
        for result in ([1, 2, 3], BatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})):
            self.assertEqual(prepare_prompts(Tokenizer(result), rows), [[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "2048-token limit"):
            prepare_prompts(Tokenizer(BatchEncoding({"input_ids": [1] * 2049})), rows)
        with self.assertRaises(ValueError):
            prepare_prompts(Tokenizer([]), rows)

    def test_checkpoint_repacking_preserves_each_expert(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base, destination = root / "teacher", root / "base", root / "converted"
            source.mkdir()
            base.mkdir()
            prefix = "model.language_model.layers.0.mlp.experts"
            original = {}
            for expert in range(2):
                for projection, shape, offset in [
                    ("gate_proj", (2, 3), 10),
                    ("up_proj", (2, 3), 20),
                    ("down_proj", (3, 2), 30),
                ]:
                    original[f"{prefix}.{expert}.{projection}.weight"] = torch.full(shape, float(expert + offset))
            original["lm_head.weight"] = torch.ones(4, 3)
            base_weights = {
                f"{prefix}.gate_up_proj": torch.zeros(2, 4, 3),
                f"{prefix}.down_proj": torch.zeros(2, 3, 2),
                "lm_head.weight": torch.zeros(4, 3),
                "mtp.unused.weight": torch.full((2, 2), 99.0),
            }
            for path, tensors in [(source, original), (base, base_weights)]:
                save_file(tensors, str(path / "weights.safetensors"))
                (path / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": dict.fromkeys(tensors, "weights.safetensors"),
                        }
                    )
                )
                (path / "config.json").write_text("{}")
            provenance = {"teacher": ["teacher", "revision"], "base": ["base", "revision"]}
            # An interrupted previous conversion must not block another attempt.
            (root / "converted.partial-old").mkdir()
            convert(source, base, destination, shard_bytes=64, provenance=provenance)
            verify_conversion(destination, provenance)
            with self.assertRaises(ValueError):
                verify_conversion(destination, {"teacher": ["teacher", "wrong-revision"]})
            mapping = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(mapping), set(base_weights))
            converted = {}
            for key, shard in mapping.items():
                with safe_open(str(destination / shard), framework="pt") as handle:
                    converted[key] = handle.get_tensor(key)
            for expert in range(2):
                expected = torch.cat(
                    [original[f"{prefix}.{expert}.gate_proj.weight"], original[f"{prefix}.{expert}.up_proj.weight"]],
                    dim=0,
                )
                self.assertTrue(torch.equal(converted[f"{prefix}.gate_up_proj"][expert], expected))
                self.assertTrue(
                    torch.equal(
                        converted[f"{prefix}.down_proj"][expert], original[f"{prefix}.{expert}.down_proj.weight"]
                    )
                )
            self.assertTrue(torch.equal(converted["lm_head.weight"], original["lm_head.weight"]))
            self.assertTrue(torch.equal(converted["mtp.unused.weight"], base_weights["mtp.unused.weight"]))
            with self.assertRaises(FileExistsError):
                convert(source, base, destination)
            shard = destination / next(iter(mapping.values()))
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                verify_conversion(destination, provenance)

    def test_scoring_requires_finished_thinking_and_uses_last_answer(self):
        label = {"task": "caesar_cipher", "answer": "HELLO WORLD", "metadata": {}}

        def score(text):
            return compute_score("caesar_cipher", text, label)["acc"]

        self.assertEqual(score("<answer>HELLO WORLD</answer>"), 0.0)
        self.assertEqual(score("thinking</think><answer>HELLO WORLD</answer>"), 1.0)
        self.assertEqual(score("</think><answer>HELLO WORLD</answer></think><answer>WRONG</answer>"), 0.0)
        with self.assertRaises(ValueError):
            compute_score("simple_geometry", "</think><answer>HELLO WORLD</answer>", label)

    def test_config_inherits_upstream_and_keeps_requested_recipe(self):
        from hydra import compose, initialize_config_dir

        example = Path(__file__).resolve().parent
        root = example.parents[2]
        names = (
            "MOPD_DATA",
            "MOPD_OUTPUT",
            "MOPD_TEACHER_BASELINES",
            "STUDENT_MODEL",
            "CAESAR_TEACHER_MODEL",
            "GEOMETRY_TEACHER_MODEL",
            "VERL_ROOT",
        )
        with patch.dict(os.environ, dict.fromkeys(names, "/tmp/test-placeholder")):
            with initialize_config_dir(version_base=None, config_dir=str(example)):
                config = compose(
                    config_name="config",
                    overrides=[
                        f"hydra.searchpath=[file://{root}/verl/trainer/config]",
                    ],
                )
            self.assertEqual(config.trainer.v1.trainer_mode, "sync")
            self.assertEqual(config.actor_rollout_ref.actor.strategy, "fsdp2")
            self.assertFalse(config.distillation.distillation_loss.use_task_rewards)
            self.assertEqual(config.distillation.distillation_loss.loss_mode, "k1")
            self.assertEqual(config.data.train_batch_size, 128)
            self.assertEqual(config.data.max_prompt_length + config.data.max_response_length, 32768)
            for teacher in (config.distillation.teacher_models.caesar, config.distillation.teacher_models.geometry):
                self.assertEqual(
                    teacher.inference.prompt_length + teacher.inference.response_length + 1,
                    teacher.inference.max_model_len,
                )
            self.assertEqual(config.trainer.total_training_steps, 20)
            self.assertEqual(config.actor_rollout_ref.rollout.n, 1)
            evaluation = config.actor_rollout_ref.rollout.val_kwargs
            self.assertEqual(evaluation.n, 4)
            self.assertTrue(evaluation.do_sample)
            self.assertEqual((evaluation.temperature, evaluation.top_p, evaluation.top_k), (1.0, 1.0, -1))


if __name__ == "__main__":
    unittest.main()
