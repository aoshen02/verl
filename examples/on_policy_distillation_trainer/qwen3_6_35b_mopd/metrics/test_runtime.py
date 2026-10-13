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
"""Exercise telemetry dispatch with the same TensorDict metadata as verl."""

import unittest
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_module
from omegaconf import open_dict

from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

from .runtime import AgentLoopManagerTQ, MeasuredAgentLoopManager, validate_teacher_loading


class DispatchTests(unittest.TestCase):
    def test_validation_and_training_metadata(self):
        for validate in (True, False, None):
            with self.subTest(validate=validate):
                prompts = get_tensordict({"uid": ["prompt-a", "prompt-b"]})
                assign_non_tensor_data(prompts, "global_steps", 7)
                if validate is not None:
                    assign_non_tensor_data(prompts, "validate", validate)
                manager = object.__new__(MeasuredAgentLoopManager)
                manager.collector = Mock()
                with (
                    patch.object(AgentLoopManagerTQ, "generate_sequences", return_value=None) as dispatch,
                    patch("ray.get") as wait,
                ):
                    manager.generate_sequences(prompts)
                self.assertIs(dispatch.call_args.args[0], prompts)
                dispatch.assert_called_once()
                if validate:
                    manager.collector.submit.remote.assert_not_called()
                    wait.assert_not_called()
                else:
                    manager.collector.submit.remote.assert_called_once_with(7, ["prompt-a", "prompt-b"])
                    wait.assert_called_once_with(manager.collector.submit.remote.return_value)


class TeacherLoadingTests(unittest.TestCase):
    def test_named_teachers_load_real_weights_and_reject_dummy_default(self):
        with patch.dict(
            "os.environ",
            {
                "STUDENT_MODEL": "/student",
                "CAESAR_TEACHER_MODEL": "/caesar",
                "GEOMETRY_TEACHER_MODEL": "/geometry",
                "MOPD_DATA": "/data",
                "MOPD_OUTPUT": "/output",
                "VERL_ROOT": "/verl",
            },
        ):
            with initialize_config_module(config_module=__package__.rsplit(".", 1)[0], version_base=None):
                config = compose(config_name="config")
            validate_teacher_loading(config)
            for name in ("caesar", "geometry"):
                with self.subTest(teacher=name):
                    bad = config.copy()
                    with open_dict(bad.distillation.teacher_models[name].inference):
                        bad.distillation.teacher_models[name].inference.pop("load_format")
                    with self.assertRaisesRegex(ValueError, "must load checkpoint weights.*dummy"):
                        validate_teacher_loading(bad)


if __name__ == "__main__":
    unittest.main()
