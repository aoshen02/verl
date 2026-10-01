# Copyright 2026 aoshen02
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

"""Explicit runtime configuration for the Nemotron Megatron-lite engine."""

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config.engine import EngineConfig


@dataclass
class MegatronLiteEngineConfig(EngineConfig):
    strategy: str = "megatron_lite"
    runtime_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy != "megatron_lite" or self.dtype != "bfloat16":
            raise ValueError("Aligned Nemotron requires megatron_lite and BF16")
        if not self.use_remove_padding:
            raise ValueError("Aligned Nemotron requires unpadded actor inputs")
        if self.router_replay.mode != "disabled":
            raise NotImplementedError("Router replay requires a separate integration gate")
