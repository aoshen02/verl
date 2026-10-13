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
"""Optional sampled avg@4 teacher evaluation; disabled by default for the training run."""

import argparse
import hashlib
import json
from pathlib import Path

from .prepare_models import MODELS
from .reward import compute_score


def prepare_prompts(tokenizer, rows):
    from verl.utils.tokenizer import normalize_token_ids

    prompts = [
        normalize_token_ids(
            tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
        for row in rows
    ]
    if not prompts or any(not prompt or any(type(token) is not int for token in prompt) for prompt in prompts):
        raise ValueError("Expected nonempty lists of integer prompt token IDs")
    if max(map(len, prompts)) > 2048:
        raise ValueError("A source prompt exceeds the configured 2048-token limit")
    return prompts


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["caesar_cipher", "simple_geometry"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()
    path = args.data / f"{args.domain}-dev.parquet"
    rows = pq.read_table(path).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = prepare_prompts(tokenizer, rows)
    print(f"Validated {len(prompts)} {args.domain} prompts; longest={max(map(len, prompts))} tokens", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=32769,
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=42,
    )
    results = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(n=4, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=30720),
    )
    scores = []
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.domain}-predictions.jsonl").open("x") as f:
        for row, result in zip(rows, results, strict=True):
            if len(result.outputs) != 4:
                raise ValueError("Expected four completions per evaluation prompt")
            for sample_index, completion in enumerate(result.outputs):
                score = compute_score(args.domain, completion.text, row["reward_model"]["ground_truth"])["acc"]
                scores.append(score)
                f.write(
                    json.dumps(
                        {
                            "index": row["extra_info"]["index"],
                            "sample_index": sample_index,
                            "score": score,
                            "output_tokens": len(completion.token_ids),
                            "response": completion.text,
                        }
                    )
                    + "\n"
                )
    baseline = {
        "accuracy": sum(scores) / len(scores),
        "count": len(scores),
        "model": MODELS[args.domain],
        "local_model": args.model,
        "dev_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoding": {"temperature": 1.0, "top_p": 1.0, "n": 4, "max_tokens": 30720, "enable_thinking": True},
    }
    (args.output / f"{args.domain}.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
