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
"""Fetch the reference's immutable data; adapt its schema without regenerating tasks."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REFERENCE_COMMIT = "383c8f49a6f4b350ae2ddf3d4b7e0e92d917c2c3"
BASE_URL = f"https://raw.githubusercontent.com/SemiAnalysisAI/opd-benchmark/{REFERENCE_COMMIT}/data"
FILES = {
    "mixed-train": (15576, "d8c46c8218174820779690e92535a16dea9244558315c4eb3d5c700812c5c278"),
    "caesar_cipher-dev": (126, "c2f97f3f252640c9652faebac523b1f5669bd952ffb92f6890a74e5765297d41"),
    "simple_geometry-dev": (128, "8dc5fc2aa9c3355d8ee3d4b521a997c7863f471723663a9889e9e4db5c5e6b8d"),
}


def convert_rows(raw):
    rows = []
    for index, line in enumerate(raw.decode().splitlines()):
        row = json.loads(line)
        label = json.loads(row["label"])
        domain = label["task"]
        if domain not in {"caesar_cipher", "simple_geometry"}:
            raise ValueError(f"Unknown domain: {domain}")
        rows.append(
            {
                "prompt": row["prompt"],
                "data_source": domain,
                "ability": domain,
                "reward_model": {"style": "rule", "ground_truth": row["label"]},
                "extra_info": {"index": index, **row["metadata"]},
            }
        )
    return rows


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_commit": REFERENCE_COMMIT, "files": {}}
    for stem, (count, expected_hash) in FILES.items():
        with urlopen(f"{BASE_URL}/{stem}.jsonl.gz", timeout=120) as response:
            raw = gzip.decompress(response.read())
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Source data hash mismatch: {stem}")
        rows = convert_rows(raw)
        if len(rows) != count:
            raise ValueError(f"Source data count mismatch: {stem}")
        if stem == "mixed-train":
            for i in range(0, len(rows) - 127, 128):
                if sum(r["data_source"] == "caesar_cipher" for r in rows[i : i + 128]) != 64:
                    raise ValueError("Training data is not balanced 64/64 in each full batch")
        name = "train.parquet" if stem == "mixed-train" else f"{stem}.parquet"
        path = args.output / name
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest["files"][name] = {
            "rows": count,
            "source_sha256": expected_hash,
            "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
