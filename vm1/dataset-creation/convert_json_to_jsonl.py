import glob
import json
import jsonlines
import os

batch_files = glob.glob("verified_reasoning_traces.json")

OUTPUT_DIR="jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for f in batch_files:
    with open(f, "r", encoding="utf-8") as input_f:
        data = json.load(input_f)

    with jsonlines.open(f"{OUTPUT_DIR}/{f}l", "w") as writer:
        writer.write_all(data)
