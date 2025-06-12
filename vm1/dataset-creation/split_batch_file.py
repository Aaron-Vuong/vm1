import glob
import json

SPLIT_ELEMENTS = 100

with open("medr1_json/train_split.json_batch_file.json", "r", encoding="utf-8") as f:
    data = json.load(f)

i = 0
split_num = 0
while i < len(data):
    split = []
    print(f"Generating split #{split_num}!")
    for _ in range(SPLIT_ELEMENTS):
        if i >= len(data):
            break
        split.append(data[i])
        i += 1

    split_num += 1
    with open(f"medr1_json/training_batch_file{split_num}.json", "w+", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

# Verify that each is unique, with no duplicates, and has the requested number of elements.
seen_data = []
training_batch_files = glob.glob("medr1_json/training_batch_file*.json")
for batch_file in training_batch_files:
    with open(batch_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Batch file: {batch_file}")
    print(f"\t Num Requests: {len(data)}")
    for req in data:
        assert req not in seen_data
        seen_data.append(req)
