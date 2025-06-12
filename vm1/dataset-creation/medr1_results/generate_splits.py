import json

TRAIN_PERCENT=0.5

with open("medr1_fails.json", "r", encoding="utf-8") as f:
    data = json.load(f)

count = {}
train_count = {}
failed = []

test_split = []
train_split = []

def get_category(r):
    return r["id"].split("_test-json")[0]

# Count up all of the different categories.
for r in data:
    category = get_category(r)
    if not count.get(category, None):
        count[category] = 0
        train_count[category] = 0
    count[category] += 1

# Split the generated data in two splits (test/train).
for r in data:
    category = get_category(r)
    if (train_count[category] / count[category]) < TRAIN_PERCENT:
        train_count[category] += 1
        train_split.append(r["question"])
    else:
        test_split.append(r["question"])

print(f"Training dataset: {train_count}")

with open("test_split.json", "w+", encoding="utf-8") as f:
    json.dump(test_split, f, indent=2)
with open("train_split.json", "w+", encoding="utf-8") as f:
    json.dump(train_split, f, indent=2)
