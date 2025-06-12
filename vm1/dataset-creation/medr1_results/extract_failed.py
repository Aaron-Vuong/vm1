import json

with open("output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total Med-R1 runs: {len(data['results'])}")

count = {}
failed = []
for r in data["results"]:
    if r["match"] is False:
        failed.append(r)
        category = r["id"].split("_test-json")[0]
        if not count.get(category, None):
            count[category] = 0
        count[category] += 1

print(f"Number of Med-R1 fails: {count}")

with open("medr1_fails.json", "w+", encoding="utf-8") as f:
    json.dump(failed, f, indent=2)
