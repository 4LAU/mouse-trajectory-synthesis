import json
r = json.load(open("autoresearch_results.json"))
print(f"{len(r)} total experiments")
s = sorted(r, key=lambda x: x["auc"])
for x in s[:10]:
    print(f"  {x['auc']:.4f}  {x['label']}")
