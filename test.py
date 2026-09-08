import json

path = r"C:\Users\neela\Downloads\The_Polymath_Initiative\results_rag\rag_results_summaries_observations_full.json"
with open(path) as f:
    data = json.load(f)
keys = set()
for sample in data:
    for qa in sample["qa"]:
        for k in qa.keys():
            if k.endswith("_prediction"):
                keys.add(k)
for k in sorted(keys):
    print(k)
