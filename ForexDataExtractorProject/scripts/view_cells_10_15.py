import json

with open("notebooks/training-model EURUSD.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(11, min(16, len(nb["cells"]))):
    print(f"\n{'='*70}\nCELL {i}\n{'='*70}")
    src = "".join(nb["cells"][i]["source"])
    print("SOURCE:\n", src[:300])
    for out in nb["cells"][i].get("outputs", []):
        if "text" in out:
            print("OUTPUT:\n", "".join(out["text"])[:800])

