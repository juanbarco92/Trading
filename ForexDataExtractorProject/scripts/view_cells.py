import json

with open("notebooks/training-model EURUSD.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [3, 4, 5, 6, 7, 8, 9]:
    print(f"\n{'='*70}\nCELL {i}\n{'='*70}")
    src = "".join(nb["cells"][i]["source"])
    print(src)
