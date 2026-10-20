import json

with open("notebooks/training-model EURUSD.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    src = "".join(cell["source"])
    print(f"\n{'='*80}\nCELL {i} [{cell_type.upper()}]\n{'='*80}")
    print(src)
