import json

with open("notebooks/training-model EURUSD.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "TARGET" in src or "pip_objetivo" in src or "OBJETIVO" in src or "pips" in src:
        print(f"=== CELL {i} ===")
        for line in src.split("\n")[:15]:
            print(" ", line)
