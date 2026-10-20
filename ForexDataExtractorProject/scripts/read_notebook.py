import json

with open("notebooks/training-model EURUSD.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Total cells in training-model EURUSD.ipynb:", len(nb["cells"]))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print("Total code cells:", len(code_cells))

for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    lines = [l.strip() for l in src.split("\n") if l.strip() and not l.strip().startswith("#")]
    header = lines[0] if lines else "empty"
    keywords = ["roc_auc", "decil", "LGBM", "LightGBM", "xgboost", "model", "feature", "predict_proba", "quantile", "target", "score", "precision"]
    if any(k.lower() in src.lower() for k in keywords):
        print(f"Cell {i}: {header[:90]}")
