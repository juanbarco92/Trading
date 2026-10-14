import json

file_path = r"c:\Users\Administrador\Documents\Proyectos\Trading\ForexDataExtractorProject\notebooks\training-model EURUSD.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "PerformanceWarning" in line:
        print(f"Line {i}: {line.strip()[:200]}")
