import json

file_path = r"c:\Users\Administrador\Documents\Proyectos\Trading\ForexDataExtractorProject\notebooks\training-model EURUSD.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace("sparse_output=False", "sparse=False").replace("sparse_output=True", "sparse=True")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
