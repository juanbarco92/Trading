import json
import os

notebook_path = r'c:\Users\Administrador\Documents\Proyectos\Trading\ForexDataExtractorProject\notebooks\training-model EURUSD.ipynb'

def update_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: File not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'PIPS_OBJETIVO_GENERACION =' in line:
                    print(f"Found line: {line.strip()}")
                    new_line = 'PIPS_OBJETIVO_GENERACION = 16 # Valor de pips para objetivo\n'
                    new_source.append(new_line)
                    print(f"Replaced with: {new_line.strip()}")
                    updated = True
                elif 'PIPS_STOP_GENERACION =' in line:
                    print(f"Found line: {line.strip()}")
                    new_line = 'PIPS_STOP_GENERACION = 4    # Valor de pips para stop\n'
                    new_source.append(new_line)
                    print(f"Replaced with: {new_line.strip()}")
                    updated = True
                else:
                    new_source.append(line)
            cell['source'] = new_source

    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("No matching lines found to update.")

if __name__ == "__main__":
    update_notebook()
