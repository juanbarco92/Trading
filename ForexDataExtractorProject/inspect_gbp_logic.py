
import json

notebook_path = r'c:\Users\Administrador\Documents\Proyectos\Trading\ForexDataExtractorProject\notebooks\training-model GBPUSD.ipynb'

def inspect_notebook():
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print("--- SEARCHING FOR 'X_train' CREATION ---")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if "X_train =" in source and "drop" in source:
                    print(f"CELL {i}:")
                    print(source[:500])
                    print("...")
        
        print("\n--- SEARCHING FOR 'engineer_features' CALL ---")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = "".join(cell['source'])
                if "engineer_features(" in source:
                    print(f"CELL {i}:")
                    print(source[:500])
                    print("...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_notebook()
