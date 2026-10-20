import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.utils.exploration_utils import fractal_engineering

df = pd.read_parquet("data/processed/NVDA_1hour.parquet").head(50).copy()
df_res = fractal_engineering(df)

bull_idx = np.where(df_res["FRACTAL_5_bullish"] == 1)[0]
print(f"Indices donde FRACTAL_5_bullish == 1: {bull_idx[:5]}")

i = bull_idx[0]
print(f"\nAnalizando vela i = {i}:")
print(f"High de vela {i}: {df['high'].iloc[i]}")
print(f"Highs de las siguientes 2 velas (i+1 = {i+1}, i+2 = {i+2}):")
print(f"  * Vela {i+1}: High = {df['high'].iloc[i+1]}")
print(f"  * Vela {i+2}: High = {df['high'].iloc[i+2]}")
max_future = max(df['high'].iloc[i+1], df['high'].iloc[i+2])
print(f"Maximo de las 2 velas futuras: {max_future}")
print(f"¿Es High[{i}] > Max(Highs futuros)? {df['high'].iloc[i] > max_future}")
print(f"\nEN CONCLUSION:")
print(f"En la barra {i}, el modelo ya sabe que FRACTAL_5_bullish = 1.")
print(f"Esto significa que en la barra {i}, el modelo ya conoce que en las barras {i+1} y {i+2} el precio NO va a superar el maximo de la barra {i}.")
print(f"En el mercado real a la barra {i}, las barras {i+1} y {i+2} todavia no existen.")
