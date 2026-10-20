import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.utils.exploration_utils import engineer_features, fractal_engineering

print("Testing user's feature engine on NVDA...")
df_nvda = pd.read_parquet("data/processed/NVDA_1hour.parquet")
print("Original NVDA shape:", df_nvda.shape)

# Run engineer_features
df_eng, clusterer = engineer_features(
    df_nvda,
    n_candles=4,
    n_clusters=6,
    min_pattern_threshold=50,
    fit_clusterer=True,
    random_state=42
)
print("Shape after engineer_features:", df_eng.shape)
print("Features generated:", df_eng.shape[1] - df_nvda.shape[1])

# Run fractal_engineering
df_fractal = fractal_engineering(df_eng)
print("Shape after fractal_engineering:", df_fractal.shape)
print("Total new columns:", df_fractal.shape[1] - df_nvda.shape[1])
