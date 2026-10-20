import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import pandas as pd
from src.utils.stock_sr import compute_stock_support_resistance_features

df = pd.read_parquet("data/processed/NVDA_1hour.parquet")
t0 = time.time()
res = compute_stock_support_resistance_features(df)
sr_cols = [c for c in res.columns if c.startswith("sr_")]
print(f"Calculated in {time.time()-t0:.2f}s! Added {len(sr_cols)} S/R cols:")
print(res[sr_cols].tail(5))
