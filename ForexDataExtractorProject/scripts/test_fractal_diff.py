import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.exploration_utils import fractal_engineering

df_raw = pd.read_parquet("data/processed/NVDA_1hour.parquet").head(600).copy()
df1 = fractal_engineering(df_raw.copy())

df_pert = df_raw.copy()
df_pert.loc[300:, 'high'] = df_pert.loc[300:, 'high'] * 2.0
df_pert.loc[300:, 'close'] = df_pert.loc[300:, 'close'] * 2.0
df2 = fractal_engineering(df_pert.copy())

diffs = {}
for c in [x for x in df1.columns if x.startswith('FRACTAL')]:
    s1 = df1[c].iloc[:300].fillna(-9999)
    s2 = df2[c].iloc[:300].fillna(-9999)
    d = (s1 != s2).sum()
    if d > 0:
        diffs[c] = d

print("Fractal diffs in bars < 300:", len(diffs))
for k, v in list(diffs.items())[:10]:
    print(f"  * {k}: {v} bars changed before bar 300")
