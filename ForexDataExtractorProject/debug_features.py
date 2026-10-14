
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(r'c:\Users\Administrador\Documents\Proyectos\Trading\ForexDataExtractorProject'))

from src.utils.exploration_utils import engineer_features, generate_target_variable

# Create dummy data
dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
df_dummy = pd.DataFrame({
    'datetime': dates,
    'open': np.random.uniform(1.0, 1.1, 100),
    'high': np.random.uniform(1.1, 1.2, 100),
    'low': np.random.uniform(0.9, 1.0, 100),
    'close': np.random.uniform(0.95, 1.15, 100),
    'volume': np.random.randint(100, 1000, 100),
    'compra_resultado': ['NO_OBJETIVO'] * 100, # Dummy
    'venta_resultado': ['NO_OBJETIVO'] * 100  # Dummy
})

print("--- Testing engineer_features output columns ---")
try:
    df_eng, _ = engineer_features(
        df_dummy,
        n_candles=15,
        n_clusters=5,
        min_pattern_threshold=10,
        fit_clusterer=True
    )
    
    print(f"Total columns generated: {len(df_eng.columns)}")
    print("\nSample Columns:")
    print(df_eng.columns.tolist())
    
    print("\nCheck for candle-related keywords:")
    keywords = ['range', 'body', 'size', 'shadow', 'wick']
    found = [c for c in df_eng.columns if any(k in c.lower() for k in keywords)]
    print(f"Found {len(found)} features matching keywords {keywords}:")
    print(found[:20]) # Print first 20 matches

except Exception as e:
    print(f"Error running engineer_features: {e}")
    import traceback
    traceback.print_exc()
