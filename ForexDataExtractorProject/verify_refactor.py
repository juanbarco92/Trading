import pandas as pd
import numpy as np
import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.abspath("src"))

from utils.exploration_utils import create_candle_features_vector, apply_candle_features

# Create dummy dataframe
df = pd.DataFrame({
    'open': np.random.rand(100),
    'high': np.random.rand(100),
    'low': np.random.rand(100),
    'close': np.random.rand(100),
    'body_size': np.random.rand(100),
    'upper_wick': np.random.rand(100),
    'lower_wick': np.random.rand(100),
    'total_range': np.random.rand(100),
    'body_ratio': np.random.rand(100),
    'upper_wick_ratio': np.random.rand(100),
    'lower_wick_ratio': np.random.rand(100)
})

print("Testing create_candle_features_vector...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df_out = create_candle_features_vector(df, n_candles=15)
    print(f"Warnings caught: {len(w)}")
    for warning in w:
        print(f"Warning: {warning.message}")
        print(f"File: {warning.filename}, Line: {warning.lineno}")

print("\nTesting apply_candle_features...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df_out_2 = apply_candle_features(df, n_candles=15)
    print(f"Warnings caught: {len(w)}")
    for warning in w:
        print(f"Warning: {warning.message}")
