import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('data/processed/labeled_universe_1hour.parquet').dropna(subset=['target_quality_trade'])

h, l, o, c = df['high'], df['low'], df['open'], df['close']
rng = np.maximum(h - l, 1e-6)
body = np.abs(c - o)

df['candle_body_pct'] = body / rng
df['candle_close_loc'] = (c - l) / rng
df['candle_upper_wick'] = (h - np.maximum(o, c)) / rng
df['candle_lower_wick'] = (np.minimum(o, c) - l) / rng
df['is_bullish_close'] = (c > o).astype(int)

df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['is_power_hour'] = (df['hour'] >= 19).astype(int)
df['is_morning_trap'] = (df['hour'] <= 14).astype(int)

# Señal base
base = (df['dist_to_res20_atr'] < 0.5) & (df['is_market_bullish'] == 1) & (df['rvol'] > 1.2)
sig = df[base]

wins = sig[sig['target_quality_trade'] == 1]
losses = sig[sig['target_quality_trade'] == 0]

print("=== PODER DE DISCRIMINACIÓN DE LAS NUEVAS VARIABLES ===")
new_feats = ['candle_close_loc', 'candle_body_pct', 'candle_upper_wick', 'candle_lower_wick', 'is_power_hour', 'is_morning_trap']
for f in new_feats:
    w = wins[f]
    l = losses[f]
    d = (w.mean() - l.mean()) / (np.sqrt((w.var() + l.var())/2) + 1e-9)
    ks, p = stats.ks_2samp(w, l)
    sig_mark = "SÍ (Predictivo)" if p < 0.05 else "NO"
    print(f"{f:22s} | Wins: {w.mean():.3f} | Losses: {l.mean():.3f} | Cohen d: {d:+.3f} | KS: {ks:.3f} | {sig_mark}")
