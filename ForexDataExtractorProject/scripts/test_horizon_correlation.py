import pandas as pd

df = pd.read_parquet("data/processed/features_universe_full_structural.parquet")
print("Correlación de features de vela con trade_net_return (Horizonte de 35 barras / 6 días):")
for col in ["body_ratio", "upper_wick_ratio", "lower_wick_ratio", "total_range", "RSI_14"]:
    if col in df.columns:
        print(f"  * {col:<20}: {df[col].corr(df['trade_net_return']):.5f}")

print("\nCorrelación con retorno de corto plazo (1 a 3 barras):")
ret_1 = df.groupby("ticker")["close"].pct_change().shift(-1)
ret_3 = df.groupby("ticker")["close"].pct_change(3).shift(-3)
for col in ["body_ratio", "upper_wick_ratio", "lower_wick_ratio", "total_range", "RSI_14"]:
    if col in df.columns:
        print(f"  * {col:<20} -> 1 barra: {df[col].corr(ret_1):+.5f} | 3 barras: {df[col].corr(ret_3):+.5f}")
