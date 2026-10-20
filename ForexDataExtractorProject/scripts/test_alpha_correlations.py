import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/features_universe_full_structural.parquet")

# Calcular retornos vs SPY
spy_df = df[df["ticker"] == "SPY"].sort_values("timestamp").set_index("timestamp")
spy_ret_5d = spy_df["close"].pct_change(35)
df["spy_ret_5d"] = df["timestamp"].map(spy_ret_5d).fillna(0.0)
df["stock_ret_5d"] = df.groupby("ticker")["close"].pct_change(35).fillna(0.0)
df["alpha_5d"] = df["stock_ret_5d"] - df["spy_ret_5d"]

corrs = {}
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and not col.startswith("target_") and not col.startswith("trade_") and not col.startswith("short_"):
        c = df[col].corr(df["trade_net_return"])
        if not np.isnan(c):
            corrs[col] = c

df_c = pd.DataFrame(list(corrs.items()), columns=["Feature", "Corr"]).sort_values("Corr", ascending=False)
print("TOP 15 CORRELACIONES POSITIVAS CON RETORNO (Horizonte 35 barras):")
print(df_c.head(15).to_string(index=False))

print("\nTOP 15 CORRELACIONES NEGATIVAS CON RETORNO:")
print(df_c.tail(15).to_string(index=False))
