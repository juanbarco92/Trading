"""
Análisis Forense de Microestructura de 10 Minutos en Breakdowns Estructurales
Investiga si la anatomía interna de la hora (las 6 velas de 10 minutos)
discrimina los verdaderos colapsos institucionales de las trampas de mercado.
"""

import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_model_2a_breakdown_with_structural_sr import (
    extract_breakdown_with_structural_sr, PARQUET_IBKR
)

def parse_ibkr_date(date_str: str) -> pd.Timestamp:
    try:
        parts = str(date_str).strip().split()
        if len(parts) == 2:
            d_part, t_part = parts[0], parts[1]
            return pd.to_datetime(f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:8]} {t_part}", utc=True)
        elif len(parts) == 1 and len(parts[0]) == 8:
            d_part = parts[0]
            return pd.to_datetime(f"{d_part[:4]}-{d_part[4:6]}-{d_part[6:8]} 16:00:00", utc=True)
    except Exception:
        pass
    return pd.to_datetime(date_str, utc=True)

def analyze_10min_microstructure():
    print("=" * 90)
    print("INVESTIGANDO MICROESTRUCTURA DE 10 MINUTOS DENTRO DE LAS VELAS DE BREAKDOWN")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = extract_breakdown_with_structural_sr(df_raw)
    
    # Identificar qué tickers tienen datos de 10 minutos disponibles
    p_ibkr = PROJECT_ROOT / "data_ibkr"
    available_10m = set()
    for d in p_ibkr.iterdir():
        if d.is_dir() and (d / "10mins").exists() and any((d / "10mins").glob("*.csv")):
            available_10m.add(d.name)
            
    print(f"\nTickers con datos de 10 minutos disponibles: {len(available_10m)}")
    
    # Filtrar eventos de breakdown en tickers que tienen 10m
    events_10m = df_events[df_events["ticker"].isin(available_10m)].copy().reset_index(drop=True)
    print(f"Eventos de Breakdown en tickers con 10m: {len(events_10m)} (2020 - 2026)")
    
    # Cachear datasets de 10 minutos por ticker
    print("\nCargando y parseando datos de 10 minutos para los tickers con eventos...")
    data_10m_by_tk = {}
    
    for tk in events_10m["ticker"].unique():
        csv_files = glob.glob(str(p_ibkr / tk / "10mins" / "*.csv"))
        if not csv_files:
            continue
        dfs = []
        for f in csv_files:
            try:
                m_df = pd.read_csv(f)
                if not m_df.empty and "date" in m_df.columns:
                    dfs.append(m_df)
            except Exception:
                continue
        if dfs:
            cat_df = pd.concat(dfs, ignore_index=True)
            cat_df = cat_df.dropna(subset=["date", "close", "open"])
            cat_df["timestamp"] = cat_df["date"].apply(parse_ibkr_date)
            cat_df = cat_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
            for c in ["open", "high", "low", "close", "volume", "barCount", "wap"]:
                if c in cat_df.columns:
                    cat_df[c] = pd.to_numeric(cat_df[c], errors="coerce")
            data_10m_by_tk[tk] = cat_df
            
    print(f"[OK] {len(data_10m_by_tk)} tickers cargados en memoria con datos de 10 minutos.")
    
    # Extraer métricas de microestructura de 10 minutos para cada vela horaria de quiebre
    print("\nExtrayendo variables de microestructura interna dentro de la vela horaria...")
    micro_records = []
    
    for _, row in events_10m.iterrows():
        tk = row["ticker"]
        t_1h = row["timestamp"]
        
        tdf_10m = data_10m_by_tk.get(tk)
        if tdf_10m is None:
            continue
            
        # Ventana de 1 hora: desde t_1h hasta t_1h + 60 mins
        # En velas horarias de IBKR, el timestamp de la vela representa el inicio de la hora
        t_end = t_1h + pd.Timedelta(minutes=60)
        sub_10m = tdf_10m[(tdf_10m["timestamp"] >= t_1h) & (tdf_10m["timestamp"] < t_end)].sort_values("timestamp").reset_index(drop=True)
        
        # Necesitamos que haya al menos 4 de las 6 velas de 10 min
        if len(sub_10m) < 4:
            continue
            
        c10 = sub_10m["close"].to_numpy()
        o10 = sub_10m["open"].to_numpy()
        h10 = sub_10m["high"].to_numpy()
        l10 = sub_10m["low"].to_numpy()
        v10 = sub_10m["volume"].to_numpy()
        bcount10 = sub_10m["barCount"].to_numpy() if "barCount" in sub_10m.columns else np.ones(len(sub_10m))
        wap10 = sub_10m["wap"].to_numpy() if "wap" in sub_10m.columns else c10
        
        # 1. Proporción de velas de 10 min que fueron rojas (presión vendedora implacable)
        red_bars_ratio = np.mean(c10 < o10)
        
        # 2. Concentración de volumen en los primeros 20 minutos (primeras 2 velas de 10m)
        tot_vol = np.sum(v10) + 1e-6
        early_vol_ratio = np.sum(v10[:2]) / tot_vol
        
        # 3. Aceleración del conteo de órdenes (trades) en los últimos 20 minutos
        tot_trades = np.sum(bcount10) + 1e-6
        late_trades_ratio = np.sum(bcount10[-2:]) / tot_trades
        
        # 4. Comportamiento frente a VWAP (WAP): ¿el precio cerró por debajo del WAP de los 10 minutos?
        below_wap_ratio = np.mean(c10 <= wap10)
        
        # 5. Máximo rebote intradiario dentro de la hora (% desde el mínimo más bajo de 10m hasta el cierre)
        min_intra = np.min(l10)
        intra_bounce = (c10[-1] - min_intra) / (min_intra + 1e-6) * 100.0
        
        # 6. Tendencia neta del WAP a lo largo de la hora (slope del WAP)
        wap_slope = (wap10[-1] - wap10[0]) / (wap10[0] + 1e-6) * 100.0
        
        micro_records.append({
            "event_id": row["event_id"],
            "ticker": tk,
            "timestamp": t_1h,
            "year": row["year"],
            "target_2a": row["target_2a"], # 1 = Quiebre genuino, 0 = Trampa
            "entry_p": row["entry_p"],
            "atr_d": row["atr_d"],
            
            # Variables de microestructura de 10 minutos
            "red_bars_ratio_10m": red_bars_ratio,
            "early_vol_ratio_10m": early_vol_ratio,
            "late_trades_ratio_10m": late_trades_ratio,
            "below_wap_ratio_10m": below_wap_ratio,
            "intra_bounce_10m": intra_bounce,
            "wap_slope_10m": wap_slope
        })
        
    df_micro = pd.DataFrame(micro_records)
    print(f"[OK] Microestructura de 10 minutos extraída para {len(df_micro):,} eventos de breakdown.")
    
    if len(df_micro) < 50:
        print("[AVISO] Menos de 50 eventos mapeados. Verificando...")
        return
        
    # -------------------------------------------------------------------------
    # COMPARATIVA ESTADÍSTICA: QUIEBRES GENUINOS VS TRAMPAS
    # -------------------------------------------------------------------------
    winners = df_micro[df_micro["target_2a"] == 1]
    losers = df_micro[df_micro["target_2a"] == 0]
    
    print("\n" + "=" * 90)
    print("COMPARATIVA DE MICROESTRUCTURA DE 10 MINUTOS: QUIEBRES GENUINOS VS TRAMPAS")
    print("=" * 90)
    print(f"Total eventos con 10m analizados: {len(df_micro)} (Ganadores: {len(winners)} | Trampas: {len(losers)})")
    
    rows = []
    features_10m = [
        "red_bars_ratio_10m", "below_wap_ratio_10m", "wap_slope_10m",
        "intra_bounce_10m", "early_vol_ratio_10m", "late_trades_ratio_10m"
    ]
    
    for f in features_10m:
        w_val = winners[f].dropna()
        l_val = losers[f].dropna()
        t_stat, p_val = stats.mannwhitneyu(w_val, l_val)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        rows.append({
            "Variable 10 Minutos": f,
            "Media Genuinos (1)": round(w_val.mean(), 4),
            "Media Trampas (0)": round(l_val.mean(), 4),
            "Diferencia": round(w_val.mean() - l_val.mean(), 4),
            "p-value": round(p_val, 4),
            "Significancia": sig
        })
    res_df = pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)
    print(res_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # ANÁLISIS BIVARIADO: WIN RATE SEGÚN RATIO DE VELAS ROJAS DE 10 MINUTOS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ANÁLISIS BIVARIADO: TASA DE ÉXITO SEGÚN LA PRESIÓN VENDEDORA DE 10 MINUTOS")
    print("=" * 90)
    df_micro["red_ratio_bin"] = pd.cut(
        df_micro["red_bars_ratio_10m"],
        bins=[-0.1, 0.40, 0.65, 0.85, 1.01],
        labels=["Baja (<= 40% rojas)", "Media (41% a 65% rojas)", "Alta (66% a 85% rojas)", "Implacable (> 85% rojas)"]
    )
    biv_summary = df_micro.groupby("red_ratio_bin", observed=False).agg(
        Eventos=("event_id", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(biv_summary.to_string())
    
    # -------------------------------------------------------------------------
    # ANÁLISIS BIVARIADO: REBOTE INTRADIARIO DENTRO DE LA HORA
    # -------------------------------------------------------------------------
    print("\nANÁLISIS BIVARIADO: TASA DE ÉXITO SEGÚN EL REBOTE INTRADIARIO (intra_bounce_10m):")
    df_micro["bounce_bin"] = pd.cut(
        df_micro["intra_bounce_10m"],
        bins=[-0.1, 0.20, 0.50, 1.0, 99.0],
        labels=["Cierre en el suelo (<= 0.2% rebote)", "Rebote Leve (0.2% a 0.5%)", "Rebote Moderado (0.5% a 1.0%)", "Rebote Fuerte (> 1.0%)"]
    )
    biv_bounce = df_micro.groupby("bounce_bin", observed=False).agg(
        Eventos=("event_id", "count"),
        Genuinos=("target_2a", "sum"),
        WinRate=("target_2a", lambda x: f"{x.mean()*100:.1f}%")
    )
    print(biv_bounce.to_string())
    print("=" * 90)

if __name__ == "__main__":
    analyze_10min_microstructure()
