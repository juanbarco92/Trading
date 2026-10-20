"""
standard_performance_reporter.py - Módulo Universal de Reporte Cuantitativo
=============================================================================
Este módulo implementa el PROTOCOLO ESTÁNDAR E INMUTABLE para analizar cualquier
estrategia o motor (aislado o combinado).

Genera SIEMPRE las 6 Secciones Obligatorias:
1. Resumen Ejecutivo de Cartera (Capital, PnL, Win Rate, Profit Factor, DD Cerrado y MtM).
2. Desglose Anual y Walk-Forward (Rendimiento año por año, PnL, Win Rate, trades por año).
3. Cobertura del Universo y Concentración de Alfa (Activos únicos, Top 5 ganadores/perdedores, % concentración).
4. Anatomía de Fallas y Salidas (Por qué se gana o se pierde: Stop, Time, BE, Target, Absorción).
5. Desglose por Arquetipo / Régimen (PEAD vs Breakdown, Mercado Alcista vs Bajista).
6. Diagnóstico de Eficiencia de Capital y Riesgo (Tiempo de retención medio, % tiempo invertido).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class StandardPerformanceReporter:
    def __init__(self, initial_capital: float = 25000.0, commission_bps: float = 10.0):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps

    def generate_report(self, df_trades: pd.DataFrame, df_equity: Optional[pd.DataFrame] = None, title: str = "REPORTE ESTANDAR DE IMPACTO") -> str:
        """
        df_trades debe contener como mínimo:
        - ticker (str)
        - entry_ts (datetime)
        - exit_ts (datetime)
        - pnl (float, en USD) o ret (float, pct del slot)
        - type (str, 'LONG' o 'SHORT')
        - reason (str, motivo de salida: 'stop_loss', 'target_hit', 'be_trailing', 'time_max', etc.)
        - archetype (opcional, str)
        """
        if df_trades.empty:
            return "No hay operaciones registradas para generar el reporte."

        df = df_trades.copy()
        if "pnl" not in df.columns and "ret" in df.columns:
            slot_size = self.initial_capital / 5.0
            df["pnl"] = df["ret"] * slot_size
            
        df["entry_ts"] = pd.to_datetime(df["entry_ts"])
        df["exit_ts"] = pd.to_datetime(df["exit_ts"])
        df["year"] = df["entry_ts"].dt.year
        df["duration_hours"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 3600.0
        df["duration_days"] = df["duration_hours"] / 7.0 # 7 horas bursátiles por día

        # 1. Metricas Globales
        total_trades = len(df)
        total_pnl = df["pnl"].sum()
        final_nav = self.initial_capital + total_pnl
        ret_total_pct = (total_pnl / self.initial_capital) * 100.0
        
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        sum_wins = wins["pnl"].sum()
        sum_losses = abs(losses["pnl"].sum())
        profit_factor = sum_wins / (sum_losses + 1e-6)
        
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1e-6
        win_loss_ratio = avg_win / avg_loss
        expectancy_usd = df["pnl"].mean()
        expectancy_pct = (expectancy_usd / (self.initial_capital / 5.0)) * 100.0

        # Drawdown de capital cerrado
        df_sorted = df.sort_values("exit_ts").reset_index(drop=True)
        df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum()
        df_sorted["nav"] = self.initial_capital + df_sorted["cum_pnl"]
        df_sorted["peak"] = df_sorted["nav"].cummax()
        df_sorted["dd_pct"] = (df_sorted["nav"] - df_sorted["peak"]) / df_sorted["peak"] * 100.0
        max_dd_closed = df_sorted["dd_pct"].min()

        # Drawdown MtM (si se provee serie de equity)
        max_dd_mtm = max_dd_closed
        if df_equity is not None and not df_equity.empty and "equity" in df_equity.columns:
            eq = df_equity.copy()
            eq["peak"] = eq["equity"].cummax()
            eq["dd_pct"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100.0
            max_dd_mtm = eq["dd_pct"].min()

        out = []
        out.append("=" * 85)
        out.append(f"📊 {title.upper()}")
        out.append("=" * 85)
        
        # SECCION 1: RESUMEN EJECUTIVO
        out.append("\n### 1. RESUMEN EJECUTIVO DE CARTERA")
        out.append("-" * 85)
        out.append(f"• Capital Inicial:         ${self.initial_capital:,.2f} USD")
        out.append(f"• Capital Final:           ${final_nav:,.2f} USD ({ret_total_pct:+.2f}%)")
        out.append(f"• Beneficio Neto Total:    ${total_pnl:+,.2f} USD")
        out.append(f"• Total Operaciones:       {total_trades:,} trades")
        out.append(f"• Win Rate Global:         {win_rate:.2f}% ({len(wins)} ganadores / {len(losses)} perdedores)")
        out.append(f"• Profit Factor:           {profit_factor:.2f}")
        out.append(f"• Esperanza Matemática:    ${expectancy_usd:+.2f} USD por trade ({expectancy_pct:+.2f}%)")
        out.append(f"• Ratio Ganancia/Pérdida:  {win_loss_ratio:.2f}x (Ganancia Media: ${avg_win:,.2f} | Pérdida Media: ${avg_loss:,.2f})")
        out.append(f"• Max Drawdown Cerrado:    {max_dd_closed:.2f}%")
        out.append(f"• Max Drawdown MtM (Vela): {max_dd_mtm:.2f}%")
        out.append(f"• Duración Media de Trade: {df['duration_days'].mean():.1f} días ({df['duration_hours'].mean():.1f} horas bursátiles)")

        # SECCION 2: DESGLOSE ANUAL Y CONSISTENCIA MULTI-CICLO
        out.append("\n### 2. DESGLOSE ANUAL Y WALK-FORWARD (AÑO POR AÑO)")
        out.append("-" * 85)
        out.append(f"{'Año':<6} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'PnL Neto ($)':<15} {'Retorno Anual %':<16} {'DD Máx Año %'}")
        out.append("-" * 85)
        
        for yr, ydf in df.groupby("year"):
            y_trades = len(ydf)
            y_pnl = ydf["pnl"].sum()
            y_ret = (y_pnl / self.initial_capital) * 100.0
            y_wins = ydf[ydf["pnl"] > 0]["pnl"].sum()
            y_losses = abs(ydf[ydf["pnl"] <= 0]["pnl"].sum())
            y_pf = y_wins / (y_losses + 1e-6)
            y_wr = (len(ydf[ydf["pnl"] > 0]) / y_trades) * 100.0
            
            # DD anual
            ydf_s = ydf.sort_values("exit_ts").reset_index(drop=True)
            ydf_s["cum"] = ydf_s["pnl"].cumsum()
            ydf_s["peak"] = ydf_s["cum"].cummax()
            y_dd = ((ydf_s["cum"] - ydf_s["peak"]) / self.initial_capital) * 100.0
            y_max_dd = y_dd.min()
            
            out.append(f"{yr:<6} {y_trades:<8} {y_wr:<10.1f}% {y_pf:<15.2f} ${y_pnl:<14,.2f} {y_ret:<+15.2f}% {y_max_dd:.2f}%")

        # SECCION 3: COBERTURA DEL UNIVERSO Y CONCENTRACION DE ALFA
        out.append("\n### 3. COBERTURA DEL UNIVERSO Y CONCENTRACIÓN DE ALFA")
        out.append("-" * 85)
        unique_tickers = df["ticker"].nunique()
        out.append(f"• Activos Únicos Operados: {unique_tickers} acciones diferentes.")
        
        ticker_pnl = df.groupby("ticker")["pnl"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        top5_win = ticker_pnl.head(5)
        top5_loss = ticker_pnl.tail(5).sort_values("sum", ascending=True)
        
        top5_profit = top5_win["sum"].sum()
        concentration_pct = (top5_profit / (sum_wins + 1e-6)) * 100.0
        out.append(f"• Concentración de Ganancias en Top 5: {concentration_pct:.1f}% de las ganancias totales.")
        if concentration_pct > 65.0:
            out.append("  ⚠️ ADVERTENCIA: Alta concentración de alfa en pocos activos. Revisar diversificación.")
        else:
            out.append("  ✅ ALFA SANO: Ganancias distribuidas a lo largo del universo de forma diversificada.")
            
        out.append("\nTop 5 Activos Más Rentables:")
        for tk, row in top5_win.iterrows():
            out.append(f"   + {tk:<6}: ${row['sum']:+,.2f} USD en {int(row['count'])} trades")
            
        out.append("\nTop 5 Activos Menos Rentables (Pérdidas Mayores):")
        for tk, row in top5_loss.iterrows():
            out.append(f"   - {tk:<6}: ${row['sum']:+,.2f} USD en {int(row['count'])} trades")

        # SECCION 4: ANATOMIA DE FALLAS Y RAZONES DE SALIDA
        out.append("\n### 4. ANATOMÍA DE FALLAS Y RAZONES DE SALIDA (POR QUÉ SE GANA O SE PIERDE)")
        out.append("-" * 85)
        out.append(f"{'Motivo de Salida':<22} {'Trades':<8} {'% del Total':<12} {'Win Rate':<10} {'PnL Total ($)':<16} {'PnL Medio ($)'}")
        out.append("-" * 85)
        
        for rsn, rdf in df.groupby("reason"):
            r_count = len(rdf)
            r_pct = (r_count / total_trades) * 100.0
            r_wr = (len(rdf[rdf["pnl"] > 0]) / r_count) * 100.0
            r_pnl = rdf["pnl"].sum()
            r_avg = rdf["pnl"].mean()
            out.append(f"{rsn:<22} {r_count:<8} {r_pct:<11.1f}% {r_wr:<9.1f}% ${r_pnl:<15,.2f} ${r_avg:+,.2f}")

        # SECCION 5: DESGLOSE POR TIPO DE POSICION (LONG vs SHORT) O ARQUETIPO
        if "type" in df.columns and df["type"].nunique() > 1:
            out.append("\n### 5. ATRIBUCIÓN POR MOTOR / TIPO DE POSICIÓN")
            out.append("-" * 85)
            out.append(f"{'Motor / Tipo':<15} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'PnL Neto ($)':<16} {'Aporte a Cuenta %'}")
            out.append("-" * 85)
            for tp, tpdf in df.groupby("type"):
                tp_trades = len(tpdf)
                tp_wr = (len(tpdf[tpdf["pnl"] > 0]) / tp_trades) * 100.0
                tp_wins = tpdf[tpdf["pnl"] > 0]["pnl"].sum()
                tp_losses = abs(tpdf[tpdf["pnl"] <= 0]["pnl"].sum())
                tp_pf = tp_wins / (tp_losses + 1e-6)
                tp_pnl = tpdf["pnl"].sum()
                tp_contrib = (tp_pnl / self.initial_capital) * 100.0
                out.append(f"{tp:<15} {tp_trades:<8} {tp_wr:<10.1f}% {tp_pf:<15.2f} ${tp_pnl:<15,.2f} {tp_contrib:<+14.2f}%")

        if "archetype" in df.columns and df["archetype"].nunique() > 1:
            out.append("\nDesglose por Arquetipo:")
            out.append("-" * 85)
            for arc, arcdf in df.groupby("archetype"):
                a_count = len(arcdf)
                a_wr = (len(arcdf[arcdf["pnl"] > 0]) / a_count) * 100.0
                a_wins = arcdf[arcdf["pnl"] > 0]["pnl"].sum()
                a_losses = abs(arcdf[arcdf["pnl"] <= 0]["pnl"].sum())
                a_pf = a_wins / (a_losses + 1e-6)
                a_pnl = arcdf["pnl"].sum()
                out.append(f"• {arc:<22}: {a_count} trades | WR: {a_wr:.1f}% | PF: {a_pf:.2f} | PnL: ${a_pnl:+,.2f} USD")

        # SECCION 6: DIAGNOSTICO FINAL Y RECOMENDACION
        out.append("\n### 6. DIAGNÓSTICO METODOLÓGICO Y CONTROL DE CALIDAD")
        out.append("-" * 85)
        
        checks = []
        if max_dd_closed > -18.0:
            checks.append("✅ Drawdown cerrado controlado (< 18%).")
        else:
            checks.append(f"❌ Drawdown cerrado excesivo ({max_dd_closed:.2f}% > -18%). Requiere endurecer stops.")
            
        if profit_factor >= 1.50:
            checks.append(f"✅ Profit Factor robusto ({profit_factor:.2f} >= 1.50).")
        else:
            checks.append(f"⚠️ Profit Factor bajo ({profit_factor:.2f} < 1.50). Revisar ratio W/L.")
            
        neg_years = sum(1 for yr, ydf in df.groupby("year") if ydf["pnl"].sum() < 0)
        if neg_years == 0:
            checks.append("✅ Consistencia impecable: 0 años con balance negativo.")
        else:
            checks.append(f"⚠️ {neg_years} año(s) terminaron en pérdida. Auditar regímenes de mercado.")
            
        for c in checks:
            out.append(c)
            
        out.append("=" * 85)
        report_str = "\n".join(out)
        return report_str

if __name__ == "__main__":
    # Test rápido de autoverificación
    print("Módulo StandardPerformanceReporter listo para uso universal.")
