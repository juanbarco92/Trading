import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import display


def analizar_metricas_por_umbral(y_pred_proba, df_test, pip_objetivo=20, pip_stop=5, pip_sin_alcance=0.0):
    """
    Analiza las métricas de trading para diferentes umbrales de probabilidad,
    incluyendo análisis mensualizado y promedio de pips por transacción.

    Args:
        y_pred_proba: Array de probabilidades predichas (shape: [n_samples, n_classes])
        df_test: DataFrame con resultados reales (debe incluir columna 'datetime',
                 'compra_resultado', 'venta_resultado')
        pip_objetivo: Pips ganados al alcanzar objetivo (default: 20)
        pip_stop: Pips perdidos al alcanzar stop loss (default: 5)
        pip_sin_alcance: Pips ganados/perdidos si no se alcanza ni objetivo ni stop (default: 0.0)

    Returns:
        tuple: (compra_df, venta_df, compra_mensual_df, venta_mensual_df) DataFrames con métricas
    """
    # Crear DataFrame con probabilidades y resultados reales
    prob_df = pd.DataFrame({
        # Asume que las clases están en orden [?, compra, venta]
        'prob_no_objetivo': y_pred_proba[:, 0],
        'prob_compra': y_pred_proba[:, 1],
        'prob_venta': y_pred_proba[:, 2],
        'actual_compra': df_test['compra_resultado'],
        'actual_venta': df_test['venta_resultado'],
        'datetime': df_test['datetime']
    })

    prob_df['year_month'] = prob_df['datetime'].dt.strftime('%Y-%m')
    thresholds = np.arange(0.1, 1.0, 0.05)

    compra_metrics = []
    venta_metrics = []
    compra_monthly_metrics = []
    venta_monthly_metrics = []

    for threshold in thresholds:
        # Para predicciones de compra
        compra_preds = (prob_df['prob_compra'] > threshold)
        compra_correct = ((compra_preds) & (prob_df['actual_compra'] == 'OBJETIVO_ALCANZADO')).sum()
        compra_stop = ((compra_preds) & (prob_df['actual_compra'] == 'STOP_ALCANZADO')).sum()
        compra_sin_alcance_count = ((compra_preds) & (prob_df['actual_compra'] == 'SIN_ALCANCE')).sum() # Renombrado para claridad
        compra_total = compra_preds.sum()
        compra_precision = compra_correct / compra_total if compra_total > 0 else 0
        # --- MODIFICACIÓN 2: Usar el parámetro 'pip_sin_alcance' ---
        compra_profit = (compra_correct * pip_objetivo) + (compra_sin_alcance_count * pip_sin_alcance) - (compra_stop * pip_stop)
        compra_avg_pips = compra_profit / compra_total if compra_total > 0 else 0
        compra_metrics.append((threshold, compra_precision, compra_total, compra_profit, compra_avg_pips))

        # Para predicciones de venta
        venta_preds = (prob_df['prob_venta'] > threshold)
        venta_correct = ((venta_preds) & (prob_df['actual_venta'] == 'OBJETIVO_ALCANZADO')).sum()
        venta_stop = ((venta_preds) & (prob_df['actual_venta'] == 'STOP_ALCANZADO')).sum()
        venta_sin_alcance_count = ((venta_preds) & (prob_df['actual_venta'] == 'SIN_ALCANCE')).sum() # Renombrado para claridad
        venta_total = venta_preds.sum()
        venta_precision = venta_correct / venta_total if venta_total > 0 else 0
        # --- MODIFICACIÓN 3: Usar el parámetro 'pip_sin_alcance' ---
        venta_profit = (venta_correct * pip_objetivo) + (venta_sin_alcance_count * pip_sin_alcance) - (venta_stop * pip_stop)
        venta_avg_pips = venta_profit / venta_total if venta_total > 0 else 0
        venta_metrics.append((threshold, venta_precision, venta_total, venta_profit, venta_avg_pips))

        # Calcular métricas mensuales para este umbral
        # --- MODIFICACIÓN 4: Pasar 'pip_sin_alcance' a la función auxiliar ---
        monthly_metrics = calcular_metricas_mensuales(prob_df, threshold, pip_objetivo, pip_stop, pip_sin_alcance)

        for month, metrics in monthly_metrics['compra'].items():
            compra_monthly_metrics.append((threshold, month, *metrics))

        for month, metrics in monthly_metrics['venta'].items():
            venta_monthly_metrics.append((threshold, month, *metrics))

    compra_df = pd.DataFrame(compra_metrics,
                            columns=['threshold', 'precision', 'total_trades', 'profit', 'avg_pips'])
    venta_df = pd.DataFrame(venta_metrics,
                           columns=['threshold', 'precision', 'total_trades', 'profit', 'avg_pips'])

    compra_mensual_df = pd.DataFrame(compra_monthly_metrics,
                                   columns=['threshold', 'month', 'precision', 'total_trades', 'profit', 'avg_pips'])
    venta_mensual_df = pd.DataFrame(venta_monthly_metrics,
                                  columns=['threshold', 'month', 'precision', 'total_trades', 'profit', 'avg_pips'])

    visualizar_metricas_con_avg_pips(compra_df, venta_df)
    visualizar_metricas_mensuales(compra_mensual_df, venta_mensual_df)

    return compra_df, venta_df, compra_mensual_df, venta_mensual_df

def calcular_metricas_mensuales(prob_df, threshold, pip_objetivo, pip_stop, pip_sin_alcance):
    """
    Calcula métricas mensualizadas para un umbral específico.
    
    Args:
        prob_df: DataFrame con probabilidades y resultados
        threshold: Umbral de probabilidad
        pip_objetivo: Pips ganados al alcanzar objetivo
        pip_stop: Pips perdidos al alcanzar stop loss
        
    Returns:
        dict: Diccionario con métricas mensuales para compra y venta
    """
    # Inicializar diccionarios para resultados
    compra_mensual = {}
    venta_mensual = {}
    
    # Obtener lista de meses únicos
    meses = prob_df['year_month'].unique()
    
    for mes in meses:
        # Filtrar datos para este mes
        mes_df = prob_df[prob_df['year_month'] == mes]
        
        # Para compras
        compra_preds = (mes_df['prob_compra'] > threshold)
        compra_correct = ((compra_preds) & (mes_df['actual_compra'] == 'OBJETIVO_ALCANZADO')).sum()
        compra_stop = ((compra_preds) & (mes_df['actual_compra'] == 'STOP_ALCANZADO')).sum()
        compra_sin_alcance = ((compra_preds) & (mes_df['actual_compra'] == 'SIN_ALCANCE')).sum()
        compra_total = compra_preds.sum()
        
        if compra_total > 0:
            compra_precision = compra_correct/compra_total
            compra_profit = (compra_correct * pip_objetivo) + (compra_sin_alcance * pip_sin_alcance) - (compra_stop * pip_stop)
            compra_avg_pips = compra_profit / compra_total
            compra_mensual[mes] = (compra_precision, compra_total, compra_profit, compra_avg_pips)
        
        # Para ventas
        venta_preds = (mes_df['prob_venta'] > threshold)
        venta_correct = ((venta_preds) & (mes_df['actual_venta'] == 'OBJETIVO_ALCANZADO')).sum()
        venta_stop = ((venta_preds) & (mes_df['actual_venta'] == 'STOP_ALCANZADO')).sum()
        venta_sin_alcance = ((venta_preds) & (mes_df['actual_venta'] == 'SIN_ALCANCE')).sum()
        venta_total = venta_preds.sum()
        
        if venta_total > 0:
            venta_precision = venta_correct/venta_total
            venta_profit = (venta_correct * pip_objetivo) + (venta_sin_alcance * pip_sin_alcance) - (venta_stop * pip_stop)
            venta_avg_pips = venta_profit / venta_total
            venta_mensual[mes] = (venta_precision, venta_total, venta_profit, venta_avg_pips)
    
    return {'compra': compra_mensual, 'venta': venta_mensual}

def visualizar_metricas_con_avg_pips(compra_df, venta_df):
    """
    Visualiza las métricas de trading incluyendo el promedio de pips por operación,
    con colores específicos para profit.
    """
    print("Buy Strategy Metrics:")
    display(compra_df.style
        .format({
            'threshold': '{:.2f}',
            'precision': '{:.2%}',
            'total_trades': '{:,.0f}',
            'profit': '{:,.0f}',
            'avg_pips': '{:.2f}'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': '', 'props': [('border', '1px solid #ddd')]}
        ])
        # Aplicar color personalizado a 'profit'
        .apply(_color_profit, subset=['profit'], axis=0)
        # Aplicar gradiente diferente a 'avg_pips' (opcional, puedes quitarlo o cambiar cmap)
        .background_gradient(subset=['avg_pips'], cmap='viridis') # Ejemplo con otro cmap
    )

    print("\nSell Strategy Metrics:")
    display(venta_df.style
        .format({
            'threshold': '{:.2f}',
            'precision': '{:.2%}',
            'total_trades': '{:,.0f}',
            'profit': '{:,.0f}',
            'avg_pips': '{:.2f}'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': '', 'props': [('border', '1px solid #ddd')]}
        ])
        # Aplicar color personalizado a 'profit'
        .apply(_color_profit, subset=['profit'], axis=0)
        # Aplicar gradiente diferente a 'avg_pips' (opcional)
        .background_gradient(subset=['avg_pips'], cmap='viridis') # Ejemplo con otro cmap
    )

    # Crear gráficos (sin cambios aquí)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_metrics_with_avg_pips(ax1, compra_df, 'Buy Signals')
    plot_metrics_with_avg_pips(ax2, venta_df, 'Sell Signals')
    plt.tight_layout()
    plt.show()

def plot_metrics_with_avg_pips(ax, df, title):
    """
    Función auxiliar para crear gráficos de métricas incluyendo avg_pips.
    """
    # Primary Y-axis - Precision
    ax.plot(df['threshold'], df['precision'], 'b-', label='Precision')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    
    # Secondary Y-axis - Profit and Avg Pips
    ax_twin = ax.twinx()
    ax_twin.plot(df['threshold'], df['profit'], 'g:', label='Profit')
    ax_twin.plot(df['threshold'], df['avg_pips'], 'm--', label='Avg Pips')
    ax_twin.set_ylabel('Profit / Avg Pips')
    
    ax.set_title(title)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

def visualizar_metricas_mensuales(compra_mensual_df, venta_mensual_df, costo_por_trade=0.2):
    """
    Visualiza las métricas mensualizadas de trading. Selecciona el mejor umbral
    basado en el profit total ajustado por un costo por transacción.
    Muestra tablas con gradiente de color divergente centrado en cero para 'profit'.

    Args:
        compra_mensual_df (pd.DataFrame): DataFrame con métricas mensuales de compra.
        venta_mensual_df (pd.DataFrame): DataFrame con métricas mensuales de venta.
        costo_por_trade (float): Costo estimado en pips por cada transacción (default: 0.2).
    """
    if compra_mensual_df.empty or venta_mensual_df.empty:
        print("Advertencia: No hay datos mensuales suficientes para visualizar.")
        return

    mejor_umbral_compra = None
    mejor_umbral_venta = None

    try:
        # --- Selección de Mejor Umbral para Compras (Profit Ajustado) ---
        if not compra_mensual_df.empty:
            # Agrupar por umbral y sumar profit y trades totales
            compra_agrupado = compra_mensual_df.groupby('threshold').agg(
                total_profit=('profit', 'sum'),
                total_trades=('total_trades', 'sum')
            )
            # Calcular profit ajustado
            compra_agrupado['adjusted_profit'] = compra_agrupado['total_profit'] - (compra_agrupado['total_trades'] * costo_por_trade)

            # Encontrar el umbral con el máximo profit ajustado
            if not compra_agrupado.empty and not compra_agrupado['adjusted_profit'].isnull().all():
                 mejor_umbral_compra = compra_agrupado['adjusted_profit'].idxmax()
                 max_adj_profit_compra = compra_agrupado['adjusted_profit'].max()
                 print(f"Mejor umbral para Compras (max profit ajustado por costo={costo_por_trade}): {mejor_umbral_compra:.2f} (Profit Ajustado: {max_adj_profit_compra:.2f})")
            else:
                 print("Advertencia: No se pudo calcular el profit ajustado para compras.")

        # --- Selección de Mejor Umbral para Ventas (Profit Ajustado) ---
        if not venta_mensual_df.empty:
            venta_agrupado = venta_mensual_df.groupby('threshold').agg(
                total_profit=('profit', 'sum'),
                total_trades=('total_trades', 'sum')
            )
            venta_agrupado['adjusted_profit'] = venta_agrupado['total_profit'] - (venta_agrupado['total_trades'] * costo_por_trade)

            if not venta_agrupado.empty and not venta_agrupado['adjusted_profit'].isnull().all():
                mejor_umbral_venta = venta_agrupado['adjusted_profit'].idxmax()
                max_adj_profit_venta = venta_agrupado['adjusted_profit'].max()
                print(f"Mejor umbral para Ventas (max profit ajustado por costo={costo_por_trade}): {mejor_umbral_venta:.2f} (Profit Ajustado: {max_adj_profit_venta:.2f})")
            else:
                print("Advertencia: No se pudo calcular el profit ajustado para ventas.")

    except Exception as e: # Captura más genérica por si acaso
        print(f"Error determinando el mejor umbral basado en profit ajustado: {e}")
        # Podrías retornar o intentar un método alternativo si falla

    # --- Continuar con la visualización usando los umbrales seleccionados ---
    # (Asegurarse que los umbrales se encontraron)

    # --- Tabla de Compras ---
    if mejor_umbral_compra is not None:
        compra_mensual_filtrada = compra_mensual_df[compra_mensual_df['threshold'] == mejor_umbral_compra]
        if compra_mensual_filtrada.empty:
            print(f"\nNo hay datos mensuales para compras en el umbral seleccionado {mejor_umbral_compra:.2f}.")
        else:
            print(f"\nMétricas Mensuales para Compras (Umbral seleccionado: {mejor_umbral_compra:.2f}):")
            compra_pivot_table = compra_mensual_filtrada.pivot_table(
                index='month',
                values=['precision', 'total_trades', 'profit', 'avg_pips'],
                aggfunc='mean' # Mantenemos 'mean' para la tabla mensual
            ).sort_index()

            max_abs_profit_compra = compra_pivot_table['profit'].abs().max()
            vmin_compra = -max_abs_profit_compra if max_abs_profit_compra > 0 else -1
            vmax_compra = max_abs_profit_compra if max_abs_profit_compra > 0 else 1

            display(compra_pivot_table.style
             .format({
                 'precision': '{:.2%}', 'total_trades': '{:,.0f}',
                 'profit': '{:,.0f}', 'avg_pips': '{:.2f}'
             })
             .background_gradient(subset=['profit'], cmap='RdYlGn', vmin=vmin_compra, vmax=vmax_compra)
             .background_gradient(subset=['avg_pips'], cmap='viridis')
            )
    else:
         print("\nNo se seleccionó un umbral para compras.")


    # --- Tabla de Ventas ---
    if mejor_umbral_venta is not None:
        venta_mensual_filtrada = venta_mensual_df[venta_mensual_df['threshold'] == mejor_umbral_venta]
        if venta_mensual_filtrada.empty:
            print(f"\nNo hay datos mensuales para ventas en el umbral seleccionado {mejor_umbral_venta:.2f}.")
        else:
            print(f"\nMétricas Mensuales para Ventas (Umbral seleccionado: {mejor_umbral_venta:.2f}):")
            venta_pivot_table = venta_mensual_filtrada.pivot_table(
                index='month',
                values=['precision', 'total_trades', 'profit', 'avg_pips'],
                aggfunc='mean'
            ).sort_index()

            max_abs_profit_venta = venta_pivot_table['profit'].abs().max()
            vmin_venta = -max_abs_profit_venta if max_abs_profit_venta > 0 else -1
            vmax_venta = max_abs_profit_venta if max_abs_profit_venta > 0 else 1

            display(venta_pivot_table.style
             .format({
                 'precision': '{:.2%}', 'total_trades': '{:,.0f}',
                 'profit': '{:,.0f}', 'avg_pips': '{:.2f}'
             })
             .background_gradient(subset=['profit'], cmap='RdYlGn', vmin=vmin_venta, vmax=vmax_venta)
             .background_gradient(subset=['avg_pips'], cmap='viridis')
            )
    else:
        print("\nNo se seleccionó un umbral para ventas.")

    # --- Gráfico de evolución mensual ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Graficar Compras si hay datos
    if mejor_umbral_compra is not None and 'compra_mensual_filtrada' in locals() and not compra_mensual_filtrada.empty:
        compra_pivot_grafico = compra_mensual_filtrada.pivot_table(
            index='month', values=['precision', 'profit', 'avg_pips'], aggfunc='mean'
        ).sort_index()
        if not compra_pivot_grafico.empty:
            plot_monthly_metrics(ax1, compra_pivot_grafico, f'Monthly Buy Performance (Threshold: {mejor_umbral_compra:.2f})')
        else:
             ax1.set_title(f'Monthly Buy Performance (Threshold: {mejor_umbral_compra:.2f}) - No Pivot Data')
    else:
        ax1.set_title(f'Monthly Buy Performance - No Threshold Selected or No Data')

    # Graficar Ventas si hay datos
    if mejor_umbral_venta is not None and 'venta_mensual_filtrada' in locals() and not venta_mensual_filtrada.empty:
        venta_pivot_grafico = venta_mensual_filtrada.pivot_table(
            index='month', values=['precision', 'profit', 'avg_pips'], aggfunc='mean'
        ).sort_index()
        if not venta_pivot_grafico.empty:
            plot_monthly_metrics(ax2, venta_pivot_grafico, f'Monthly Sell Performance (Threshold: {mejor_umbral_venta:.2f})')
        else:
             ax2.set_title(f'Monthly Sell Performance (Threshold: {mejor_umbral_venta:.2f}) - No Pivot Data')
    else:
         ax2.set_title(f'Monthly Sell Performance - No Threshold Selected or No Data')

    plt.tight_layout()
    plt.show()

def plot_monthly_metrics(ax, df, title):
    """
    Función auxiliar para crear gráficos de métricas mensuales.
    """
    months = df.index
    
    # Gráfico de barras para profit
    bars = ax.bar(months, df['profit'], alpha=0.6, color='lightgreen', label='Profit')
    
    # Añadir etiquetas de valor sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Línea para avg_pips
    ax_twin1 = ax.twinx()
    ax_twin1.plot(months, df['avg_pips'], 'm-', label='Avg Pips')
    ax_twin1.set_ylabel('Avg Pips', color='m')
    ax_twin1.tick_params(axis='y', labelcolor='m')
    
    # Línea para precision en otro eje
    ax_twin2 = ax.twinx()
    # Mover este eje a la derecha
    ax_twin2.spines['right'].set_position(('outward', 60))
    ax_twin2.plot(months, df['precision'], 'b--', label='Precision')
    ax_twin2.set_ylabel('Precision', color='b')
    ax_twin2.tick_params(axis='y', labelcolor='b')
    
    # Configuración del eje X para mostrar todos los meses
    ax.set_xticks(months)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_ylabel('Profit', color='green')
    ax.tick_params(axis='y', labelcolor='green')
    ax.set_title(title)
    
    # Combinar leyendas de los tres ejes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin1.get_legend_handles_labels()
    lines3, labels3 = ax_twin2.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

def plot_candlestick_with_clusters_plotly(df, df_clusters, N: None):
    """
    Dibuja un candlestick interactivo usando Plotly, con líneas horizontales
    punteadas para los niveles de cluster_center que aparecen en df_clusters.
    
    Requisitos:
      - df con columnas: ['date','open','high','low','close']
      - df_clusters con columna 'cluster_center' (float) para cada nivel
    """

    # Asegurarnos de que 'date' sea datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['datetime'])
    df.sort_values('date', inplace=True)

    # Normalizar las masas para opacidad (0.1 a 1) y color
    if 'mass' in df_clusters.columns:
        max_mass = df_clusters['mass'].max()
        min_mass = df_clusters['mass'].min()
        
        # Opacidad entre 0.2 y 0.9 para mejor visibilidad
        df_clusters['opacity'] = df_clusters['mass'].apply(
            lambda x: 0.2 + 0.7 * (x - min_mass) / (max_mass - min_mass) 
            if max_mass != min_mass else 0.6
        )
        
        # Intensidad de color más moderada
        df_clusters['color_intensity'] = df_clusters['mass'].apply(
            lambda x: (x - min_mass) / (max_mass - min_mass) 
            if max_mass != min_mass else 0.5
        )
    else:
        df_clusters['opacity'] = 0.6
        df_clusters['color_intensity'] = 0.5
        
    # Crear figura
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candlestick'
            )
        ]
    )
    
    # Agregar líneas horizontales con opacidad y color variables
    shapes = []
    for row in df_clusters.itertuples():
        level = getattr(row, 'cluster_center', None)
        opacity = getattr(row, 'opacity', 0.6)
        intensity = getattr(row, 'color_intensity', 0.5)
        
        if level is not None:
            # Escala de color más suave
            r = int(50 + 150 * intensity)   # Rojo: 50-200
            b = int(200 - 100 * intensity)  # Azul: 200-100
            
            shape = dict(
                type="line",
                xref="paper", x0=0, x1=1,
                yref="y", y0=level, y1=level,
                line=dict(
                    dash="dot",
                    color=f"rgba({r}, 20, {b}, {opacity})",
                    width=1.5 + intensity * 1.5  # Grosor más sutil
                )
            )
            shapes.append(shape)
            
    # Línea vertical en N si se especifica
    if N is not None and N < len(df):
        shapes.append(dict(
            type="line",
            xref="x",
            x0=df['date'].iloc[N],
            x1=df['date'].iloc[N],
            yref="paper",
            y0=0,
            y1=1,
            line=dict(
                dash="dot",
                color="red",
                width=1.5
            )
        ))
    
    fig.update_layout(
        title="Niveles de clusters por masa",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        shapes=shapes,
        xaxis_rangeslider_visible=False
    )
    
    fig.show()

def _color_profit(series):
    """Aplica color de fondo a una Serie: verde para > 0, rojo para < 0."""
    # Normalizar positivos para alpha (0.1 a 0.9)
    max_pos = series[series > 0].max()
    norm_pos = series / max_pos if max_pos > 0 else series

    # Normalizar negativos para alpha (0.1 a 0.9)
    # Se invierte para que el más negativo tenga más alpha
    min_neg = series[series < 0].min()
    norm_neg = series / min_neg if min_neg < 0 else series

    styles = []
    for val, n_pos, n_neg in zip(series, norm_pos, norm_neg):
        if val > 0:
            alpha = 0.1 + 0.8 * n_pos # Escala de 0.1 a 0.9
            styles.append(f'background-color: rgba(0, 128, 0, {alpha:.2f})') # Verde
        elif val < 0:
            alpha = 0.1 + 0.8 * n_neg # Escala de 0.1 a 0.9
            styles.append(f'background-color: rgba(255, 0, 0, {alpha:.2f})') # Rojo
        else:
            styles.append('') # Sin estilo para cero
    return styles