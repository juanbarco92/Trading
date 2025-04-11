import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import display


def analizar_metricas_por_umbral(y_pred_proba, df_test, pip_objetivo=20, pip_stop=5):
    """
    Analiza las métricas de trading para diferentes umbrales de probabilidad.
    
    Args:
        y_pred_proba: Array de probabilidades predichas (shape: [n_samples, 3])
        df_test: DataFrame con resultados reales
        pip_objetivo: Pips ganados al alcanzar objetivo (default: 20)
        pip_stop: Pips perdidos al alcanzar stop loss (default: 5)
    
    Returns:
        tuple: (compra_df, venta_df) DataFrames con métricas por umbral
    """
    # Crear DataFrame con probabilidades y resultados reales
    prob_df = pd.DataFrame({
        'prob_no_objetivo': y_pred_proba[:, 0],
        'prob_compra': y_pred_proba[:, 1],
        'prob_venta': y_pred_proba[:, 2],
        'actual_compra': df_test['compra_resultado'],
        'actual_venta': df_test['venta_resultado']
    })
    
    # Crear umbrales de probabilidad
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    # Inicializar listas para métricas
    compra_metrics = []
    venta_metrics = []
    
    # Calcular métricas para cada umbral
    for threshold in thresholds:
        # Para predicciones de compra
        compra_preds = (prob_df['prob_compra'] > threshold)
        compra_correct = ((compra_preds) & (prob_df['actual_compra'] == 'OBJETIVO_ALCANZADO')).sum()
        compra_stop = ((compra_preds) & (prob_df['actual_compra'] == 'STOP_ALCANZADO')).sum()
        compra_sin_alcance = ((compra_preds) & (prob_df['actual_compra'] == 'SIN_ALCANCE')).sum()
        compra_total = compra_preds.sum()
        compra_precision = compra_correct/compra_total if compra_total > 0 else 0
        compra_profit = (compra_correct * pip_objetivo) + (compra_sin_alcance * 2.5) - (compra_stop * pip_stop)
        compra_metrics.append((threshold, compra_precision, compra_total, compra_profit))
        
        # Para predicciones de venta
        venta_preds = (prob_df['prob_venta'] > threshold)
        venta_correct = ((venta_preds) & (prob_df['actual_venta'] == 'OBJETIVO_ALCANZADO')).sum()
        venta_stop = ((venta_preds) & (prob_df['actual_venta'] == 'STOP_ALCANZADO')).sum()
        venta_sin_alcance = ((venta_preds) & (prob_df['actual_venta'] == 'SIN_ALCANCE')).sum()
        venta_total = venta_preds.sum()
        venta_precision = venta_correct/venta_total if venta_total > 0 else 0
        venta_profit = (venta_correct * pip_objetivo)+ (venta_sin_alcance * 2.5) - (venta_stop * pip_stop)
        venta_metrics.append((threshold, venta_precision, venta_total, venta_profit))
    
    # Convertir métricas a DataFrames
    compra_df = pd.DataFrame(compra_metrics, columns=['threshold', 'precision', 'total_trades', 'profit'])
    venta_df = pd.DataFrame(venta_metrics, columns=['threshold', 'precision', 'total_trades', 'profit'])
    
    # Visualizar resultados
    visualizar_metricas(compra_df, venta_df)
    
    return compra_df, venta_df

def visualizar_metricas(compra_df, venta_df):
    """
    Visualiza las métricas de trading en tablas y gráficos.
    """
    # Mostrar tablas formateadas
    print("Buy Strategy Metrics:")
    display(compra_df.style
        .format({
            'threshold': '{:.2f}',
            'precision': '{:.2%}',
            'total_trades': '{:,.0f}',
            'profit': '{:,.0f}'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': '', 'props': [('border', '1px solid #ddd')]}
        ])
        .background_gradient(subset=['profit'], cmap='RdYlGn'))
    
    print("\nSell Strategy Metrics:")
    display(venta_df.style
        .format({
            'threshold': '{:.2f}',
            'precision': '{:.2%}',
            'total_trades': '{:,.0f}',
            'profit': '{:,.0f}'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
            {'selector': '', 'props': [('border', '1px solid #ddd')]}
        ])
        .background_gradient(subset=['profit'], cmap='RdYlGn'))
    
    # Crear gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de compra
    plot_metrics(ax1, compra_df, 'Buy Signals')
    # Gráfico de venta
    plot_metrics(ax2, venta_df, 'Sell Signals')
    
    plt.tight_layout()
    plt.show()

def plot_metrics(ax, df, title):
    """
    Función auxiliar para crear gráficos de métricas.
    """
    ax.plot(df['threshold'], df['precision'], 'b-', label='Precision')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax_twin = ax.twinx()
    ax_twin.plot(df['threshold'], df['profit'], 'g:', label='Profit')
    ax_twin.set_ylabel('Profit')
    ax.set_title(title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')





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