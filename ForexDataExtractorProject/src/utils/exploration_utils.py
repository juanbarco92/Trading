import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations


def preprocess_data(df):
    # Convert the 'date' column to datetime, considering the format and timezone
    df['datetime'] = pd.to_datetime(df['date'], format='%Y%m%d %H:%M:%S %Z')
    
    # Extract date components
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.strftime('%Y%m')
    df['Day'] = df['datetime'].dt.day
    df['Date'] = df['datetime'].dt.strftime('%Y%m%d')  # Format as 'YYYYMMDD'
    df['Hour'] = df['datetime'].dt.hour
    df['Minute'] = df['datetime'].dt.minute
    
    # Sort data by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


def check_future_events(row, df, steps=5):
    """
    Vectorized version to check when target or stop loss is hit first within next 'steps' rows
    """
    current_index = row.name
    target_price = row['Target_Price']
    sl_price = row['Stop_Loss_Price']
    
    # Get slice of future data
    max_index = min(current_index + steps, len(df) - 1)
    future_slice = df.loc[current_index+1:max_index]
    
    if future_slice.empty:
        return ("None", None)
    
    # Find first occurrence of target or stop loss
    target_hit = future_slice[future_slice['high'] >= target_price]
    stop_hit = future_slice[future_slice['low'] <= sl_price]
    
    if target_hit.empty and stop_hit.empty:
        return ("None", None)
    
    # Get earliest index for each condition
    target_idx = target_hit.index[0] if not target_hit.empty else float('inf')
    stop_idx = stop_hit.index[0] if not stop_hit.empty else float('inf')
    
    # Return whichever happened first
    if target_idx < stop_idx:
        return ("Target", target_hit.iloc[0]['high'])
    elif stop_idx < target_idx:
        return ("Stop", stop_hit.iloc[0]['low'])
    else:
        return ("None", None)

# Calcular características de las velas
def calcular_features_vela(df):
    df['body_size'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # Evitar división por cero en caso de velas con rango 0
    df['total_range'] = df['total_range'].replace(0, np.nan)

    # Calcular proporciones
    df['body_ratio'] = df['body_size'].abs() / df['total_range']
    df['upper_wick_ratio'] = df['upper_wick'] / df['total_range']
    df['lower_wick_ratio'] = df['lower_wick'] / df['total_range']

    df['total_range'].fillna(0, inplace=True)
    df['body_ratio'].fillna(0, inplace=True)
    df['upper_wick_ratio'].fillna(0, inplace=True)
    df['lower_wick_ratio'].fillna(0, inplace=True)

    return df


def sma_engineering_from_close(df, sma_periods, lookback=5, slope_window=5):
    """
    A partir del DataFrame con columna 'close' y una lista de periodos [5, 10, 20, ...],
    crea las SMAs y genera features entre estas SMAs, incluyendo:
      - Diferencia y ratios entre SMAs (diferencias en pips)
      - Orden y pendientes (slopes en pips)
      - Diferencia relativa entre el precio de cierre y cada SMA
      - Indicadores de cruce (crossovers) entre la SMA de corto y largo plazo
      - Spread o rango entre SMAs (en pips)
      - Aceleración de las SMAs (segunda diferencia en pips)
      - Momentum (cambio en un lookback determinado en pips)
      - Pendiente promedio en una ventana (en pips)
    """
    df = df.copy()
    
    # Verificar que exista la columna 'close'
    if 'close' not in df.columns:
        raise ValueError("La columna 'close' no existe en el DataFrame.")
    
    # Crear columnas de SMA
    sma_cols = [f"SMA_{p}" for p in sma_periods]
    for p, col in zip(sma_periods, sma_cols):
        df[col] = df['close'].rolling(window=p, min_periods=p).mean()
    
    # Eliminar filas iniciales sin datos completos según la SMA de mayor periodo
    longest_sma = f"SMA_{max(sma_periods)}"
    df.dropna(subset=[longest_sma], inplace=True)
    
    # 1. Diferencias y ratios entre cada par de SMAs
    for c1, c2 in combinations(sma_cols, 2):
        # Diferencia absoluta en pips
        df[f'diff_{c1}_{c2}'] = (df[c1] - df[c2]) * 10000
        # Ratios sin escalado (se evita división por cero)
        df[f'ratio_{c1}_{c2}'] = df[c1] / df[c2].replace(0, np.nan)
        df[f'ratio_{c2}_{c1}'] = df[c2] / df[c1].replace(0, np.nan)
    
    # 2. Métrica de orden (número de diferencias positivas entre SMAs consecutivas)
    df['sma_order_score'] = (df[sma_cols].diff(axis=1) > 0).sum(axis=1)
    
    # 3. Pendientes (slopes) de cada SMA y diferencias entre ellas
    slope_cols = []
    for col in sma_cols:
        slope_col = f'slope_{col}'
        # La pendiente en pips (diferencia entre periodos)
        df[slope_col] = df[col].diff() * 10000
        slope_cols.append(slope_col)
    
    for c1, c2 in combinations(slope_cols, 2):
        # Diferencia entre pendientes en pips
        df[f'diff_{c1}_{c2}'] = (df[c1] - df[c2]) * 10000
    
    # 4. Normalización respecto a la SMA más larga (relativa, por lo que no se escala)
    for col in sma_cols:
        if col != longest_sma:
            df[f'norm_diff_{col}_{longest_sma}'] = (df[col] - df[longest_sma]) / df[longest_sma]
    
    # 5. Diferencia relativa entre el precio de cierre y cada SMA (relativa, sin escalado)
    for col in sma_cols:
        df[f'close_diff_{col}'] = (df['close'] - df[col]) / df[col].replace(0, np.nan)
    
    # 6. Indicadores de cruce (crossovers)
    # Se asume que la SMA de menor periodo es la de corto plazo y la de mayor periodo la de largo plazo
    short_sma = sma_cols[0]
    long_sma = sma_cols[-1]
    df['crossover'] = np.where(
        (df[short_sma].shift(1) < df[long_sma].shift(1)) & (df[short_sma] > df[long_sma]),
        1,
        np.where(
            (df[short_sma].shift(1) > df[long_sma].shift(1)) & (df[short_sma] < df[long_sma]),
            -1,
            0
        )
    )
    
    # 7. Spread o rango entre SMAs en pips
    df['sma_spread'] = (df[sma_cols].max(axis=1) - df[sma_cols].min(axis=1)) * 10000
    
    # 8. Aceleración de las SMAs (segunda diferencia en pips)
    for col in sma_cols:
        df[f'acc_{col}'] = df[col].diff().diff() * 10000
    
    # 9. Momentum o cambio en un lookback determinado para cada SMA (en pips)
    for col in sma_cols:
        df[f'{col}_momentum'] = (df[col] - df[col].shift(lookback)) * 10000
        df[f'{col}_pct_momentum'] = df[f'{col}_momentum'] / df[col].shift(lookback).replace(0, np.nan)
    
    # 10. Pendiente promedio en una ventana para cada SMA (media móvil de la pendiente en pips)
    for col in sma_cols:
        slope_col = f'slope_{col}'  # ya calculado anteriormente
        df[f'{col}_slope_mean'] = df[slope_col].rolling(window=slope_window, min_periods=slope_window).mean()
    
    # Eliminar filas con NaN generadas en los cálculos (opcional)
    df.dropna(inplace=True)
    
    return df


def compute_rsi(series, period=14):
    """
    Calcula el RSI sobre un DataFrame o Series de precios de cierre.
    Usaremos un cálculo aproximado clásico de RSI.
    """
    delta = series.diff(1)
    # Ganancias y pérdidas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Media móvil exponencial para suavizado (Wilder)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feature_engineering_rsi(df, close_col='close', period=14):
    df = df.copy()

    if close_col not in df.columns:
        raise ValueError(f"La columna {close_col} no existe en el DataFrame.")

    # Calcular RSI
    df['RSI'] = compute_rsi(df[close_col], period=period)
    
    # Eliminar filas iniciales sin RSI calculado
    df = df.dropna(subset=['RSI']).copy()

    # Indicadores binarios
    df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
    df['rsi_oversold'] = (df['RSI'] < 30).astype(int)

    # Distancias a niveles clave
    df['dist_rsi_70'] = df['RSI'] - 70
    df['dist_rsi_30'] = df['RSI'] - 30
    df['dist_rsi_50'] = df['RSI'] - 50

    # Pendiente del RSI (slope)
    df['rsi_slope'] = df['RSI'].diff(1)
    # Podrías añadir slope a 3 períodos:
    df['rsi_slope_3'] = df['RSI'].diff(3)

    # Señales de cruce de 30 y 70
    df['rsi_cross_30_up'] = ((df['RSI'].shift(1) < 30) & (df['RSI'] > 30)).astype(int)
    df['rsi_cross_70_down'] = ((df['RSI'].shift(1) > 70) & (df['RSI'] < 70)).astype(int)

    return df


def evaluar_escenarios_trading(row, pips_objetivo, pips_stop):
    """
    Evalúa simultáneamente los escenarios de compra y venta para una operación,
    determinando si alcanza el objetivo o el stop loss y calculando el tiempo en velas.
    
    Args:
        row (pd.Series): Fila con datos de la operación (close, future_high_values, etc.)
        pips_objetivo (float): Objetivo de ganancia en pips (positivo)
        pips_stop (float): Nivel de stop loss en pips (positivo)
        
    Returns:
        pd.Series: [
            compra_resultado, compra_velas, compra_pips,
            venta_resultado, venta_velas, venta_pips
        ] donde para cada operación:
            - resultado: "OBJETIVO_ALCANZADO", "STOP_ALCANZADO" o "SIN_ALCANCE"
            - velas: número de velas hasta el resultado
            - pips: movimiento en pips desde el precio de entrada
    """
    precio_entrada = row['close']
    maximos_futuros = np.array(row['future_high_values'])
    minimos_futuros = np.array(row['future_low_values'])
    cierres_futuros = np.array(row['future_close_values'])
    
    # Definir niveles para compra
    nivel_objetivo_compra = precio_entrada + pips_objetivo/10000
    nivel_stop_compra = precio_entrada - pips_stop/10000
    
    # Definir niveles para venta
    nivel_objetivo_venta = precio_entrada - pips_objetivo/10000
    nivel_stop_venta = precio_entrada + pips_stop/10000
    
    # Verificar alcance de niveles
    alcanza_objetivo_compra = maximos_futuros >= nivel_objetivo_compra
    alcanza_stop_compra = minimos_futuros <= nivel_stop_compra
    alcanza_objetivo_venta = minimos_futuros <= nivel_objetivo_venta
    alcanza_stop_venta = maximos_futuros >= nivel_stop_venta
    
    # Encontrar primera ocurrencia para cada escenario
    velas_objetivo_compra = np.argmax(alcanza_objetivo_compra) if np.any(alcanza_objetivo_compra) else None
    velas_stop_compra = np.argmax(alcanza_stop_compra) if np.any(alcanza_stop_compra) else None
    velas_objetivo_venta = np.argmax(alcanza_objetivo_venta) if np.any(alcanza_objetivo_venta) else None
    velas_stop_venta = np.argmax(alcanza_stop_venta) if np.any(alcanza_stop_venta) else None
    
    def calcular_resultado(velas_objetivo, velas_stop, precios_objetivo, precios_stop, multiplicador=1):
        if velas_objetivo is not None and velas_stop is not None:
            if velas_objetivo < velas_stop:
                pips = round((precios_objetivo[velas_objetivo] - precio_entrada) * 10000 * multiplicador, 1)
                return "OBJETIVO_ALCANZADO", velas_objetivo + 1, pips
            else:
                pips = round((precios_stop[velas_stop] - precio_entrada) * 10000 * multiplicador, 1)
                return "STOP_ALCANZADO", velas_stop + 1, pips
        elif velas_objetivo is not None:
            pips = round((precios_objetivo[velas_objetivo] - precio_entrada) * 10000 * multiplicador, 1)
            return "OBJETIVO_ALCANZADO", velas_objetivo + 1, pips
        elif velas_stop is not None:
            pips = round((precios_stop[velas_stop] - precio_entrada) * 10000 * multiplicador, 1)
            return "STOP_ALCANZADO", velas_stop + 1, pips
        else:
            pips = round((cierres_futuros[-1] - precio_entrada) * 10000 * multiplicador, 1)
            return "SIN_ALCANCE", len(cierres_futuros), pips
    
    # Evaluar escenario de compra
    compra_resultado = calcular_resultado(
        velas_objetivo_compra, velas_stop_compra,
        maximos_futuros, minimos_futuros, 1
    )
    
    # Evaluar escenario de venta
    venta_resultado = calcular_resultado(
        velas_objetivo_venta, velas_stop_venta,
        minimos_futuros, maximos_futuros, -1
    )
    
    # Retornar resultados combinados
    return pd.Series([
        *compra_resultado,  # Desempaqueta (resultado, velas, pips) de compra
        *venta_resultado    # Desempaqueta (resultado, velas, pips) de venta
    ], index=[
        'compra_resultado', 'compra_velas', 'compra_pips',
        'venta_resultado', 'venta_velas', 'venta_pips'
    ])


def get_time_window(df, columns, window=5):
    """
    Construye una matriz de ventanas deslizantes usando sliding_window_view.
    
    Para cada fila i (donde i >= window) se toma la ventana compuesta por las filas anteriores,
    es decir, df.iloc[i-window:i]. Se retorna un array en el que cada fila es un vector aplanado 
    con los valores de las columnas especificadas para esa ventana.
    
    Parámetros:
        df: pd.DataFrame con los datos.
        columns: lista de columnas a incluir en la ventana.
        window: número de velas anteriores a considerar (sin incluir la vela actual).
        
    Retorna:
        windows: np.array donde cada fila es un vector aplanado representativo de las filas anteriores.
                  Se generan (n - window) ventanas, donde n es el número de filas de df.
    """
    # Extraer los datos de las columnas indicadas
    data = df[columns].values  # forma: (n, num_columns)
    
    # Crear ventanas consecutivas de tamaño 'window'
    # Esto retorna un array de forma (n - window + 1, window, num_columns)
    all_windows = sliding_window_view(data, window_shape=window, axis=0)
    
    # Cada ventana en all_windows[i] corresponde a df.iloc[i:i+window]
    # Queremos asignar estas ventanas a la vela "actual" que es la siguiente fila a la ventana,
    # es decir, para i en range(window, n) se asocia la ventana df.iloc[i-window:i].
    # Por ello, descartamos la última ventana, ya que no tendría vela actual posterior.
    windows = all_windows[:-1]
    
    # Aplanamos cada ventana en un vector unidimensional
    windows = windows.reshape(windows.shape[0], -1)
    
    return windows

