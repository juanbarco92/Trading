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
    
    # MODIFICADO: Ya no eliminamos filas sin datos completos
    longest_sma = f"SMA_{max(sma_periods)}"
    
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
        df[f'{col}_slope_mean'] = df[slope_col].rolling(window=slope_window, min_periods=1).mean()
    
    # MODIFICADO: Ya no eliminamos filas con NaN
    # Opcionalmente se pueden reemplazar los NaN por valores sensatos
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(0, inplace=True)
    
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

def feature_engineering_rsi(df, periods=[14, 7, 21], close_col='close', lookback=5, slope_window=3):
    """
    A partir del DataFrame con columna de precios, crea el RSI para diferentes periodos
    y genera características avanzadas basadas en RSI, incluyendo:
      - Valores de RSI para múltiples periodos
      - Estados del RSI (sobrecomprado/sobrevendido)
      - Distancias a niveles clave (30, 50, 70)
      - Pendientes y cambios de dirección
      - Cruces de niveles importantes
      - Velocidad y aceleración de RSI
      - Media móvil del RSI y divergencias
      - Volatilidad y momentum del RSI

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos de precios
    periods : list, default [14, 7, 21]
        Lista de periodos para calcular diferentes RSIs
    close_col : str, default 'close'
        Nombre de la columna con precios de cierre
    lookback : int, default 5
        Periodo para cálculos de momentum
    slope_window : int, default 3
        Tamaño de la ventana para cálculos de pendiente media

    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de RSI añadidas
    """
    df = df.copy()

    # Verificar que exista la columna de precios
    if close_col not in df.columns:
        raise ValueError(f"La columna {close_col} no existe en el DataFrame.")

    # 1. Calcular RSI para diferentes periodos
    rsi_cols = [f"RSI_{p}" for p in periods]
    for p, col in zip(periods, rsi_cols):
        df[col] = compute_rsi(df[close_col], period=p)

    # Usar el RSI del primer periodo como referencia principal
    main_rsi = rsi_cols[0]
    
    # MODIFICADO: Ya no eliminamos filas sin RSI calculado
    
    # 2. Estados del RSI (indicadores binarios)
    df['rsi_extreme_overbought'] = (df[main_rsi] > 80).astype(int)
    df['rsi_overbought'] = ((df[main_rsi] <= 80) & (df[main_rsi] > 70)).astype(int)
    df['rsi_strong'] = ((df[main_rsi] <= 70) & (df[main_rsi] > 60)).astype(int)
    df['rsi_neutral_high'] = ((df[main_rsi] <= 60) & (df[main_rsi] > 50)).astype(int)
    df['rsi_neutral_low'] = ((df[main_rsi] <= 50) & (df[main_rsi] > 40)).astype(int)
    df['rsi_weak'] = ((df[main_rsi] <= 40) & (df[main_rsi] > 30)).astype(int)
    df['rsi_oversold'] = ((df[main_rsi] <= 30) & (df[main_rsi] > 20)).astype(int)
    df['rsi_extreme_oversold'] = (df[main_rsi] <= 20).astype(int)

    # 3. Distancias a niveles clave
    df['dist_rsi_70'] = df[main_rsi] - 70
    df['dist_rsi_50'] = df[main_rsi] - 50
    df['dist_rsi_30'] = df[main_rsi] - 30
    
    # Normalizar distancias (como % del rango posible)
    df['norm_dist_rsi_70'] = (df[main_rsi] - 70) / 30  # normaliza al rango superior (70-100)
    df['norm_dist_rsi_50'] = (df[main_rsi] - 50) / 50  # normaliza a todo el rango (0-100)
    df['norm_dist_rsi_30'] = (df[main_rsi] - 30) / 30  # normaliza al rango inferior (0-30)

    # 4. Pendientes (slopes) del RSI
    for col in rsi_cols:
        # Pendiente directa (variación respecto a la vela anterior)
        df[f'{col}_slope'] = df[col].diff(1)
        
        # Pendiente móvil (media de pendientes en una ventana)
        df[f'{col}_slope_mean'] = df[f'{col}_slope'].rolling(
            window=slope_window, min_periods=1).mean()
        
        # Aceleración (cambio en la pendiente)
        df[f'{col}_acceleration'] = df[f'{col}_slope'].diff(1)

    # 5. Señales de cruce de niveles importantes
    df['rsi_cross_30_up'] = ((df[main_rsi].shift(1) < 30) & (df[main_rsi] > 30)).astype(int)
    df['rsi_cross_50_up'] = ((df[main_rsi].shift(1) < 50) & (df[main_rsi] > 50)).astype(int)
    df['rsi_cross_70_up'] = ((df[main_rsi].shift(1) < 70) & (df[main_rsi] > 70)).astype(int)
    
    df['rsi_cross_30_down'] = ((df[main_rsi].shift(1) > 30) & (df[main_rsi] < 30)).astype(int)
    df['rsi_cross_50_down'] = ((df[main_rsi].shift(1) > 50) & (df[main_rsi] < 50)).astype(int)
    df['rsi_cross_70_down'] = ((df[main_rsi].shift(1) > 70) & (df[main_rsi] < 70)).astype(int)

    # 6. Comparaciones entre diferentes periodos de RSI
    if len(rsi_cols) > 1:
        # RSI rápido vs RSI lento
        fast_rsi = rsi_cols[1]  # Asumimos que el segundo es más rápido (p.ej. 7)
        slow_rsi = rsi_cols[-1]  # Asumimos que el último es más lento (p.ej. 21)
        
        # Diferencia y ratio
        df['rsi_fast_minus_slow'] = df[fast_rsi] - df[slow_rsi]
        df['rsi_fast_ratio_slow'] = df[fast_rsi] / df[slow_rsi].replace(0, np.nan)
        
        # Cruce de RSIs
        df['rsi_fast_cross_above_slow'] = ((df[fast_rsi].shift(1) < df[slow_rsi].shift(1)) & 
                                        (df[fast_rsi] > df[slow_rsi])).astype(int)
        df['rsi_fast_cross_below_slow'] = ((df[fast_rsi].shift(1) > df[slow_rsi].shift(1)) & 
                                        (df[fast_rsi] < df[slow_rsi])).astype(int)

    # 7. Momentum y volatilidad del RSI
    for col in rsi_cols:
        # Momentum (cambio en un periodo de lookback)
        df[f'{col}_momentum'] = df[col] - df[col].shift(lookback)
        df[f'{col}_pct_momentum'] = df[f'{col}_momentum'] / df[col].shift(lookback).replace(0, np.nan)
        
        # Volatilidad (desviación estándar móvil)
        df[f'{col}_volatility'] = df[col].rolling(window=lookback).std()

    # 8. Media móvil del RSI y divergencias
    df[f'{main_rsi}_ma'] = df[main_rsi].rolling(window=lookback).mean()
    
    # Divergencia: RSI baja mientras precio sube o viceversa
    df['rsi_price_divergence'] = ((df[close_col].diff(1) > 0) & 
                                (df[main_rsi].diff(1) < 0) | 
                                (df[close_col].diff(1) < 0) & 
                                (df[main_rsi].diff(1) > 0)).astype(int)
    
    # MODIFICADO: Ya no reemplazamos valores NaN automáticamente
    # Permitimos que el usuario decida cómo manejar los NaN posteriormente
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(0, inplace=True)
    
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
            # Si no hay valores futuros válidos (todos NaN), retornar SIN_ALCANCE con 0 pips
            if np.isnan(cierres_futuros).all():
                 return "SIN_ALCANCE", len(cierres_futuros), 0.0
            # Si hay valores futuros, usar el último cierre válido
            last_valid_index = np.where(~np.isnan(cierres_futuros))[0]
            if len(last_valid_index) > 0:
                last_idx = last_valid_index[-1]
                pips = round((cierres_futuros[last_idx] - precio_entrada) * 10000 * multiplicador, 1)
                return "SIN_ALCANCE", last_idx + 1, pips
            else: # Caso raro donde todos los futuros son NaN
                return "SIN_ALCANCE", len(cierres_futuros), 0.0

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


def generate_target_variable(df, pips_objetivo=10, pips_stop=3, future_periods=24):
    """Generates future values and evaluates trading scenarios to create target."""
    print(f"Generating target variable (pips_objetivo={pips_objetivo}, pips_stop={pips_stop}, future_periods={future_periods})...")
    df_copy = df.copy() # Avoid modifying original df passed to transformer

    required_cols = ['high', 'low', 'close']
    if not all(col in df_copy.columns for col in required_cols):
        raise ValueError(f"Missing required columns for target generation: {required_cols}")

    # Generate future value columns using sliding window
    if future_periods > 0:
        from numpy.lib.stride_tricks import sliding_window_view
        padding = np.full((future_periods - 1), np.nan) # Padding for the end

        for col in ['high', 'low', 'close']:
            future_col_name = f'future_{col}_values'
            # Get values shifted forward (current row looks into the future)
            # shifted_values = df_copy[col].shift(-future_periods).values # Not needed with sliding_window
            # Create sliding windows of the *original* future values
            # We need the next 'future_periods' values for *each* row
            values_for_sliding = df_copy[col].values
            if len(values_for_sliding) >= future_periods:
                 # Window view creates (len - N + 1, N) shape
                 # We need windows starting from index 1 up to index len - N + 1
                 # to represent the future relative to index 0 up to len - N
                 windows = sliding_window_view(values_for_sliding[1:], window_shape=future_periods)
                 # Pad appropriately at the beginning (for the first row) and end
                 padding_array_start = np.full((1, future_periods), np.nan)
                 padding_array_end = np.full((future_periods -1 , future_periods), np.nan)
                 # Combine: start padding, windows, end padding
                 future_values_array = np.vstack([padding_array_start, windows, padding_array_end])
                 # Ensure the final array has the same length as the original DataFrame
                 if len(future_values_array) > len(df_copy):
                     future_values_array = future_values_array[:len(df_copy)]
                 elif len(future_values_array) < len(df_copy):
                      # This case should ideally not happen with correct padding, but handle defensively
                      padding_needed = len(df_copy) - len(future_values_array)
                      extra_padding = np.full((padding_needed, future_periods), np.nan)
                      future_values_array = np.vstack([future_values_array, extra_padding])

                 df_copy[future_col_name] = list(future_values_array)
            else:
                 # Handle cases with insufficient data
                 df_copy[future_col_name] = [np.full(future_periods, np.nan)] * len(df_copy)


    # Evaluate trading scenarios if future columns were created
    if f'future_high_values' in df_copy.columns:
        print("Evaluating trading scenarios...")
        # Ensure future columns exist before applying
        future_cols_exist = all(f'future_{c}_values' in df_copy.columns for c in ['high', 'low', 'close'])
        if future_cols_exist:
            trade_results = df_copy.apply(
                lambda row: evaluar_escenarios_trading(row, pips_objetivo=pips_objetivo, pips_stop=pips_stop),
                axis=1,
                result_type='expand' # Return multiple columns
            )
            trade_results.columns = ['compra_resultado', 'compra_velas', 'compra_pips',
                                     'venta_resultado', 'venta_velas', 'venta_pips']
            df_copy = pd.concat([df_copy, trade_results], axis=1)

            # Create multiclass target
            df_copy["multiclass_target"] = np.where(
                (df_copy['compra_resultado'] == 'OBJETIVO_ALCANZADO') & (df_copy['venta_resultado'] != 'OBJETIVO_ALCANZADO'),
                'objetivo_compra_alcanzado',
                np.where(
                    (df_copy['compra_resultado'] != 'OBJETIVO_ALCANZADO') & (df_copy['venta_resultado'] == 'OBJETIVO_ALCANZADO'),
                    'objetivo_venta_alcanzado',
                    'no_objetivo_alcanzado'
                )
            )
            # Handle cases where evaluation resulted in NaN (e.g., due to NaN future values)
            df_copy['multiclass_target'] = df_copy['multiclass_target'].fillna('undefined')
            print("Target variable 'multiclass_target' created.")
        else:
             print("Skipping trading scenario evaluation as future value columns are missing or incomplete.")
             df_copy['multiclass_target'] = 'undefined' # Or handle as error

    else:
        print("Skipping trading scenario evaluation as future value columns were not generated.")
        df_copy['multiclass_target'] = 'undefined' # Or handle as error

    # Drop intermediate future value columns if desired (optional)
    # df_copy = df_copy.drop(columns=[f'future_{col}_values' for col in ['high', 'low', 'close']], errors='ignore')

    return df_copy


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


def create_candle_features_vector(df, feature_columns=None, n_candles=4):
    """
    Crea un vector de características con los valores de las n_candles anteriores
    para cada fila del DataFrame.
    
    Args:
        df: DataFrame que contiene las características de las velas
        feature_columns: Lista de nombres de columnas con características a incluir.
                        Si es None, se usará ['body_size', 'upper_wick', 'lower_wick', 
                        'total_range', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio']
        n_candles: Número de velas anteriores a considerar
        
    Returns:
        DataFrame con las características de las velas anteriores, donde cada columna
        tiene el formato '{feature_name}_lag{i}' donde i es el número de velas hacia atrás.
    """
    if feature_columns is None:
        feature_columns = [
            'body_size', 'upper_wick', 'lower_wick', 'total_range',
            'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'
        ]
    
    result_df = df.copy()
    
    # Para cada característica
    for feature in feature_columns:
        # Para cada desplazamiento (lag) de 1 a n_candles
        for i in range(1, n_candles + 1):
            # Crear nueva columna con el valor desplazado
            result_df[f'{feature}_lag{i}'] = df[feature].shift(i)
    
    # Opcional: Rellenar NaN con ceros o la media de la columna
    # result_df.fillna(0, inplace=True)
    
    return result_df


def bollinger_bands_engineering(df, periods=[20], std_devs=[2.0], close_col='close', lookback=5):
    """
    A partir del DataFrame con columna de precios, crea las Bandas de Bollinger para diferentes periodos
    y genera características avanzadas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos de precios
    periods : list, default [20]
        Lista de periodos para calcular diferentes Bandas de Bollinger
    std_devs : list, default [2.0]
        Lista de desviaciones estándar para cada período
    close_col : str, default 'close'
        Nombre de la columna con precios de cierre
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de Bandas de Bollinger añadidas
    """
    df = df.copy()
    
    # Verificar que exista la columna de precios
    if close_col not in df.columns:
        raise ValueError(f"La columna {close_col} no existe en el DataFrame.")
    
    # Si std_devs es un solo valor, convertirlo a lista del tamaño de periods
    if isinstance(std_devs, (int, float)):
        std_devs = [std_devs] * len(periods)
    
    # Asegurarse que ambas listas tengan la misma longitud
    if len(periods) != len(std_devs):
        raise ValueError("Las listas 'periods' y 'std_devs' deben tener la misma longitud.")
    
    # Crear las Bandas de Bollinger para cada período
    for p, std in zip(periods, std_devs):
        # Nombre base para este conjunto de bandas
        bb_base = f"BB_{p}_{int(std*10)}"  # Por ejemplo: "BB_20_20" para 20 periodos y 2.0 std
        
        # Calcular las Bandas de Bollinger
        # 1. Media móvil
        df[f"{bb_base}_middle"] = df[close_col].rolling(window=p).mean()
        
        # 2. Desviación estándar
        rolling_std = df[close_col].rolling(window=p).std()
        
        # 3. Bandas superior e inferior
        df[f"{bb_base}_upper"] = df[f"{bb_base}_middle"] + (rolling_std * std)
        df[f"{bb_base}_lower"] = df[f"{bb_base}_middle"] - (rolling_std * std)
        
        # 4. Porcentaje B (posición relativa entre las bandas, de 0 a 1)
        # %B = (Price - Lower Band) / (Upper Band - Lower Band)
        band_width = df[f"{bb_base}_upper"] - df[f"{bb_base}_lower"]
        df[f"{bb_base}_percent_b"] = (df[close_col] - df[f"{bb_base}_lower"]) / band_width.replace(0, np.nan)
        
        # 5. Ancho de banda (indicador de volatilidad)
        # Normalizamos por el nivel de precios para comparar entre diferentes activos
        df[f"{bb_base}_bandwidth"] = band_width / df[f"{bb_base}_middle"]
        
        # 6. Señales de cruce con las bandas
        df[f"{bb_base}_price_cross_upper"] = ((df[close_col].shift(1) <= df[f"{bb_base}_upper"].shift(1)) & 
                                             (df[close_col] > df[f"{bb_base}_upper"])).astype(int)
        df[f"{bb_base}_price_cross_lower"] = ((df[close_col].shift(1) >= df[f"{bb_base}_lower"].shift(1)) & 
                                             (df[close_col] < df[f"{bb_base}_lower"])).astype(int)
        df[f"{bb_base}_price_cross_middle"] = ((df[close_col].shift(1) <= df[f"{bb_base}_middle"].shift(1)) & 
                                              (df[close_col] > df[f"{bb_base}_middle"])).astype(int) - \
                                             ((df[close_col].shift(1) >= df[f"{bb_base}_middle"].shift(1)) & 
                                              (df[close_col] < df[f"{bb_base}_middle"])).astype(int)
        
        # 7. Tendencia de la media móvil (banda central)
        df[f"{bb_base}_middle_slope"] = df[f"{bb_base}_middle"].diff(1)
        df[f"{bb_base}_middle_slope_mean"] = df[f"{bb_base}_middle_slope"].rolling(lookback).mean()
        
        # 8. Contracción/expansión de bandas (cambio en ancho de banda)
        df[f"{bb_base}_bandwidth_change"] = df[f"{bb_base}_bandwidth"].pct_change(1)
        df[f"{bb_base}_squeeze"] = df[f"{bb_base}_bandwidth"] < df[f"{bb_base}_bandwidth"].rolling(lookback*2).mean()
        df[f"{bb_base}_squeeze"] = df[f"{bb_base}_squeeze"].astype(int)
        
        # 9. Momentum relativo a las bandas
        df[f"{bb_base}_percent_b_change"] = df[f"{bb_base}_percent_b"].diff(1)
        df[f"{bb_base}_percent_b_slope"] = df[f"{bb_base}_percent_b"].diff(1).rolling(lookback).mean()
        
        # 10. Distancia a bandas en pips
        df[f"{bb_base}_dist_upper_pips"] = (df[f"{bb_base}_upper"] - df[close_col]) * 10000
        df[f"{bb_base}_dist_lower_pips"] = (df[close_col] - df[f"{bb_base}_lower"]) * 10000
        df[f"{bb_base}_dist_middle_pips"] = abs(df[close_col] - df[f"{bb_base}_middle"]) * 10000
        
        # 11. Momentum vs Bandas (divergencia)
        price_momentum = df[close_col].diff(lookback)
        df[f"{bb_base}_divergence"] = ((price_momentum > 0) & (df[f"{bb_base}_middle_slope"] < 0)) | \
                                     ((price_momentum < 0) & (df[f"{bb_base}_middle_slope"] > 0))
        df[f"{bb_base}_divergence"] = df[f"{bb_base}_divergence"].astype(int)
        
        # 12. Rebote desde bandas (posible señal de reversión)
        df[f"{bb_base}_bounce_upper"] = ((df[close_col].shift(1) >= df[f"{bb_base}_upper"].shift(1)) & 
                                        (df[close_col] < df[close_col].shift(1))).astype(int)
        df[f"{bb_base}_bounce_lower"] = ((df[close_col].shift(1) <= df[f"{bb_base}_lower"].shift(1)) & 
                                        (df[close_col] > df[close_col].shift(1))).astype(int)
    
    # MODIFICADO: Ya no eliminamos filas con NaN
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(0, inplace=True)
    
    return df


def stochastic_engineering(df, k_periods=[14], d_periods=[3], lookback=5, slow=True):
    """
    A partir del DataFrame con columnas de precios, crea el Stochastic Oscillator para diferentes periodos
    y genera características avanzadas basadas en este indicador, incluyendo:
      - Líneas %K y %D para múltiples periodos
      - Estados de sobrecompra/sobreventa
      - Cruces de líneas %K y %D
      - Divergencias entre precio y estocástico
      - Pendientes y cambios de dirección
      - Patrones de convergencia/divergencia
      - Velocidad de cambio y aceleración
      - Volatilidad y momentum del indicador

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos de precios (debe contener columnas high, low y close)
    k_periods : list, default [14]
        Lista de periodos para calcular %K
    d_periods : list, default [3]
        Lista de periodos para calcular %D (media móvil de %K)
    lookback : int, default 5
        Periodo para cálculos de momentum
    slow : bool, default True
        Si es True, calcula el estocástico lento, si es False, calcula el rápido
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características del Stochastic Oscillator añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    for col in ['high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    stoch_cols = []
    
    # Calcular el Estocástico para cada combinación de periodos
    for k_period in k_periods:
        for d_period in d_periods:
            # Nombre base para este conjunto de características
            base_name = f"STOCH_{k_period}_{d_period}"
            
            # Calcular el rango más alto y más bajo durante el periodo k_period
            df[f"{base_name}_highest_high"] = df['high'].rolling(window=k_period).max()
            df[f"{base_name}_lowest_low"] = df['low'].rolling(window=k_period).min()
            
            # Calcular %K (estocástico rápido)
            df[f"{base_name}_K_fast"] = ((df['close'] - df[f"{base_name}_lowest_low"]) / 
                                         (df[f"{base_name}_highest_high"] - df[f"{base_name}_lowest_low"] + 1e-10)) * 100
            
            # Calcular %D (media móvil de %K)
            df[f"{base_name}_D_fast"] = df[f"{base_name}_K_fast"].rolling(window=d_period).mean()
            
            # Si se solicita el estocástico lento
            if slow:
                df[f"{base_name}_K"] = df[f"{base_name}_D_fast"]  # K lento es igual a D rápido
                df[f"{base_name}_D"] = df[f"{base_name}_K"].rolling(window=d_period).mean()  # D lento es SMA de K lento
            else:
                df[f"{base_name}_K"] = df[f"{base_name}_K_fast"]  # K rápido
                df[f"{base_name}_D"] = df[f"{base_name}_D_fast"]  # D rápido
                
            # Guardar los nombres de columnas para el procesamiento posterior
            k_col = f"{base_name}_K"
            d_col = f"{base_name}_D"
            stoch_cols.append((k_col, d_col))
            
            # Eliminar columnas intermedias para no sobrecargar el DataFrame
            df = df.drop(columns=[f"{base_name}_highest_high", f"{base_name}_lowest_low",
                                 f"{base_name}_K_fast", f"{base_name}_D_fast"], errors='ignore')
            
            # 1. Estados del estocástico (indicadores binarios)
            df[f"{base_name}_extreme_overbought"] = (df[k_col] > 90).astype(int)
            df[f"{base_name}_overbought"] = ((df[k_col] <= 90) & (df[k_col] > 80)).astype(int)
            df[f"{base_name}_strong"] = ((df[k_col] <= 80) & (df[k_col] > 60)).astype(int)
            df[f"{base_name}_neutral_high"] = ((df[k_col] <= 60) & (df[k_col] > 50)).astype(int)
            df[f"{base_name}_neutral_low"] = ((df[k_col] <= 50) & (df[k_col] > 40)).astype(int)
            df[f"{base_name}_weak"] = ((df[k_col] <= 40) & (df[k_col] > 20)).astype(int)
            df[f"{base_name}_oversold"] = ((df[k_col] <= 20) & (df[k_col] > 10)).astype(int)
            df[f"{base_name}_extreme_oversold"] = (df[k_col] <= 10).astype(int)
            
            # 2. Distancias a niveles clave
            df[f"{base_name}_dist_to_80"] = df[k_col] - 80
            df[f"{base_name}_dist_to_50"] = df[k_col] - 50
            df[f"{base_name}_dist_to_20"] = df[k_col] - 20
            
            # 3. Pendientes y aceleraciones
            df[f"{base_name}_K_slope"] = df[k_col].diff(1)
            df[f"{base_name}_D_slope"] = df[d_col].diff(1)
            df[f"{base_name}_K_slope_mean"] = df[f"{base_name}_K_slope"].rolling(window=lookback).mean()
            df[f"{base_name}_D_slope_mean"] = df[f"{base_name}_D_slope"].rolling(window=lookback).mean()
            df[f"{base_name}_K_acceleration"] = df[f"{base_name}_K_slope"].diff(1)
            df[f"{base_name}_D_acceleration"] = df[f"{base_name}_D_slope"].diff(1)
            
            # 4. Señales de cruce
            df[f"{base_name}_K_cross_D_up"] = ((df[k_col].shift(1) < df[d_col].shift(1)) & 
                                              (df[k_col] > df[d_col])).astype(int)
            df[f"{base_name}_K_cross_D_down"] = ((df[k_col].shift(1) > df[d_col].shift(1)) & 
                                                (df[k_col] < df[d_col])).astype(int)
            
            df[f"{base_name}_cross_20_up"] = ((df[k_col].shift(1) < 20) & (df[k_col] > 20)).astype(int)
            df[f"{base_name}_cross_80_down"] = ((df[k_col].shift(1) > 80) & (df[k_col] < 80)).astype(int)
            
            # 5. Volatilidad y momentum
            df[f"{base_name}_K_volatility"] = df[k_col].rolling(window=lookback).std()
            df[f"{base_name}_K_momentum"] = df[k_col] - df[k_col].shift(lookback)
            df[f"{base_name}_K_rate_of_change"] = (df[k_col] / df[k_col].shift(lookback) - 1) * 100
            
            # 6. Relación entre K y D
            df[f"{base_name}_K_minus_D"] = df[k_col] - df[d_col]
            df[f"{base_name}_KD_ratio"] = df[k_col] / df[d_col].replace(0, np.nan)
            
            # 7. Diferencia de velocidades entre K y D
            df[f"{base_name}_K_D_velocity_diff"] = df[f"{base_name}_K_slope"] - df[f"{base_name}_D_slope"]
            
            # 8. Convergencia/Divergencia entre K y D
            df[f"{base_name}_KD_convergence"] = -abs(df[k_col] - df[d_col]).diff(1)
            df[f"{base_name}_KD_divergence"] = abs(df[k_col] - df[d_col]).diff(1)
            
            # 9. Divergencias precio-indicador
            df[f"{base_name}_bullish_divergence"] = ((df['close'].diff(lookback) < 0) & 
                                                    (df[k_col].diff(lookback) > 0) & 
                                                    (df[k_col] < 30)).astype(int)
            
            df[f"{base_name}_bearish_divergence"] = ((df['close'].diff(lookback) > 0) & 
                                                    (df[k_col].diff(lookback) < 0) & 
                                                    (df[k_col] > 70)).astype(int)
            
            # 10. Patrones estocásticos
            df[f"{base_name}_peak_formation"] = ((df[k_col].shift(2) < df[k_col].shift(1)) & 
                                                (df[k_col].shift(1) > df[k_col]) & 
                                                (df[k_col].shift(1) > 75)).astype(int)
            
            df[f"{base_name}_valley_formation"] = ((df[k_col].shift(2) > df[k_col].shift(1)) & 
                                                 (df[k_col].shift(1) < df[k_col]) & 
                                                 (df[k_col].shift(1) < 25)).astype(int)
            
            # 11. Indicador compuesto (combinación de varias señales)
            buy_signal = (df[f"{base_name}_K_cross_D_up"] & (df[k_col] < 40)).astype(int)
            sell_signal = (df[f"{base_name}_K_cross_D_down"] & (df[k_col] > 60)).astype(int)
            df[f"{base_name}_signal_combined"] = buy_signal - sell_signal
    
    # Reemplazar valores infinitos y NaN
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(0, inplace=True)
    
    return df


def macd_engineering(df, fast_periods=[12], slow_periods=[26], signal_periods=[9], close_col='close', lookback=5):
    """
    A partir del DataFrame con columna de precios, crea el MACD (Moving Average Convergence Divergence)
    para diferentes combinaciones de periodos y genera características avanzadas, incluyendo:
      - MACD Line, Signal Line y Histogram
      - Estados del MACD (positivo/negativo)
      - Cruces de líneas MACD y Signal
      - Divergencias entre precio y MACD
      - Pendientes y cambios de dirección
      - Histograma y cambios de momentum
      - Tendencias y patrones

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos de precios
    fast_periods : list, default [12]
        Lista de periodos para EMA rápida
    slow_periods : list, default [26]
        Lista de periodos para EMA lenta
    signal_periods : list, default [9]
        Lista de periodos para Signal Line (EMA del MACD)
    close_col : str, default 'close'
        Nombre de la columna con precios de cierre
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de MACD añadidas
    """
    df = df.copy()
    
    # Verificar que exista la columna de precios
    if close_col not in df.columns:
        raise ValueError(f"La columna {close_col} no existe en el DataFrame.")
    
    # Calcular MACD para cada combinación de periodos
    for fast in fast_periods:
        for slow in slow_periods:
            for signal in signal_periods:
                # Verificar que slow > fast
                if slow <= fast:
                    continue
                    
                # Nombre base para este conjunto de MACD
                base_name = f"MACD_{fast}_{slow}_{signal}"
                
                # 1. Calcular las EMAs
                fast_ema = df[close_col].ewm(span=fast, adjust=False).mean()
                slow_ema = df[close_col].ewm(span=slow, adjust=False).mean()
                
                # 2. Calcular MACD Line
                df[f"{base_name}_line"] = fast_ema - slow_ema
                
                # 3. Calcular Signal Line (EMA del MACD)
                df[f"{base_name}_signal"] = df[f"{base_name}_line"].ewm(span=signal, adjust=False).mean()
                
                # 4. Calcular Histograma (MACD - Signal)
                df[f"{base_name}_histogram"] = df[f"{base_name}_line"] - df[f"{base_name}_signal"]
                
                # 5. Estados del MACD
                df[f"{base_name}_is_positive"] = (df[f"{base_name}_line"] > 0).astype(int)
                df[f"{base_name}_is_above_signal"] = (df[f"{base_name}_line"] > df[f"{base_name}_signal"]).astype(int)
                df[f"{base_name}_histogram_positive"] = (df[f"{base_name}_histogram"] > 0).astype(int)
                
                # 6. Cruces de líneas
                df[f"{base_name}_cross_above_signal"] = ((df[f"{base_name}_line"].shift(1) <= df[f"{base_name}_signal"].shift(1)) & 
                                                        (df[f"{base_name}_line"] > df[f"{base_name}_signal"])).astype(int)
                df[f"{base_name}_cross_below_signal"] = ((df[f"{base_name}_line"].shift(1) >= df[f"{base_name}_signal"].shift(1)) & 
                                                        (df[f"{base_name}_line"] < df[f"{base_name}_signal"])).astype(int)
                df[f"{base_name}_cross_zero_up"] = ((df[f"{base_name}_line"].shift(1) <= 0) & 
                                                   (df[f"{base_name}_line"] > 0)).astype(int)
                df[f"{base_name}_cross_zero_down"] = ((df[f"{base_name}_line"].shift(1) >= 0) & 
                                                    (df[f"{base_name}_line"] < 0)).astype(int)
                
                # 7. Pendientes y aceleraciones
                df[f"{base_name}_line_slope"] = df[f"{base_name}_line"].diff(1)
                df[f"{base_name}_signal_slope"] = df[f"{base_name}_signal"].diff(1)
                df[f"{base_name}_histogram_slope"] = df[f"{base_name}_histogram"].diff(1)
                
                df[f"{base_name}_line_acceleration"] = df[f"{base_name}_line_slope"].diff(1)
                df[f"{base_name}_histogram_acceleration"] = df[f"{base_name}_histogram_slope"].diff(1)
                
                # 8. Cambios en el signo del histograma (cambio de momentum)
                df[f"{base_name}_histogram_sign_change"] = (np.sign(df[f"{base_name}_histogram"]) != 
                                                           np.sign(df[f"{base_name}_histogram"].shift(1))).astype(int)
                
                # 9. Divergencias
                # Bullish divergence: precio hace mínimos más bajos pero MACD hace mínimos más altos
                df[f"{base_name}_bullish_divergence"] = (
                    (df[close_col] < df[close_col].shift(lookback)) &  # Precio bajando
                    (df[f"{base_name}_line"] > df[f"{base_name}_line"].shift(lookback)) &  # MACD subiendo
                    (df[f"{base_name}_line"] < 0)  # MACD negativo
                ).astype(int)
                
                # Bearish divergence: precio hace máximos más altos pero MACD hace máximos más bajos
                df[f"{base_name}_bearish_divergence"] = (
                    (df[close_col] > df[close_col].shift(lookback)) &  # Precio subiendo
                    (df[f"{base_name}_line"] < df[f"{base_name}_line"].shift(lookback)) &  # MACD bajando
                    (df[f"{base_name}_line"] > 0)  # MACD positivo
                ).astype(int)
                
                # 10. Medidas de tendencia
                df[f"{base_name}_trend_strength"] = abs(df[f"{base_name}_line"]) / df[f"{base_name}_line"].rolling(window=slow).std()
                df[f"{base_name}_line_mean"] = df[f"{base_name}_line"].rolling(window=lookback).mean()
                
                # 11. Patrones del histograma
                # Histograma creciendo mientras está en negativo (posible cambio de tendencia alcista)
                df[f"{base_name}_growing_histogram_below_zero"] = (
                    (df[f"{base_name}_histogram"] < 0) &
                    (df[f"{base_name}_histogram"] > df[f"{base_name}_histogram"].shift(1)) &
                    (df[f"{base_name}_histogram"].shift(1) > df[f"{base_name}_histogram"].shift(2))
                ).astype(int)
                
                # Histograma decreciendo mientras está en positivo (posible cambio de tendencia bajista)
                df[f"{base_name}_shrinking_histogram_above_zero"] = (
                    (df[f"{base_name}_histogram"] > 0) &
                    (df[f"{base_name}_histogram"] < df[f"{base_name}_histogram"].shift(1)) &
                    (df[f"{base_name}_histogram"].shift(1) < df[f"{base_name}_histogram"].shift(2))
                ).astype(int)
                
                # 12. Señales combinadas
                df[f"{base_name}_buy_signal"] = (
                    df[f"{base_name}_cross_above_signal"] & 
                    (df[f"{base_name}_line"] < 0)  # Cruce desde abajo mientras MACD es negativo
                ).astype(int)
                
                df[f"{base_name}_sell_signal"] = (
                    df[f"{base_name}_cross_below_signal"] & 
                    (df[f"{base_name}_line"] > 0)  # Cruce desde arriba mientras MACD es positivo
                ).astype(int)
                
                # 13. Volatilidad del MACD
                df[f"{base_name}_volatility"] = df[f"{base_name}_line"].rolling(window=lookback).std()
    
    # Reemplazar valores infinitos y NaN
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(0, inplace=True)
    
    return df


def atr_engineering(df, periods=[14], lookback=5):
    """
    A partir del DataFrame con datos OHLC, calcula el ATR (Average True Range)
    para diferentes periodos y genera características avanzadas, incluyendo:
      - ATR para varios periodos
      - ATR normalizado por el precio
      - Volatilidad relativa
      - Cambios en la volatilidad
      - Identificación de periodos de alta/baja volatilidad
      - Bandas de volatilidad
      - Señales de expansión/contracción de volatilidad

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC (debe contener columnas high, low, close)
    periods : list, default [14]
        Lista de periodos para calcular ATR
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de ATR añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    # 1. Calcular True Range para todas las filas
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    # True Range es el máximo de los tres valores
    df['true_range'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # 2. Calcular ATR para cada periodo
    for period in periods:
        base_name = f"ATR_{period}"
        
        # ATR - se puede usar SMA o EMA, el original de Wilder usa EMA con alpha=1/period
        df[base_name] = df['true_range'].ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # 3. ATR normalizado (como porcentaje del precio)
        df[f"{base_name}_pct"] = (df[base_name] / df['close']) * 100
        
        # 4. Volatilidad relativa (comparada con la volatilidad histórica)
        df[f"{base_name}_relative"] = df[base_name] / df[base_name].rolling(window=period*3).mean()
        
        # 5. Cambios en ATR
        df[f"{base_name}_change"] = df[base_name].pct_change()
        df[f"{base_name}_change_5d"] = df[base_name] / df[base_name].shift(lookback) - 1
        
        # 6. Volatilidad del ATR
        df[f"{base_name}_volatility"] = df[base_name].rolling(window=lookback).std() / df[base_name].rolling(window=lookback).mean()
        
        # 7. Identificación de alta/baja volatilidad basada en percentiles
        rolling_window = max(period * 5, 50)  # Ventana para percentiles históricos
        df[f"{base_name}_percentile"] = df[base_name].rolling(rolling_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=True)
        
        df[f"{base_name}_high_volatility"] = (df[f"{base_name}_percentile"] > 80).astype(int)
        df[f"{base_name}_low_volatility"] = (df[f"{base_name}_percentile"] < 20).astype(int)
        
        # 8. Bandas de volatilidad (similar a bandas de Bollinger pero basadas en ATR)
        df[f"{base_name}_upper_band"] = df['close'] + df[base_name] * 2
        df[f"{base_name}_lower_band"] = df['close'] - df[base_name] * 2
        
        # 9. Bandas estrechas (señal de posible movimiento explosivo)
        df[f"{base_name}_squeeze"] = (df[base_name] < df[base_name].rolling(window=period*2).quantile(0.2)).astype(int)
        
        # 10. Expansión y contracción de volatilidad
        df[f"{base_name}_expansion"] = ((df[base_name] > df[base_name].shift(1)) & 
                                       (df[base_name].shift(1) > df[base_name].shift(2))).astype(int)
        
        df[f"{base_name}_contraction"] = ((df[base_name] < df[base_name].shift(1)) & 
                                         (df[base_name].shift(1) < df[base_name].shift(2))).astype(int)
        
        # 11. Identificación de fin de tendencia (volatilidad en expansión después de un periodo de contracción)
        df[f"{base_name}_volatility_breakout"] = ((df[f"{base_name}_expansion"] == 1) & 
                                                 (df[f"{base_name}_squeeze"].shift(1) == 1)).astype(int)
        
        # 12. Relación precio/volatilidad (ADR - Average Daily Range)
        df[f"{base_name}_range_to_atr"] = high_low / df[base_name]
        
        # 13. Tendencia de ATR
        df[f"{base_name}_trend"] = np.where(
            df[base_name] > df[base_name].shift(lookback), 
            1,  # ATR creciente (volatilidad aumentando)
            np.where(
                df[base_name] < df[base_name].shift(lookback),
                -1,  # ATR decreciente (volatilidad disminuyendo)
                0    # ATR estable
            )
        )
    
    # Eliminar columna temporal de true_range
    df = df.drop(columns=['true_range'], errors='ignore')
    
    # Reemplazar valores infinitos y NaN
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(0, inplace=True)
    
    return df


def adx_engineering(df, periods=[14], lookback=5):
    """
    A partir del DataFrame con datos OHLC, calcula el ADX (Average Directional Index)
    para diferentes periodos y genera características avanzadas, incluyendo:
      - ADX, +DI, -DI para varios periodos
      - Fuerza de la tendencia basada en ADX
      - Dirección de la tendencia basada en +DI y -DI
      - Cruces de líneas direccionales
      - Cambios en la fuerza y dirección de la tendencia
      - Señales de trading basadas en ADX
      - Divergencias entre ADX y precio

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC (debe contener columnas high, low, close)
    periods : list, default [14]
        Lista de periodos para calcular ADX
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de ADX añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    # Para cada periodo, calcular ADX y características derivadas
    for period in periods:
        base_name = f"ADX_{period}"
        
        # 1. Calcular movimientos direccionales
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()  # Diferencia con valor anterior
        minus_dm = minus_dm.where(minus_dm > 0, 0)  # Solo tomar valores positivos
        
        # Condiciones para +DM y -DM según Wilder
        plus_dm = np.where(
            (plus_dm > minus_dm) & (plus_dm > 0),
            plus_dm,
            0
        )
        
        minus_dm = np.where(
            (minus_dm > plus_dm) & (minus_dm > 0),
            minus_dm,
            0
        )
        
        # 2. Calcular True Range (necesario para el ADX)
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # 3. Suavizado del True Range y +DM/-DM (usando el método original de Wilder)
        # Primero calculamos la suma inicial del periodo
        tr_period = true_range.rolling(window=period).sum()
        plus_dm_period = pd.Series(plus_dm).rolling(window=period).sum()
        minus_dm_period = pd.Series(minus_dm).rolling(window=period).sum()
        
        # Luego aplicamos el suavizado de Wilder
        tr_smoothed = tr_period.copy()
        plus_dm_smoothed = plus_dm_period.copy()
        minus_dm_smoothed = minus_dm_period.copy()
        
        for i in range(period, len(df)):
            tr_smoothed.iloc[i] = tr_smoothed.iloc[i-1] - (tr_smoothed.iloc[i-1] / period) + true_range.iloc[i]
            plus_dm_smoothed.iloc[i] = plus_dm_smoothed.iloc[i-1] - (plus_dm_smoothed.iloc[i-1] / period) + plus_dm[i]
            minus_dm_smoothed.iloc[i] = minus_dm_smoothed.iloc[i-1] - (minus_dm_smoothed.iloc[i-1] / period) + minus_dm[i]
        
        # 4. Calcular +DI y -DI
        plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
        minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
        
        # 5. Calcular DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # 6. Calcular ADX (media móvil del DX)
        adx = dx.rolling(window=period).mean()
        
        # Guardar valores principales
        df[f"{base_name}_plus_di"] = plus_di
        df[f"{base_name}_minus_di"] = minus_di
        df[f"{base_name}"] = adx
        
        # 7. Fuerza de la tendencia basada en ADX
        df[f"{base_name}_weak_trend"] = (adx < 25).astype(int)
        df[f"{base_name}_moderate_trend"] = ((adx >= 25) & (adx < 50)).astype(int)
        df[f"{base_name}_strong_trend"] = ((adx >= 50) & (adx < 75)).astype(int)
        df[f"{base_name}_extreme_trend"] = (adx >= 75).astype(int)
        
        # 8. Dirección de la tendencia
        df[f"{base_name}_bullish"] = (plus_di > minus_di).astype(int)
        df[f"{base_name}_bearish"] = (minus_di > plus_di).astype(int)
        
        # 9. Señales de entrada
        df[f"{base_name}_buy_signal"] = ((plus_di > minus_di) & 
                                        (plus_di.shift(1) <= minus_di.shift(1)) & 
                                        (adx > 25)).astype(int)
        
        df[f"{base_name}_sell_signal"] = ((minus_di > plus_di) & 
                                         (minus_di.shift(1) <= plus_di.shift(1)) & 
                                         (adx > 25)).astype(int)
        
        # 10. Divergencias
        df[f"{base_name}_adx_rising"] = (adx > adx.shift(1)).astype(int)
        df[f"{base_name}_adx_falling"] = (adx < adx.shift(1)).astype(int)
        
        # 11. Cruces de líneas direccionales
        df[f"{base_name}_di_cross_bullish"] = ((plus_di > minus_di) & 
                                              (plus_di.shift(1) <= minus_di.shift(1))).astype(int)
        
        df[f"{base_name}_di_cross_bearish"] = ((minus_di > plus_di) & 
                                              (minus_di.shift(1) <= plus_di.shift(1))).astype(int)
        
        # 12. Cambios en ADX
        df[f"{base_name}_slope"] = adx.diff(1)
        df[f"{base_name}_acceleration"] = adx.diff(1).diff(1)
        
        # 13. Estrechez o separación entre líneas DI
        df[f"{base_name}_di_spread"] = abs(plus_di - minus_di)
        df[f"{base_name}_di_spread_change"] = df[f"{base_name}_di_spread"].diff(1)
        
        # 14. ADX en niveles críticos
        df[f"{base_name}_adx_above_threshold"] = (adx > 25).astype(int)
        df[f"{base_name}_adx_cross_25_up"] = ((adx > 25) & (adx.shift(1) <= 25)).astype(int)
        df[f"{base_name}_adx_cross_25_down"] = ((adx < 25) & (adx.shift(1) >= 25)).astype(int)
        
        # 15. Tendencia ADX vs periodo anterior
        df[f"{base_name}_trend_strengthening"] = ((adx > adx.shift(lookback)) & (adx > 25)).astype(int)
        df[f"{base_name}_trend_weakening"] = ((adx < adx.shift(lookback)) & (adx > 25)).astype(int)
        
        # 16. Estado consolidación/expansión
        df[f"{base_name}_consolidation"] = ((adx < 25) & (df[f"{base_name}_di_spread"] < 5)).astype(int)
        df[f"{base_name}_expansion"] = ((adx > 25) & (df[f"{base_name}_di_spread"] > df[f"{base_name}_di_spread"].rolling(lookback).mean())).astype(int)
        
        # 17. Señales combinadas
        # Confirmación de tendencia alcista fuerte
        df[f"{base_name}_strong_bullish"] = ((adx > 30) & (plus_di > minus_di) & 
                                          (adx > adx.shift(1)) & (plus_di > plus_di.shift(1))).astype(int)
        
        # Confirmación de tendencia bajista fuerte
        df[f"{base_name}_strong_bearish"] = ((adx > 30) & (minus_di > plus_di) & 
                                           (adx > adx.shift(1)) & (minus_di > minus_di.shift(1))).astype(int)
    
    # Reemplazar valores infinitos y NaN
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(0, inplace=True)
    
    return df


def round_number_pivot_engineering(df, close_col='close', high_col='high', low_col='low', 
                                  levels=[10, 50, 100, 1000], lookback=20, pip_value=0.0001):
    """
    Identifies round numbers in price charts and analyzes the relationship between
    current price and these psychological levels.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    close_col : str, default 'close'
        Name of the column with close prices
    high_col : str, default 'high'
        Name of the column with high prices
    low_col : str, default 'low'
        Name of the column with low prices
    levels : list, default [10, 50, 100, 1000]
        List of round number levels to analyze (e.g., 10 means multiples of 10)
    lookback : int, default 20
        Period for analyzing historical interaction with round numbers
    pip_value : float, default 0.0001
        Value of 1 pip in the currency pair (0.0001 for most forex pairs)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with round number pivot features added
    """
    import math
    
    df = df.copy()
    
    # Verify required columns exist
    required_cols = [close_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"The column {col} does not exist in the DataFrame")
    
    # Check if high/low columns exist, if not use close
    if high_col not in df.columns:
        df[high_col] = df[close_col]
    if low_col not in df.columns:
        df[low_col] = df[close_col]
    
    # Calculate pip multiplier (inverse of pip value)
    pip_multiplier = 1 / pip_value
    
    # Proximity threshold for testing/bouncing (5 pips)
    proximity_threshold = 5 * pip_value
    
    # For each level (10, 50, 100, etc.)
    for level in levels:
        base_name = f"round_{level}"
        
        # Find the nearest round numbers above and below current price
        df[f"{base_name}_above"] = (df[close_col].apply(lambda x: math.ceil(x / level) * level))
        df[f"{base_name}_below"] = (df[close_col].apply(lambda x: math.floor(x / level) * level))
        
        # Calculate distance to these levels in pips
        df[f"{base_name}_above_dist_pips"] = (df[f"{base_name}_above"] - df[close_col]) * pip_multiplier
        df[f"{base_name}_below_dist_pips"] = (df[close_col] - df[f"{base_name}_below"]) * pip_multiplier
        
        # Determine the closest round number level
        df[f"{base_name}_closest"] = np.where(
            df[f"{base_name}_above_dist_pips"] < df[f"{base_name}_below_dist_pips"],
            df[f"{base_name}_above"],
            df[f"{base_name}_below"]
        )
        df[f"{base_name}_closest_dist_pips"] = np.minimum(
            df[f"{base_name}_above_dist_pips"],
            df[f"{base_name}_below_dist_pips"]
        )
        
        # Calculate trend towards or away from the closest round number
        # Negative value means price is getting closer to the level
        # Positive value means price is moving away from the level
        df[f"{base_name}_trend_to_closest"] = df[f"{base_name}_closest_dist_pips"] - df[f"{base_name}_closest_dist_pips"].shift(1)
        
        # Binary indicators for approaching or moving away from level
        df[f"{base_name}_approaching"] = (df[f"{base_name}_trend_to_closest"] < 0).astype(int)
        df[f"{base_name}_moving_away"] = (df[f"{base_name}_trend_to_closest"] > 0).astype(int)
        
        # Calculate recent tests of these levels (price came within threshold of the level)
        df[f"{base_name}_above_tested"] = ((df[high_col] >= df[f"{base_name}_above"] - proximity_threshold) & 
                                         (df[high_col] <= df[f"{base_name}_above"] + proximity_threshold)).astype(int)
        
        df[f"{base_name}_below_tested"] = ((df[low_col] >= df[f"{base_name}_below"] - proximity_threshold) & 
                                         (df[low_col] <= df[f"{base_name}_below"] + proximity_threshold)).astype(int)
        
        # Rolling sum of tests over lookback period
        df[f"{base_name}_above_test_count"] = df[f"{base_name}_above_tested"].rolling(
            window=lookback, min_periods=1).sum()
        
        df[f"{base_name}_below_test_count"] = df[f"{base_name}_below_tested"].rolling(
            window=lookback, min_periods=1).sum()
        
        # Overall level strength (sum of both test counts)
        df[f"{base_name}_level_strength"] = df[f"{base_name}_above_test_count"] + df[f"{base_name}_below_test_count"]
        
        # Detect bounce patterns (price tested level then moved significantly away)
        # Bounce from above level (resistance)
        df[f"{base_name}_above_bounce"] = (
            df[f"{base_name}_above_tested"] & 
            (df[close_col] < df[f"{base_name}_above"] - 2 * proximity_threshold) & 
            (df[close_col].shift(1) >= df[f"{base_name}_above"] - proximity_threshold)
        ).astype(int)
        
        # Bounce from below level (support)
        df[f"{base_name}_below_bounce"] = (
            df[f"{base_name}_below_tested"] & 
            (df[close_col] > df[f"{base_name}_below"] + 2 * proximity_threshold) & 
            (df[close_col].shift(1) <= df[f"{base_name}_below"] + proximity_threshold)
        ).astype(int)
        
        # Detect breakouts through these levels
        df[f"{base_name}_above_breakout"] = (
            (df[close_col] > df[f"{base_name}_above"]) & 
            (df[close_col].shift(1) <= df[f"{base_name}_above"]) & 
            (df[f"{base_name}_above_test_count"] > 0)
        ).astype(int)
        
        df[f"{base_name}_below_breakout"] = (
            (df[close_col] < df[f"{base_name}_below"]) & 
            (df[close_col].shift(1) >= df[f"{base_name}_below"]) & 
            (df[f"{base_name}_below_test_count"] > 0)
        ).astype(int)
    
    # Replace infinite values and NaN
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df.fillna(0, inplace=True)
    
    return df


def create_basic_patterns(df, cluster_column='cluster', n_candles=4):
    """
    Crea patrones concatenando los valores de cluster de las n_candles anteriores.
    Esta función solo genera los patrones sin agruparlos.
    
    Args:
        df: DataFrame que contiene la columna de clusters
        cluster_column: Nombre de la columna que contiene los clusters de velas
        n_candles: Número de velas anteriores a considerar
        
    Returns:
        Series con los patrones básicos de velas
    """
    # Generar las columnas desplazadas (lags)
    lag_cols = [df[cluster_column].shift(i) for i in range(1, n_candles+1)]
    
    # Método más eficiente usando listas y join
    pattern_parts = [lag_cols[0].astype(str)]
    
    for i in range(1, n_candles):
        pattern_parts.append(lag_cols[i].astype(str))
    
    # Unir todos los componentes con "_"
    pattern_series = pattern_parts[0]
    for i in range(1, len(pattern_parts)):
        pattern_series = pattern_series + "_" + pattern_parts[i]
    
    # Reemplazar 'nan' con un valor especial
    pattern_series = pattern_series.replace('nan_.*|.*_nan', 'incomplete_pattern', regex=True)
    
    return pattern_series

def create_pattern_mapping(patterns, min_threshold=10, similarity_method='hamming'):
    """
    Crea un mapa para agrupar patrones poco frecuentes con patrones similares más frecuentes.
    
    Args:
        patterns: Series con patrones para analizar frecuencias
        min_threshold: Umbral mínimo de ocurrencias para mantener un patrón
        similarity_method: Método para calcular similitud entre patrones ('hamming' o 'edit')
        
    Returns:
        Dict con mapeo de patrones raros a sus patrones similares frecuentes
    """
    import Levenshtein
    
    # Contar la frecuencia de cada patrón
    pattern_counts = patterns.value_counts()
    
    # Identificar patrones frecuentes (por encima del umbral)
    frequent_patterns = pattern_counts[pattern_counts >= min_threshold].index.tolist()
    
    # Identificar patrones poco frecuentes (por debajo del umbral)
    rare_patterns = pattern_counts[pattern_counts < min_threshold].index.tolist()
    
    # Si no hay ningún patrón común o no hay patrones raros, devolver un diccionario vacío
    if not frequent_patterns or not rare_patterns:
        return {}
    
    # Función para calcular la distancia entre patrones
    def calculate_distance(pattern1, pattern2):
        if pattern1 == 'incomplete_pattern' or pattern2 == 'incomplete_pattern':
            return float('inf')  # No queremos emparejar patrones incompletos
            
        if similarity_method == 'edit':
            # Distancia de edición (Levenshtein)
            return Levenshtein.distance(pattern1, pattern2)
        else:
            # Distancia Hamming (patrones deben tener igual longitud)
            distance = 0
            parts1 = pattern1.split('_')
            parts2 = pattern2.split('_')
            
            for i in range(min(len(parts1), len(parts2))):
                if parts1[i] != parts2[i]:
                    distance += 1
                    
            # Penalizar longitudes diferentes
            distance += abs(len(parts1) - len(parts2))
            return distance
    
    # Para cada patrón raro, encontrar el patrón frecuente más similar
    pattern_mapping = {}
    for rare in rare_patterns:
        if rare == 'incomplete_pattern':
            continue  # Mantener el patrón incompleto como está
            
        # Calcular la distancia a cada patrón frecuente
        distances = [(frequent, calculate_distance(rare, frequent)) for frequent in frequent_patterns]
        
        # Encontrar el patrón frecuente con menor distancia
        nearest_pattern, min_distance = min(distances, key=lambda x: x[1])
        
        # Solo asignar al patrón más cercano si la distancia es razonable
        # (menos de la mitad de componentes diferentes)
        max_acceptable_distance = len(rare.split('_')) / 2
        
        if min_distance <= max_acceptable_distance:
            pattern_mapping[rare] = nearest_pattern
        else:
            pattern_mapping[rare] = 'other_pattern'  # Si no hay nada suficientemente cercano
    
    return pattern_mapping

def apply_pattern_mapping(patterns, pattern_mapping):
    """
    Aplica un mapeo de patrones para reemplazar los patrones raros.
    
    Args:
        patterns: Series con patrones
        pattern_mapping: Dict con el mapeo de patrones raros a frecuentes
        
    Returns:
        Series con los patrones mapeados
    """
    return patterns.replace(pattern_mapping)

def create_candle_patterns(df, cluster_column='cluster', n_candles=4, min_threshold=10, similarity_method='hamming'):
    """
    Función de compatibilidad para mantener la interfaz anterior.
    Crea patrones concatenando los valores de cluster y agrupa los patrones raros.
    
    Args:
        df: DataFrame que contiene la columna de clusters
        cluster_column: Nombre de la columna que contiene los clusters de velas
        n_candles: Número de velas anteriores a considerar
        min_threshold: Umbral mínimo de ocurrencias para mantener un patrón
        similarity_method: Método para calcular similitud entre patrones
        
    Returns:
        Series con los patrones de velas
    """
    patterns = create_basic_patterns(df, cluster_column, n_candles)
    pattern_mapping = create_pattern_mapping(patterns, min_threshold, similarity_method)
    return apply_pattern_mapping(patterns, pattern_mapping)




from sklearn.decomposition import PCA

def create_pca_features(df, columns, window_size=5, n_components=3):
    """
    Flatten the last `window_size` rows of specified `columns` into a single vector
    for each time step, then apply PCA to reduce dimensionality.
    The resulting principal components are joined to the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the candlestick attributes.
    columns : list of str
        The columns in `df` you want to include for each window.
    window_size : int, optional (default=5)
        The number of consecutive candlesticks to flatten.
    n_components : int, optional (default=3)
        Number of principal components to keep.

    Returns
    -------
    df_with_pca : pd.DataFrame
        The original DataFrame plus the new PCA component columns.
        The first `window_size - 1` rows will have NaNs for the PCA columns.
    pca : PCA object
        The fitted PCA model. You can inspect `explained_variance_ratio_`, etc.
    """
    X_data = []
    valid_indices = []

    # Loop over each possible time window
    for t in range(window_size - 1, len(df)):
        # Slice the last `window_size` rows
        window = df.iloc[t - window_size + 1 : t + 1]

        # Flatten the selected columns (shape: window_size * number_of_columns)
        flattened = window[columns].values.flatten()
        X_data.append(flattened)

        # This index corresponds to the end of the window
        valid_indices.append(df.index[t])

    # Convert to numpy array
    X = np.array(X_data)

    # Create and fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Build a DataFrame for the PCA result
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=valid_indices)

    # Print explained variance information
    print("Explained variance ratio per component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.2%}")
    total_var = pca.explained_variance_ratio_.sum()
    print(f"Total variance explained by the top {n_components} PCs: {total_var:.2%}")

    # Join PCA columns to the original DataFrame
    df_with_pca = df.join(df_pca, how='outer')

    return df_with_pca


def cci_engineering(df, periods=[14, 20], lookback=5, slope_window=3):
    """
    Calcula el CCI (Commodity Channel Index) para diferentes periodos y genera
    características avanzadas basadas en este indicador, incluyendo:
      - Valores de CCI para múltiples periodos
      - Estados del CCI (sobrecomprado/sobrevendido)
      - Cruces de niveles importantes (+100, 0, -100)
      - Pendientes y cambios de dirección
      - Divergencias entre precio y CCI
      - Volatilidad y momentum del CCI
      - Señales de trading basadas en CCI

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC (debe contener columnas high, low, close)
    periods : list, default [14, 20]
        Lista de periodos para calcular el CCI
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
    slope_window : int, default 3
        Tamaño de la ventana para cálculos de pendiente media

    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de CCI añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    # Calcular el precio típico
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calcular CCI para cada periodo
    cci_cols = []
    for period in periods:
        base_name = f"CCI_{period}"
        cci_cols.append(base_name)
        
        # 1. Calcular la media móvil simple del precio típico
        df[f"{base_name}_ma"] = df['typical_price'].rolling(window=period).mean()
        
        # 2. Calcular la desviación media
        df[f"{base_name}_mean_deviation"] = df['typical_price'].rolling(window=period).apply(
            lambda x: abs(x - x.mean()).mean()
        )
        
        # 3. Calcular el CCI
        # CCI = (Precio Típico - SMA) / (0.015 * Desviación media)
        df[base_name] = (df['typical_price'] - df[f"{base_name}_ma"]) / (0.015 * df[f"{base_name}_mean_deviation"])
        
        # Eliminar columnas intermedias para no sobrecargar el DataFrame
        df = df.drop(columns=[f"{base_name}_ma", f"{base_name}_mean_deviation"], errors='ignore')
        
        # 4. Estados del CCI (indicadores binarios)
        df[f"{base_name}_extreme_overbought"] = (df[base_name] > 200).astype(int)
        df[f"{base_name}_overbought"] = ((df[base_name] <= 200) & (df[base_name] > 100)).astype(int)
        df[f"{base_name}_strong"] = ((df[base_name] <= 100) & (df[base_name] > 0)).astype(int)
        df[f"{base_name}_weak"] = ((df[base_name] <= 0) & (df[base_name] > -100)).astype(int)
        df[f"{base_name}_oversold"] = ((df[base_name] <= -100) & (df[base_name] > -200)).astype(int)
        df[f"{base_name}_extreme_oversold"] = (df[base_name] <= -200).astype(int)
        
        # 5. Pendientes y cambios de dirección
        df[f"{base_name}_slope"] = df[base_name].diff(1)
        df[f"{base_name}_slope_mean"] = df[f"{base_name}_slope"].rolling(window=slope_window, min_periods=1).mean()
        df[f"{base_name}_acceleration"] = df[f"{base_name}_slope"].diff(1)
        
        # 6. Cruces de niveles importantes
        df[f"{base_name}_cross_100_up"] = ((df[base_name].shift(1) < 100) & (df[base_name] >= 100)).astype(int)
        df[f"{base_name}_cross_0_up"] = ((df[base_name].shift(1) < 0) & (df[base_name] >= 0)).astype(int)
        df[f"{base_name}_cross_neg100_up"] = ((df[base_name].shift(1) < -100) & (df[base_name] >= -100)).astype(int)
        
        df[f"{base_name}_cross_100_down"] = ((df[base_name].shift(1) > 100) & (df[base_name] <= 100)).astype(int)
        df[f"{base_name}_cross_0_down"] = ((df[base_name].shift(1) > 0) & (df[base_name] <= 0)).astype(int)
        df[f"{base_name}_cross_neg100_down"] = ((df[base_name].shift(1) > -100) & (df[base_name] <= -100)).astype(int)
        
        # 7. Momentum y volatilidad del CCI
        df[f"{base_name}_momentum"] = df[base_name] - df[base_name].shift(lookback)
        df[f"{base_name}_volatility"] = df[base_name].rolling(window=lookback).std()
        
        # 8. Divergencias con el precio
        # Divergencia alcista: precio hace mínimos más bajos pero CCI hace mínimos más altos
        df[f"{base_name}_bullish_divergence"] = (
            (df['close'] < df['close'].shift(lookback)) &  # Precio bajando
            (df[base_name] > df[base_name].shift(lookback)) &  # CCI subiendo
            (df[base_name] < -100)  # CCI en zona de sobreventa
        ).astype(int)
        
        # Divergencia bajista: precio hace máximos más altos pero CCI hace máximos más bajos
        df[f"{base_name}_bearish_divergence"] = (
            (df['close'] > df['close'].shift(lookback)) &  # Precio subiendo
            (df[base_name] < df[base_name].shift(lookback)) &  # CCI bajando
            (df[base_name] > 100)  # CCI en zona de sobrecompra
        ).astype(int)
        
        # 9. Tendencia del CCI
        df[f"{base_name}_uptrend"] = (df[base_name] > 0).astype(int)
        df[f"{base_name}_downtrend"] = (df[base_name] < 0).astype(int)
        
        # 10. Señales de negociación basadas en el CCI
        # Señal de compra: cruce por encima de -100 desde la zona de sobreventa
        df[f"{base_name}_buy_signal"] = (
            df[f"{base_name}_cross_neg100_up"] & 
            (df[base_name].shift(1) < -100)
        ).astype(int)
        
        # Señal de venta: cruce por debajo de 100 desde la zona de sobrecompra
        df[f"{base_name}_sell_signal"] = (
            df[f"{base_name}_cross_100_down"] & 
            (df[base_name].shift(1) > 100)
        ).astype(int)
        
        # 11. Fuerza de la tendencia
        df[f"{base_name}_trend_strength"] = abs(df[base_name]) / 100
        
        # 12. Tiempo en zonas extremas
        df[f"{base_name}_in_overbought"] = (df[base_name] > 100).rolling(window=lookback).sum()
        df[f"{base_name}_in_oversold"] = (df[base_name] < -100).rolling(window=lookback).sum()
        
        # 13. Oscilación extrema (volatilidad de corto plazo)
        df[f"{base_name}_extreme_move"] = ((abs(df[base_name] - df[base_name].shift(1))) > 50).astype(int)
    
    # Comparaciones entre diferentes periodos de CCI (si hay más de uno)
    if len(cci_cols) > 1:
        short_cci = cci_cols[0]  # Asumimos que el primero es más corto
        long_cci = cci_cols[-1]  # Asumimos que el último es más largo
        
        # Diferencia entre CCI corto y largo
        df["CCI_short_minus_long"] = df[short_cci] - df[long_cci]
        
        # Cruce del CCI corto por encima/debajo del CCI largo
        df["CCI_short_cross_above_long"] = ((df[short_cci].shift(1) <= df[long_cci].shift(1)) & 
                                          (df[short_cci] > df[long_cci])).astype(int)
        df["CCI_short_cross_below_long"] = ((df[short_cci].shift(1) >= df[long_cci].shift(1)) & 
                                          (df[short_cci] < df[long_cci])).astype(int)
    
    # Eliminar la columna temporal de precio típico
    df = df.drop(columns=['typical_price'], errors='ignore')
    
    # No eliminamos filas con NaN, siguiendo la política establecida
    return df


def parabolic_sar_engineering(df, af_starts=[0.02], af_maxes=[0.2], lookback=5):
    """
    Calcula el indicador Parabolic SAR para diferentes combinaciones de parámetros
    y genera características avanzadas basadas en él, incluyendo:
      - Valores de SAR para múltiples configuraciones
      - Estado de la tendencia según SAR (alcista/bajista)
      - Reversiones de tendencia (cambios de dirección)
      - Distancia entre precio y SAR (en pips)
      - Aceleración de movimiento del SAR
      - Señales de trading basadas en SAR
      - Fuerza de la tendencia y protección de ganancias

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC (debe contener columnas high, low, close)
    af_starts : list, default [0.02]
        Lista de valores iniciales del factor de aceleración
    af_maxes : list, default [0.2]
        Lista de valores máximos del factor de aceleración
    lookback : int, default 5
        Periodo para cálculos de momentum y tendencias
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características del Parabolic SAR añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    # Para cada combinación de parámetros
    for af_start in af_starts:
        for af_max in af_maxes:
            # Nombre base para este conjunto de parámetros
            base_name = f"SAR_{int(af_start*1000)}_{int(af_max*1000)}"
            
            # Calcular Parabolic SAR
            is_long = [True]  # Suponemos tendencia inicial alcista
            sar_values = [df['low'].iloc[0]]  # Empezamos con el primer mínimo
            ep_values = [df['high'].iloc[0]]  # Primer extremo
            af_values = [af_start]  # Factor de aceleración inicial
            
            for i in range(1, len(df)):
                # Obtener valores previos
                prev_sar = sar_values[-1]
                prev_ep = ep_values[-1]
                prev_af = af_values[-1]
                prev_is_long = is_long[-1]
                
                # Calcular nuevo SAR
                if prev_is_long:
                    # En tendencia alcista
                    sar = prev_sar + prev_af * (prev_ep - prev_sar)
                    
                    # Asegurar que el SAR no esté por encima de los mínimos de los 2 periodos anteriores
                    if i >= 2:
                        min_low = min(df['low'].iloc[i-1], df['low'].iloc[i-2])
                        sar = min(sar, min_low)
                    
                    # Comprobar si hay cambio de tendencia
                    if sar > df['low'].iloc[i]:
                        # Cambio a tendencia bajista
                        is_long_new = False
                        sar = max(df['high'].iloc[max(0, i-2):i+1])  # Máximo de los últimos periodos
                        ep_new = df['low'].iloc[i]  # Nuevo extremo
                        af_new = af_start  # Resetear AF
                    else:
                        # Continuar tendencia alcista
                        is_long_new = True
                        ep_new = max(prev_ep, df['high'].iloc[i])  # Actualizar extremo si hay nuevo máximo
                        
                        # Actualizar factor de aceleración si hay nuevo extremo
                        if ep_new > prev_ep:
                            af_new = min(prev_af + af_start, af_max)
                        else:
                            af_new = prev_af
                else:
                    # En tendencia bajista
                    sar = prev_sar - prev_af * (prev_sar - prev_ep)
                    
                    # Asegurar que el SAR no esté por debajo de los máximos de los 2 periodos anteriores
                    if i >= 2:
                        max_high = max(df['high'].iloc[i-1], df['high'].iloc[i-2])
                        sar = max(sar, max_high)
                    
                    # Comprobar si hay cambio de tendencia
                    if sar < df['high'].iloc[i]:
                        # Cambio a tendencia alcista
                        is_long_new = True
                        sar = min(df['low'].iloc[max(0, i-2):i+1])  # Mínimo de los últimos periodos
                        ep_new = df['high'].iloc[i]  # Nuevo extremo
                        af_new = af_start  # Resetear AF
                    else:
                        # Continuar tendencia bajista
                        is_long_new = False
                        ep_new = min(prev_ep, df['low'].iloc[i])  # Actualizar extremo si hay nuevo mínimo
                        
                        # Actualizar factor de aceleración si hay nuevo extremo
                        if ep_new < prev_ep:
                            af_new = min(prev_af + af_start, af_max)
                        else:
                            af_new = prev_af
                
                # Guardar los valores calculados
                sar_values.append(sar)
                ep_values.append(ep_new)
                af_values.append(af_new)
                is_long.append(is_long_new)
            
            # Asignar valores al DataFrame
            df[f"{base_name}"] = sar_values
            df[f"{base_name}_trend"] = [1 if x else -1 for x in is_long]  # 1 para alcista, -1 para bajista
            df[f"{base_name}_af"] = af_values
            
            # 1. Distancia entre precio y SAR en pips
            df[f"{base_name}_distance_pips"] = ((df['close'] - df[f"{base_name}"]) * 10000 * 
                                              df[f"{base_name}_trend"])
            
            # 2. Detectar cambios de tendencia (SAR flips)
            df[f"{base_name}_trend_change"] = df[f"{base_name}_trend"].diff().abs() / 2
            
            # 3. Señales de compra/venta basadas en cambios de tendencia
            df[f"{base_name}_buy_signal"] = ((df[f"{base_name}_trend"] == 1) & 
                                           (df[f"{base_name}_trend"].shift(1) == -1)).astype(int)
            df[f"{base_name}_sell_signal"] = ((df[f"{base_name}_trend"] == -1) & 
                                            (df[f"{base_name}_trend"].shift(1) == 1)).astype(int)
            
            # 4. Velocidad de cambio del SAR
            df[f"{base_name}_velocity"] = df[f"{base_name}"].diff().abs() * 10000  # en pips
            
            # 5. Aceleración del SAR
            df[f"{base_name}_acceleration"] = df[f"{base_name}_velocity"].diff()
            
            # 6. Media móvil de la velocidad del SAR
            df[f"{base_name}_velocity_mean"] = df[f"{base_name}_velocity"].rolling(window=lookback).mean()
            
            # 7. Posición relativa del precio respecto al SAR
            df[f"{base_name}_rel_position"] = (df['close'] - df[f"{base_name}"]) / df[f"{base_name}"] * 100
            
            # 8. Días consecutivos en la misma tendencia
            df[f"{base_name}_trend_duration"] = df[f"{base_name}_trend"].groupby(
                (df[f"{base_name}_trend"] != df[f"{base_name}_trend"].shift(1)).cumsum()
            ).cumcount() + 1
            
            # 9. Fuerza de la tendencia (distancia normalizada por volatilidad)
            atr = df['high'] - df['low']  # Simplificación de ATR
            df[f"{base_name}_trend_strength"] = df[f"{base_name}_distance_pips"] / (atr * 10000)
            
            # 10. Proximidad al SAR (inversa de la distancia, valores más altos = más cerca)
            df[f"{base_name}_proximity"] = 1 / (df[f"{base_name}_distance_pips"] + 1)
            
            # 11. Señal de riesgo (precio acercándose al SAR en tendencia)
            df[f"{base_name}_risk_signal"] = ((df[f"{base_name}_distance_pips"] < 
                                             df[f"{base_name}_distance_pips"].shift(1)) & 
                                            (df[f"{base_name}_trend"] == df[f"{base_name}_trend"].shift(1))).astype(int)
            
            # 12. Señal de aceleración (SAR acelerando)
            df[f"{base_name}_accel_signal"] = (df[f"{base_name}_velocity"] > 
                                             df[f"{base_name}_velocity_mean"]).astype(int)
            
            # 13. Protección de ganancias (SAR cerca del precio durante tendencia prolongada)
            df[f"{base_name}_profit_protection"] = ((df[f"{base_name}_proximity"] > 0.1) & 
                                                  (df[f"{base_name}_trend_duration"] > 5)).astype(int)
            
            # 14. Tendencia con momento (tendencia sostenida con precio alejándose del SAR)
            df[f"{base_name}_trend_momentum"] = ((df[f"{base_name}_trend_duration"] > 3) & 
                                               (df[f"{base_name}_distance_pips"] > 
                                                df[f"{base_name}_distance_pips"].shift(1))).astype(int)
            
            # 15. Convergencia precio-SAR (precio acercándose al SAR)
            df[f"{base_name}_price_sar_convergence"] = ((df[f"{base_name}_distance_pips"] < 
                                                       df[f"{base_name}_distance_pips"].shift(1))).astype(int)
            
            # 16. Divergencia precio-SAR (precio alejándose del SAR)
            df[f"{base_name}_price_sar_divergence"] = ((df[f"{base_name}_distance_pips"] > 
                                                      df[f"{base_name}_distance_pips"].shift(1))).astype(int)
    
    # No eliminamos filas con NaN, siguiendo la política establecida
    return df



def fractal_engineering(df, window_sizes=[5], lookback=10, high_col='high', low_col='low', close_col='close'):
    """
    Calcula los fractales de Bill Williams para identificar potenciales puntos de reversión
    y genera características avanzadas basadas en ellos, incluyendo:
      - Identificación de fractales alcistas y bajistas
      - Distancia entre fractales sucesivos
      - Fortaleza de los fractales (diferencia relativa)
      - Frecuencia de aparición de fractales
      - Líneas de tendencia basadas en fractales
      - Divergencias entre precio y fractales
      - Señales de trading basadas en rupturas de fractales
      - Patrones formados por series de fractales

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC (debe contener columnas high, low, close)
    window_sizes : list, default [5]
        Lista de tamaños de ventana para identificar fractales (ej: 5 = patrón clásico de 5 velas)
    lookback : int, default 10
        Periodo para calcular la línea de tendencia y otras características
    high_col : str, default 'high'
        Nombre de la columna con precios máximos
    low_col : str, default 'low'
        Nombre de la columna con precios mínimos
    close_col : str, default 'close'
        Nombre de la columna con precios de cierre
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las características de fractales añadidas
    """
    df = df.copy()
    
    # Verificar que existan las columnas necesarias
    required_cols = [high_col, low_col, close_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    # Para cada tamaño de ventana
    for window in window_sizes:
        # Necesitamos un número impar para la ventana (centro + lados iguales)
        if window % 2 == 0:
            window += 1
            
        half_window = window // 2
        base_name = f"FRACTAL_{window}"
        
        # 1. Identificar fractales alcistas (bullish)
        # Un fractal alcista ocurre cuando una vela tiene el máximo más alto que n velas a cada lado
        bullish_fractal = np.zeros(len(df))
        
        for i in range(half_window, len(df) - half_window):
            # Verificar que el máximo central sea mayor que todos los máximos circundantes
            is_bullish = True
            central_high = df[high_col].iloc[i]
            
            for j in range(1, half_window + 1):
                if (df[high_col].iloc[i-j] >= central_high) or (df[high_col].iloc[i+j] >= central_high):
                    is_bullish = False
                    break
            
            if is_bullish:
                bullish_fractal[i] = 1
                
        # 2. Identificar fractales bajistas (bearish)
        # Un fractal bajista ocurre cuando una vela tiene el mínimo más bajo que n velas a cada lado
        bearish_fractal = np.zeros(len(df))
        
        for i in range(half_window, len(df) - half_window):
            # Verificar que el mínimo central sea menor que todos los mínimos circundantes
            is_bearish = True
            central_low = df[low_col].iloc[i]
            
            for j in range(1, half_window + 1):
                if (df[low_col].iloc[i-j] <= central_low) or (df[low_col].iloc[i+j] <= central_low):
                    is_bearish = False
                    break
            
            if is_bearish:
                bearish_fractal[i] = 1
        
        # Guardar los fractales identificados en el DataFrame
        df[f"{base_name}_bullish"] = bullish_fractal
        df[f"{base_name}_bearish"] = bearish_fractal
        
        # Capturar los valores de precio donde ocurren los fractales
        df[f"{base_name}_bullish_price"] = np.where(bullish_fractal == 1, df[high_col], np.nan)
        df[f"{base_name}_bearish_price"] = np.where(bearish_fractal == 1, df[low_col], np.nan)
        
        # 3. Distancia entre fractales sucesivos (en barras)
        # Calcular la distancia desde el último fractal alcista/bajista
        last_bullish_idx = 0
        last_bearish_idx = 0
        bullish_distances = np.zeros(len(df))
        bearish_distances = np.zeros(len(df))
        
        for i in range(len(df)):
            if bullish_fractal[i] == 1:
                for j in range(last_bullish_idx, i + 1):
                    bullish_distances[j] = i - last_bullish_idx
                last_bullish_idx = i
            
            if bearish_fractal[i] == 1:
                for j in range(last_bearish_idx, i + 1):
                    bearish_distances[j] = i - last_bearish_idx
                last_bearish_idx = i
        
        df[f"{base_name}_bull_distance"] = bullish_distances
        df[f"{base_name}_bear_distance"] = bearish_distances
        
        # 4. Dirección del fractal (alcista o bajista)
        # 1 para alcista, -1 para bajista, 0 para sin fractal
        df[f"{base_name}_direction"] = np.where(
            bullish_fractal == 1, 1, np.where(
                bearish_fractal == 1, -1, 0
            )
        )
        
        # 5. Fortaleza del fractal (diferencia porcentual entre el fractal y el precio anterior)
        df[f"{base_name}_bullish_strength"] = np.where(
            bullish_fractal == 1, 
            (df[high_col] / df[close_col].shift(1) - 1) * 100,
            np.nan
        )
        
        df[f"{base_name}_bearish_strength"] = np.where(
            bearish_fractal == 1, 
            (1 - df[low_col] / df[close_col].shift(1)) * 100,
            np.nan
        )
        
        # 6. Líneas de tendencia basadas en fractales
        # Forward fill los precios de los fractales para crear líneas de tendencia
        df[f"{base_name}_bull_trend_line"] = df[f"{base_name}_bullish_price"].fillna(method='ffill')
        df[f"{base_name}_bear_trend_line"] = df[f"{base_name}_bearish_price"].fillna(method='ffill')
        
        # 7. Señales de ruptura de líneas de tendencia
        df[f"{base_name}_bull_break"] = (df[close_col] > df[f"{base_name}_bull_trend_line"]).astype(int)
        df[f"{base_name}_bear_break"] = (df[close_col] < df[f"{base_name}_bear_trend_line"]).astype(int)
        
        # 8. Canales de fractales (líneas de tendencia superior e inferior)
        df[f"{base_name}_upper_channel"] = df[f"{base_name}_bullish_price"].rolling(lookback).max().fillna(method='ffill')
        df[f"{base_name}_lower_channel"] = df[f"{base_name}_bearish_price"].rolling(lookback).min().fillna(method='ffill')
        
        # 9. Posición relativa dentro del canal de fractales (0 a 1)
        channel_height = df[f"{base_name}_upper_channel"] - df[f"{base_name}_lower_channel"]
        df[f"{base_name}_channel_position"] = (df[close_col] - df[f"{base_name}_lower_channel"]) / channel_height.replace(0, np.nan)
        
        # 10. Fractales consecutivos del mismo tipo
        df[f"{base_name}_consecutive_bull"] = ((df[f"{base_name}_direction"] == 1) & 
                                              (df[f"{base_name}_direction"].shift(1) == 1)).astype(int)
        df[f"{base_name}_consecutive_bear"] = ((df[f"{base_name}_direction"] == -1) & 
                                              (df[f"{base_name}_direction"].shift(1) == -1)).astype(int)
        
        # 11. Divergencias entre precio y fractales
        # Bullish divergence: precio hace mínimos más bajos, pero fractales hacen mínimos más altos
        last_bearish_prices = df[f"{base_name}_bearish_price"].fillna(method='ffill')
        
        df[f"{base_name}_bull_divergence"] = (
            (df[close_col] < df[close_col].shift(lookback)) & 
            (last_bearish_prices > last_bearish_prices.shift(lookback)) &
            (last_bearish_prices > 0)  # Asegurarse de que hay un fractal reciente
        ).astype(int)
        
        # Bearish divergence: precio hace máximos más altos, pero fractales hacen máximos más bajos
        last_bullish_prices = df[f"{base_name}_bullish_price"].fillna(method='ffill')
        
        df[f"{base_name}_bear_divergence"] = (
            (df[close_col] > df[close_col].shift(lookback)) & 
            (last_bullish_prices < last_bullish_prices.shift(lookback)) &
            (last_bullish_prices > 0)  # Asegurarse de que hay un fractal reciente
        ).astype(int)
        
        # 12. Frecuencia de fractales (conteo en ventana móvil)
        df[f"{base_name}_bull_frequency"] = df[f"{base_name}_bullish"].rolling(lookback).sum()
        df[f"{base_name}_bear_frequency"] = df[f"{base_name}_bearish"].rolling(lookback).sum()
        df[f"{base_name}_total_frequency"] = df[f"{base_name}_bull_frequency"] + df[f"{base_name}_bear_frequency"]
        
        # 13. Señal de compra: precio supera un fractal alcista reciente
        df[f"{base_name}_buy_signal"] = ((df[close_col] > df[f"{base_name}_bullish_price"].shift(1)) & 
                                         (df[f"{base_name}_bullish"].shift(1) == 1)).astype(int)
        
        # 14. Señal de venta: precio cae por debajo de un fractal bajista reciente
        df[f"{base_name}_sell_signal"] = ((df[close_col] < df[f"{base_name}_bearish_price"].shift(1)) & 
                                          (df[f"{base_name}_bearish"].shift(1) == 1)).astype(int)
        
        # 15. Nivel de soporte (último fractal bajista)
        df[f"{base_name}_support"] = df[f"{base_name}_bearish_price"].fillna(method='ffill')
        
        # 16. Nivel de resistencia (último fractal alcista)
        df[f"{base_name}_resistance"] = df[f"{base_name}_bullish_price"].fillna(method='ffill')
        
        # 17. Distancia a soporte/resistencia (en pips)
        df[f"{base_name}_dist_to_support"] = (df[close_col] - df[f"{base_name}_support"]) * 10000
        df[f"{base_name}_dist_to_resistance"] = (df[f"{base_name}_resistance"] - df[close_col]) * 10000
        
        # 18. Patrones de fractales (secuencia de los últimos 3 fractales)
        # Codificamos: 1=alcista, -1=bajista, usamos ventana deslizante para los últimos 3
        directions = df[f"{base_name}_direction"]
        pattern_matrix = np.zeros((len(df), 3))  # Almacena los últimos 3 fractales
        
        last_fractals = []  # Almacena los últimos 3 tipos de fractales
        
        for i in range(len(df)):
            if directions[i] != 0:  # Si es un fractal
                last_fractals.append(directions[i])
                if len(last_fractals) > 3:
                    last_fractals.pop(0)
            
            # Rellenar la matriz con los valores disponibles
            for j in range(min(3, len(last_fractals))):
                pattern_matrix[i, j] = last_fractals[-(j+1)]
        
        # Crear columnas de patrón
        df[f"{base_name}_pattern_1"] = pattern_matrix[:, 0]  # Fractal más reciente
        df[f"{base_name}_pattern_2"] = pattern_matrix[:, 1]  # Segundo fractal más reciente
        df[f"{base_name}_pattern_3"] = pattern_matrix[:, 2]  # Tercer fractal más reciente
        
        # 19. Identificar patrones específicos de fractales
        # Patrón alcista: 3 fractales bajistas consecutivos
        df[f"{base_name}_triple_bottom"] = ((df[f"{base_name}_pattern_1"] == -1) & 
                                            (df[f"{base_name}_pattern_2"] == -1) & 
                                            (df[f"{base_name}_pattern_3"] == -1)).astype(int)
        
        # Patrón bajista: 3 fractales alcistas consecutivos
        df[f"{base_name}_triple_top"] = ((df[f"{base_name}_pattern_1"] == 1) & 
                                         (df[f"{base_name}_pattern_2"] == 1) & 
                                         (df[f"{base_name}_pattern_3"] == 1)).astype(int)
        
        # Patrón de reversa: alcista-bajista-alcista
        df[f"{base_name}_v_reversal"] = ((df[f"{base_name}_pattern_1"] == 1) & 
                                         (df[f"{base_name}_pattern_2"] == -1) & 
                                         (df[f"{base_name}_pattern_3"] == 1)).astype(int)
        
        # Patrón de reversa: bajista-alcista-bajista
        df[f"{base_name}_inverted_v"] = ((df[f"{base_name}_pattern_1"] == -1) & 
                                         (df[f"{base_name}_pattern_2"] == 1) & 
                                         (df[f"{base_name}_pattern_3"] == -1)).astype(int)
    
    # No eliminamos filas con NaN, siguiendo la política establecida
    return df



