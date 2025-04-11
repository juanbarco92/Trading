import numpy as np
import pandas as pd 
from scipy.signal import find_peaks
import json
import datetime

def detect_valleys_in_low(df, prominence=0.0001, distance=5, time_reference=None):
    """Detecta valles (soportes) en la columna 'low' e incluye el timestamp."""
    low_array = df['low'].values
    # Invertimos los valores para encontrar los valles como picos invertidos
    valley_indices, properties = find_peaks(-low_array, prominence=prominence, distance=distance)
    prominences = properties["prominences"]
    
    result = []
    for idx, prom in zip(valley_indices, prominences):
        # Convertir índice local a timestamp si está disponible
        timestamp = None
        if time_reference is not None:
            # Asegurarse de que el acceso a time_reference sea seguro
            try:
                if isinstance(time_reference, pd.DatetimeIndex):
                    # Si es un DatetimeIndex, conseguir el valor directamente
                    timestamp = time_reference[idx]
                elif isinstance(time_reference, pd.Series):
                    # Si es una Series, acceder al valor del índice específico
                    if idx in time_reference.index:
                        timestamp = time_reference.iloc[idx]
                    else:
                        # Usar una estrategia alternativa si el índice no existe
                        timestamp = None
            except (KeyError, IndexError):
                # Si ocurre un error al acceder al índice, establecer timestamp a None
                timestamp = None
        
        # Incluir timestamp en la tupla de retorno
        result.append((int(idx), low_array[idx], float(prom), 'S', timestamp))
    
    return result

def detect_peaks_in_high(df, prominence=0.0001, distance=5, time_reference=None):
    """Detecta picos en la columna 'high' e incluye el timestamp."""
    high_array = df['high'].values
    peak_indices, properties = find_peaks(high_array, prominence=prominence, distance=distance)
    prominences = properties["prominences"]
    
    result = []
    for idx, prom in zip(peak_indices, prominences):
        # Convertir índice local a timestamp si está disponible
        timestamp = None
        if time_reference is not None:
            # Asegurarse de que el acceso a time_reference sea seguro
            try:
                if isinstance(time_reference, pd.DatetimeIndex):
                    # Si es un DatetimeIndex, conseguir el valor directamente
                    timestamp = time_reference[idx]
                elif isinstance(time_reference, pd.Series):
                    # Si es una Series, acceder al valor del índice específico
                    if idx in time_reference.index:
                        timestamp = time_reference.iloc[idx]
                    else:
                        # Usar una estrategia alternativa si el índice no existe
                        timestamp = None
            except (KeyError, IndexError):
                # Si ocurre un error al acceder al índice, establecer timestamp a None
                timestamp = None
        
        # Incluir timestamp en la tupla de retorno
        result.append((int(idx), high_array[idx], float(prom), 'R', timestamp))
    
    return result

def detect_valleys_in_close(df, prominence=0.0001, distance=5, time_reference=None):
    """Detecta valles (soportes) en la columna 'close' e incluye el timestamp."""
    close_array = df['close'].values
    # Invertimos los valores para encontrar los valles como picos invertidos
    valley_indices, properties = find_peaks(-close_array, prominence=prominence, distance=distance)
    prominences = properties["prominences"]
    
    result = []
    for idx, prom in zip(valley_indices, prominences):
        # Convertir índice local a timestamp si está disponible
        timestamp = None
        if time_reference is not None:
            # Asegurarse de que el acceso a time_reference sea seguro
            try:
                if isinstance(time_reference, pd.DatetimeIndex):
                    # Si es un DatetimeIndex, conseguir el valor directamente
                    timestamp = time_reference[idx]
                elif isinstance(time_reference, pd.Series):
                    # Si es una Series, acceder al valor del índice específico
                    if idx in time_reference.index:
                        timestamp = time_reference.iloc[idx]
                    else:
                        # Usar una estrategia alternativa si el índice no existe
                        timestamp = None
            except (KeyError, IndexError):
                # Si ocurre un error al acceder al índice, establecer timestamp a None
                timestamp = None
        
        # Incluir timestamp en la tupla de retorno
        result.append((int(idx), close_array[idx], float(prom), 'S', timestamp))
    
    return result

def detect_peaks_in_close(df, prominence=0.0001, distance=5, time_reference=None):
    """Detecta picos en la columna 'close' e incluye el timestamp."""
    close_array = df['close'].values
    peak_indices, properties = find_peaks(close_array, prominence=prominence, distance=distance)
    prominences = properties["prominences"]
    
    result = []
    for idx, prom in zip(peak_indices, prominences):
        # Convertir índice local a timestamp si está disponible
        timestamp = None
        if time_reference is not None:
            # Asegurarse de que el acceso a time_reference sea seguro
            try:
                if isinstance(time_reference, pd.DatetimeIndex):
                    # Si es un DatetimeIndex, conseguir el valor directamente
                    timestamp = time_reference[idx]
                elif isinstance(time_reference, pd.Series):
                    # Si es una Series, acceder al valor del índice específico
                    if idx in time_reference.index:
                        timestamp = time_reference.iloc[idx]
                    else:
                        # Usar una estrategia alternativa si el índice no existe
                        timestamp = None
            except (KeyError, IndexError):
                # Si ocurre un error al acceder al índice, establecer timestamp a None
                timestamp = None
        
        # Incluir timestamp en la tupla de retorno
        result.append((int(idx), close_array[idx], float(prom), 'R', timestamp))
    
    return result

def cluster_all_points_dynamic(points, df, pips_range=0.001, reaction_window=50, current_index=None):
    """
    Agrupa todos los puntos (picos y valles) en clusters con centro dinámico,
    teniendo en cuenta los cambios de rol.
    """
    if not points:
        return []
    
    revalidation_threshold = 50  # Número de velas para considerar revalidación
    
    # Ordenamos por precio
    points_sorted = sorted(points, key=lambda x: x[1])
    clusters = []
    
    # Ahora manejamos puntos con timestamp (5 elementos)
    current_cluster = [points_sorted[0]]
    cluster_center = points_sorted[0][1]
    last_point_index = points_sorted[0][0]
    last_point_type = points_sorted[0][3]
    last_point_timestamp = points_sorted[0][4] if len(points_sorted[0]) > 4 else None
    revalidation_count = 0
    role_changes = 0  # Contador de cambios de rol
    
    for i in range(1, len(points_sorted)):
        # Extraer componentes del punto, incluyendo timestamp si existe
        if len(points_sorted[i]) > 4:
            idx, price, prom, point_type, timestamp = points_sorted[i]
        else:
            idx, price, prom, point_type = points_sorted[i]
            timestamp = None
            
        dist = abs(price - cluster_center)
        
        if dist <= pips_range:
            # Si el tipo es diferente al último punto, hay un cambio de rol
            if point_type != last_point_type:
                role_changes += 1
                
            if idx - last_point_index > revalidation_threshold:
                revalidation_count += 1
                
            # Guardamos el punto completo con su timestamp
            current_cluster.append(points_sorted[i])
            cluster_center = np.mean([p[1] for p in current_cluster])
            last_point_index = idx
            last_point_type = point_type
            if timestamp is not None:
                last_point_timestamp = timestamp
        else:
            # Procesar el cluster actual
            process_cluster(current_cluster, clusters, df, current_index, reaction_window,
                           role_changes, last_point_timestamp)
            
            # Comenzar un nuevo cluster
            current_cluster = [points_sorted[i]]
            cluster_center = price
            last_point_index = idx
            last_point_type = point_type
            last_point_timestamp = timestamp
            revalidation_count = 0
            role_changes = 0
    
    # Procesar el último cluster si existe
    if current_cluster:
        process_cluster(current_cluster, clusters, df, current_index, reaction_window,
                       role_changes, last_point_timestamp)
    
    # Ordenar clusters por centro
    clusters.sort(key=lambda c: c['cluster_center'])
    return clusters

def process_cluster(current_cluster, clusters, df, current_index, reaction_window,
                   role_changes, last_point_timestamp=None):
    """
    Procesa un cluster y calcula sus métricas, incluyendo el timestamp del último toque.
    """
    indices = [p[0] for p in current_cluster]
    prices = [p[1] for p in current_cluster]
    prominences = [p[2] for p in current_cluster]
    types = [p[3] for p in current_cluster]
    
    # Extraer timestamps si existen
    timestamps = []
    for point in current_cluster:
        if len(point) > 4 and point[4] is not None:
            timestamps.append(point[4])
    
    touch_count = len(current_cluster)
    prominence_mean = np.mean(prominences)
    duration = max(indices) - min(indices)
    max_idx = max(indices)
    
    # Determinamos el tipo predominante del cluster
    r_count = types.count('R')
    s_count = types.count('S')
    predominant_type = 'R' if r_count >= s_count else 'S'
    
    # Calcular last_touch_age
    last_touch_age = current_index - max_idx if current_index else 0
    
    # Llamar a la función de depuración antes del cálculo de magnitud 
    cluster_center = float(np.mean(prices))
    debug_reaction_magnitude(cluster_center, predominant_type, max_idx, df, reaction_window, prices)
    
    # Calcular reaction_magnitude según el tipo predominante
    start = max_idx + 1
    end = min(start + reaction_window, len(df))
    if start >= len(df):
        reaction_magnitude = 0
    else:
        window_prices = df.iloc[start:end]['close'].values
        if predominant_type == 'R':
            reaction_magnitude = float(np.mean(prices)) - np.min(window_prices)
        else:
            reaction_magnitude = np.max(window_prices) - float(np.mean(prices))
        reaction_magnitude = max(reaction_magnitude, 0)
    
    # Determinar el timestamp del último toque
    last_touch_timestamp = None
    if timestamps:
        # Encontrar el timestamp correspondiente al punto con índice máximo
        max_idx_point = next((p for p in current_cluster if p[0] == max_idx and len(p) > 4), None)
        if max_idx_point and len(max_idx_point) > 4:
            last_touch_timestamp = max_idx_point[4]
        else:
            # Si no se encuentra, usar el último timestamp disponible
            last_touch_timestamp = timestamps[-1]
    else:
        # Si no hay timestamps, usar el que se pasó como parámetro
        last_touch_timestamp = last_point_timestamp
    
    cluster_info = {
        'cluster_center': float(np.mean(prices)),
        'touch_count': touch_count,
        'revalidation_count': len(current_cluster) - 1,
        'duration': duration,
        'prominence_mean': prominence_mean,
        'reaction_magnitude': reaction_magnitude,
        'last_touch_age': last_touch_age,
        'role_changes': role_changes,
        'predominant_type': predominant_type,
        'min_idx': min(indices),
        'max_idx': max_idx
    }
    
    # Agregar timestamp del último toque si está disponible
    if last_touch_timestamp is not None:
        cluster_info['last_touch_timestamp'] = last_touch_timestamp
    
    clusters.append(cluster_info)

def calculate_progressive_support_resistance(
    df,
    initial_window=5000,
    step_size=100,
    lookback_window=10000,
    prominence=0.0010,
    distance=10,
    pips_range=0.001,
    reaction_window=50,
    merge_threshold=0.0002,
    use_hl = True,
    add_relevant_levels=True,  # Nuevo parámetro para activar la funcionalidad
    n_closest=5,                # Número de niveles cercanos en precio a considerar
    n_recent=5   
):
    """
    Calcula soportes y resistencias progresivamente usando una ventana deslizante,
    manteniendo todos los niveles activos en el historial.
    """
    # Configuración inicial y manejo temporal
    FAR_FUTURE_DATE = pd.Timestamp('2099-12-31 23:59:59', tz='US/Eastern')
    
    df_working = df.copy()
    
    # Manejo de información temporal
    if isinstance(df_working.index, pd.DatetimeIndex):
        has_time_index = True
        time_reference = df_working.index
    else:
        has_time_index = False
        if 'datetime' in df_working.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_working['datetime']):
                df_working['datetime'] = pd.to_datetime(df_working['datetime'])
            time_reference = df_working['datetime']
        elif 'timestamp' in df_working.columns:
            df_working['datetime'] = pd.to_datetime(df_working['timestamp'], unit='ms')
            time_reference = df_working['datetime']
        else:
            raise ValueError("ERROR: No se encontró información temporal válida en el DataFrame.")
    
    # Diccionario para seguimiento de niveles
    active_levels = {}  # Por ID
    next_id = 1  # Contador para IDs únicos
    levels_history = []  # Para almacenar el historial completo
    
    print(f"Iniciando procesamiento progresivo con {len(df_working)} velas.")
    
    for end_idx in range(initial_window, len(df_working) + 1, step_size):
        actual_end = min(end_idx, len(df_working))
        start_idx = max(0, actual_end - lookback_window)
        current_timestamp = time_reference[actual_end - 1]
        
        # Procesar ventana deslizante y detectar puntos
        current_slice = df_working.iloc[start_idx:actual_end].copy()

        # Crear una referencia temporal para los puntos dentro de la ventana
        if has_time_index:
            # Si el DataFrame original tiene un índice datetime, usar el índice del slice
            slice_time_reference = current_slice.index
        else:
            # Si el DataFrame tiene una columna datetime, usar esa columna como referencia
            if 'datetime' in current_slice.columns:
                # Importante: reset_index para que los índices coincidan con las posiciones
                slice_time_reference = current_slice['datetime'].reset_index(drop=True)
            else:
                slice_time_reference = None

        # Llamar a las funciones modificadas pasando la referencia temporal
        peaks_close = detect_peaks_in_close(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)
        valleys_close = detect_valleys_in_close(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)
        
        # Detectar puntos adicionales usando high/low si está habilitado
        if use_hl:
            peaks_high = detect_peaks_in_high(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)
            valleys_low = detect_valleys_in_low(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)
            
            # Combinar todos los puntos
            all_points = peaks_close + valleys_close + peaks_high + valleys_low
        else:
            # Solo usar los puntos basados en close
            all_points = peaks_close + valleys_close


        all_points.sort(key=lambda x: x[0])
        # Ajustar el índice pero mantener el timestamp
        all_points = [(idx + start_idx, price, prom, type_, timestamp) for idx, price, prom, type_, timestamp in all_points]
        
        # Seguimiento de cambios en este ciclo
        updated_in_cycle = set()
        
        if all_points:
            # Clusterizar puntos
            local_clusters = cluster_all_points_dynamic(
                all_points, df_working, pips_range=pips_range, 
                reaction_window=reaction_window, current_index=actual_end-1
            )
            
            # Niveles actualizados para este ciclo
            current_levels = {}
            
            # 1. Procesar los nuevos clusters
            for new_cluster in local_clusters:
                matched_id = None
                best_distance = float('inf')
                
                # Buscar si coincide con algún nivel existente
                for level_id, existing_level in active_levels.items():
                    dist = abs(new_cluster['cluster_center'] - existing_level['price'])
                    if dist <= merge_threshold and dist < best_distance:
                        matched_id = level_id
                        best_distance = dist
                
                # Depurar el nivel específico
                debug_specific_level(new_cluster['cluster_center'], new_cluster, 
                                    active_levels.get(matched_id) if matched_id else None,
                                    "EVALUANDO COINCIDENCIA")
                
                # Actualizar nivel existente o crear uno nuevo
                if matched_id:
                    # Nivel existente - actualizar propiedades
                    existing = active_levels[matched_id]
                            
                    # CLAVE: Calcular si realmente hay toques nuevos
                    # Compara los rangos de índices para ver si hay toques no contabilizados anteriormente
                    has_new_touches = new_cluster['max_idx'] > existing['max_idx']
        
                    if has_new_touches:
                        debug_specific_level(existing['price'], new_cluster, existing, "ACTUALIZANDO - HAY NUEVOS TOQUES")
            
                        # Crear copia del registro anterior con fecha de fin
                        hist_record = existing.copy()
                        hist_record['valid_to_dt'] = current_timestamp
                        hist_record['is_current'] = False
                        levels_history.append(hist_record)
                        
                        # Actualizar nivel con nuevas propiedades usando lógicas consistentes
                        updated = {
                            'id': matched_id,
                            
                            # Precio: promedio ponderado por cantidad de toques
                            'price': (existing['price'] * existing['touch_count'] + new_cluster['cluster_center'] * new_cluster['touch_count']) / 
                                    (existing['touch_count'] + new_cluster['touch_count']),
                            
                            # Touch count: suma (asumimos que son conjuntos diferentes de toques)
                            'touch_count': existing['touch_count'] + new_cluster['touch_count'],
                            
                            # Revalidaciones: suma (son eventos independientes)
                            'revalidation_count': existing['revalidation_count'] + new_cluster['revalidation_count'],
                            
                            # Duración: desde el primer toque hasta el último
                            'duration': max(existing['max_idx'], new_cluster['max_idx']) - min(existing['min_idx'], new_cluster['min_idx']),
                            
                            # Prominencia media: promedio ponderado por toques
                            'prominence_mean': (existing['prominence_mean'] * existing['touch_count'] + 
                                            new_cluster['prominence_mean'] * new_cluster['touch_count']) /
                                            (existing['touch_count'] + new_cluster['touch_count']),
                            
                            # Reacción: usar la más significativa
                            'reaction_magnitude': max(existing['reaction_magnitude'], new_cluster['reaction_magnitude']),
                            
                            # Cambios de rol: suma (son eventos acumulativos)
                            'role_changes': existing['role_changes'] + new_cluster.get('role_changes', 0),
                            
                            # Índices de rango: mínimo y máximo combinados
                            'min_idx': min(existing['min_idx'], new_cluster['min_idx']),
                            'max_idx': max(existing['max_idx'], new_cluster['max_idx']),
                            
                            # Mantener datos de origen originales
                            'origin_date': existing['origin_date'],
                            'origin_idx': existing['origin_idx'],
                            'type': existing['type'],
                            
                            # Fechas de validez y última actualización
                            'valid_from_dt': current_timestamp,
                            'valid_to_dt': FAR_FUTURE_DATE,
                            'last_update_dt': current_timestamp,
                            'last_touch_dt': new_cluster.get('last_touch_timestamp', current_timestamp),  # Usamos el timestamp del último toque
                            'is_current': True
                        }
                        
                        current_levels[matched_id] = updated
                        updated_in_cycle.add(matched_id)
                    else:
                        # Si no hay toques nuevos, mantener el nivel sin cambios
                        debug_specific_level(existing['price'], new_cluster, existing, "MANTENIENDO - NO HAY TOQUES NUEVOS")
                        current_levels[matched_id] = existing
                        updated_in_cycle.add(matched_id)
                            
                else:
                    debug_specific_level(new_cluster['cluster_center'], new_cluster, None, "NUEVO NIVEL ADICIONADO")
                    # Nuevo nivel - crear registro desde cero
                    origin_idx = new_cluster['max_idx']
                    # Verifica que el índice esté dentro del rango válido
                    if 0 <= origin_idx < len(time_reference):
                        origin_date = time_reference[origin_idx]
                        
                        # Asegúrate de que last_touch_dt refleje la fecha del último toque real
                        last_touch_idx = new_cluster['max_idx']
                        last_touch_dt = time_reference[last_touch_idx] if 0 <= last_touch_idx < len(time_reference) else current_timestamp
                    else:
                        # Usar una fecha por defecto o la actual si el índice está fuera de rango
                        origin_date = current_timestamp
                        last_touch_dt = current_timestamp
                    
                    new_level = {
                        'id': next_id,
                        'price': new_cluster['cluster_center'],
                        'touch_count': new_cluster['touch_count'],
                        'revalidation_count': new_cluster['revalidation_count'],
                        'duration': new_cluster['duration'],
                        'prominence_mean': new_cluster['prominence_mean'],
                        'reaction_magnitude': new_cluster['reaction_magnitude'],
                        'role_changes': new_cluster.get('role_changes', 0),
                        'min_idx': new_cluster['min_idx'],
                        'max_idx': new_cluster['max_idx'],
                        'origin_date': origin_date,
                        'origin_idx': origin_idx,
                        'type': 'Soporte' if new_cluster['predominant_type'] == 'S' else 'Resistencia',
                        'valid_from_dt': current_timestamp,
                        'valid_to_dt': FAR_FUTURE_DATE,
                        'last_update_dt': current_timestamp,
                        'last_touch_dt': new_cluster.get('last_touch_timestamp', current_timestamp),  # Usamos el timestamp del último toque
                        'is_current': True
                    }

                    
                    current_levels[next_id] = new_level
                    updated_in_cycle.add(next_id)
                    next_id += 1
            
            # Mantener los niveles antiguos que no fueron actualizados
            for level_id, existing_level in active_levels.items():
                if level_id not in updated_in_cycle:
                    current_levels[level_id] = existing_level

            # Actualizar el diccionario de niveles activos
            active_levels = current_levels
        
        # Mostrar progreso
        if end_idx % 1000 == 0 or actual_end == len(df_working):
            completion = actual_end / len(df_working) * 100
            print(f"Procesadas {actual_end} de {len(df_working)} velas ({completion:.1f}%)")
            print(f"Niveles activos: {len(active_levels)}")
    
    # Al finalizar, agregar todos los niveles activos al historial
    for level in active_levels.values():
        levels_history.append(level)
    
    # Convertir el historial a DataFrame
    history_df = pd.DataFrame(levels_history)

    date_columns = ['valid_from_dt', 'valid_to_dt', 'last_update_dt', 'last_touch_dt', 'origin_date']
    for col in date_columns:
        if col in history_df.columns:
            # Primero convertir a datetime
            history_df[col] = pd.to_datetime(history_df[col], errors='coerce')
            
            # Verificar si ya tiene zona horaria
            if history_df[col].dt.tz is None:
                # Si no tiene zona horaria, asumimos que es UTC y convertimos a US/Eastern
                history_df[col] = history_df[col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                # Si ya tiene zona horaria, solo convertir a US/Eastern
                history_df[col] = history_df[col].dt.tz_convert('US/Eastern')
    
    # Si se solicita añadir niveles relevantes
    if add_relevant_levels:
        print("Calculando niveles relevantes para cada vela...")
        df_with_levels = add_relevant_levels_to_df(df_working, history_df, 2, 6)
        return history_df, df_with_levels
    

    return history_df

# Añade esta función para depurar específicamente el nivel 1.3022
def debug_specific_level(level_price, new_cluster, matched_level=None, operation=None):
    # Definir el nivel específico a rastrear
    TARGET_LEVEL = 1.3022
    TOLERANCE = 0.0002
    
    # Solo depurar si estamos cerca del nivel objetivo
    if abs(level_price - TARGET_LEVEL) <= TOLERANCE:
        print(f"\n--- DEBUG NIVEL {TARGET_LEVEL} [{operation}] ---")
        print(f"Precio del nivel: {level_price}")
        
        if matched_level:
            print(f"Nivel existente ID: {matched_level['id']}")
            print(f"Toques acumulados: {matched_level['touch_count']}")
            print(f"Última fecha de toque: {matched_level.get('last_touch_dt')}")
        
        print(f"Nuevo cluster encontrado:")
        print(f"  - Centro: {new_cluster['cluster_center']}")
        print(f"  - Toques: {new_cluster['touch_count']}")
        print(f"  - Índice máx: {new_cluster['max_idx']}")
        if 'last_touch_timestamp' in new_cluster:
            print(f"  - Último toque: {new_cluster['last_touch_timestamp']}")
        print("-----------------------------------\n")

def debug_reaction_magnitude(cluster_center, predominant_type, max_idx, df, reaction_window, prices=None):
    """
    Depura el cálculo de reaction_magnitude para un nivel específico.
    """
    TARGET_LEVEL = 1.438133
    TOLERANCE = 0.0002
    
    if abs(cluster_center - TARGET_LEVEL) <= TOLERANCE:
        print(f"\n--- DEBUG REACTION MAGNITUDE PARA NIVEL {cluster_center:.6f} ---")
        print(f"Tipo predominante: {predominant_type}")
        print(f"Índice máximo: {max_idx}")
        
        # Verificar si hay datos futuros disponibles
        start = max_idx + 1
        end = min(start + reaction_window, len(df))
        
        print(f"Ventana de reacción: {start} a {end}")
        print(f"¿Hay datos futuros? {'No' if start >= len(df) else 'Sí'}")
        
        if start < len(df):
            window_prices = df.iloc[start:end]['close'].values
            print(f"Precios futuros: {window_prices[:5]}{'...' if len(window_prices) > 5 else ''}")
            print(f"Min futuro: {np.min(window_prices):.6f}, Max futuro: {np.max(window_prices):.6f}")
            
            # Si tenemos el precio medio del nivel
            if prices is not None:
                mean_price = float(np.mean(prices))
                print(f"Precio medio del nivel: {mean_price:.6f}")
                
                if predominant_type == 'R':
                    reaction = mean_price - np.min(window_prices)
                    print(f"Cálculo (R): {mean_price:.6f} - {np.min(window_prices):.6f} = {reaction:.6f}")
                else:
                    reaction = np.max(window_prices) - mean_price
                    print(f"Cálculo (S): {np.max(window_prices):.6f} - {mean_price:.6f} = {reaction:.6f}")
                
                print(f"Magnitud final (max con 0): {max(reaction, 0):.6f}")
        
        print("-----------------------------------\n")



def add_relevant_levels_to_df(df, history_df, n_closest=5, n_recent=5):
    """
    Añade una columna con los niveles más relevantes para cada fila del dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con datos OHLC
    history_df : pandas.DataFrame
        DataFrame con el historial de niveles de soporte y resistencia
    n_closest : int, default 5
        Número de niveles más cercanos en precio a incluir
    n_recent : int, default 5
        Número de niveles más recientes a incluir

    Returns:
    --------
    pandas.DataFrame
        DataFrame original con columna adicional 'relevant_levels'
    """
    import json
    import pandas as pd

    # Crear una copia para no modificar el original
    df_result = df.copy()

    # Determinar la columna de tiempo
    time_col = None
    if 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'

    # Si no hay columna de tiempo pero hay índice de tiempo, extraerlo
    if time_col is None and isinstance(df.index, pd.DatetimeIndex):
        df_result['datetime'] = df.index
        time_col = 'datetime'

    if time_col is None:
        raise ValueError("No se encontró información temporal en el dataframe")

    # Preparar la columna de niveles relevantes
    relevant_levels_list = []
    total_rows = len(df_result)

    print(f"Procesando {total_rows} filas...")

    # Procesar por bloques para mejorar el rendimiento
    chunk_size = 1000
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        print(f"Procesando filas {start_idx} a {end_idx}...")

        for i in range(start_idx, end_idx):
            row = df_result.iloc[i]

            # Obtener fecha y precio
            current_dt = row[time_col]
            price = row['close']


            # Filtrar niveles válidos para esta fecha
            valid_levels = history_df[
                (history_df['valid_from_dt'] <= current_dt) & 
                (history_df['valid_to_dt'] >= current_dt)
            ]

            if valid_levels.empty:
                # No hay niveles válidos para esta fecha
                relevant_levels_list.append(json.dumps({
                    "closest": [],
                    "recent": []
                }))
                continue

            # Calcular distancia al precio actual
            valid_levels = valid_levels.copy()  # Para evitar SettingWithCopyWarning
            valid_levels['price_distance'] = abs(valid_levels['price'] - price)

            # Calcular velas transcurridas desde el último toque en minutos
            valid_levels['bars_since_touch'] = (current_dt - valid_levels['last_touch_dt']).dt.total_seconds() / (60 * 5)

            # Calcular un score combinado para cada nivel (normalizado entre 0-100)
            # Pesos para cada componente - ajustar según necesidad
            w_touch = 0.3
            w_reaction = 0.3
            w_prominence = 0.2
            w_duration = 0.2

            # Normalizar los componentes (para que estén en rango similar)
            max_touch = max(valid_levels['touch_count'].max(), 1)
            max_reaction = max(valid_levels['reaction_magnitude'].max(), 0.001)
            max_prominence = max(valid_levels['prominence_mean'].max(), 0.001)
            max_duration = max(valid_levels['duration'].max(), 1)

            valid_levels['importance_score'] = (
                w_touch * (valid_levels['touch_count'] / max_touch) * 100 +
                w_reaction * (valid_levels['reaction_magnitude'] / max_reaction) * 100 +
                w_prominence * (valid_levels['prominence_mean'] / max_prominence) * 100 +
                w_duration * (valid_levels['duration'] / max_duration) * 100
            )

            # Obtener los niveles más cercanos en precio
            closest_levels = valid_levels.sort_values('price_distance').head(n_closest)

            # Obtener los niveles más recientes (ordenados por last_touch_dt descendente)
            recent_levels = valid_levels.sort_values('last_touch_dt', ascending=False).head(n_recent)

            # Función auxiliar para crear un diccionario con todos los atributos requeridos
            def level_to_dict(level):
                return {
                    'id': int(level['id']),
                    'price': float(level['price']),
                    'type': level['type'],
                    'touch_count': int(level['touch_count']),
                    'distance': float(level['price_distance']),
                    'bars_since_touch': int(round(level['bars_since_touch'])),
                    'last_touch': level['last_touch_dt'].strftime('%Y-%m-%d %H:%M:%S'),
                    'reaction_magnitude': float(level['reaction_magnitude']),
                    'prominence': float(level['prominence_mean']),
                    'duration': int(level['duration']),
                    'importance_score': float(level['importance_score'])
                }

            # Convertir a formato serializable: Se permite que un mismo nivel aparezca en ambos grupos
            closest_list = [level_to_dict(level) for _, level in closest_levels.iterrows()]
            # En recent se conservan todos los niveles, incluso si se repite el mismo nivel varias veces
            recent_list = [level_to_dict(level) for _, level in recent_levels.iterrows()]

            # Crear el objeto JSON con ambas listas
            relevant_json = {
                "closest": closest_list,
                "recent": recent_list
            }

            # Añadir al listado
            relevant_levels_list.append(json.dumps(relevant_json))

    # Añadir la columna al DataFrame
    df_result['relevant_levels'] = relevant_levels_list

    print("Procesamiento completado.")
    return df_result


def calcular_metricas_fila(row):
    """
    Calcula las métricas definidas a partir de la información en la columna 'relevant_levels'
    para cada fila del DataFrame.
    """
    # Extraer el precio actual y el datetime de la fila
    precio_actual = row['close']
    dt_actual = row['datetime']  # Se asume que ya es un objeto datetime

    # Convertir la cadena JSON de la columna 'relevant_levels' en un diccionario
    try:
        niveles_dict = json.loads(row['relevant_levels'])
    except Exception as e:
        # En caso de error en el parseo, se retorna una serie con valores nulos
        return pd.Series({
            "direccion_ultimo": None,
            "fuerza_ultimo": None,
            "comportamiento": None,
            "posicion_precio_recent": None,
            "pips_desde_ultimo": None,
            "minutos_desde_ultimo_touch": None,
            "tendencia_recent": None,
            "pips_cercano": None,
            "minutos_desde_cercano": None,
            "nivel_en_recent": None,
            "posicion_nivel_cercano": None,
            "fuerza_nivel_cercano": None,
            "pips_nivel_arriba": None,
            "pips_nivel_abajo": None
        })

    recent_levels = niveles_dict.get("recent", [])
    closest_levels = niveles_dict.get("closest", [])

    # Si no existen los niveles necesarios, retorna valores nulos
    if not recent_levels or not closest_levels:
        return pd.Series({
            "direccion_ultimo": None,
            "fuerza_ultimo": None,
            "comportamiento": None,
            "posicion_precio_recent": None,
            "pips_desde_ultimo": None,
            "minutos_desde_ultimo_touch": None,
            "tendencia_recent": None,
            "pips_cercano": None,
            "minutos_desde_cercano": None,
            "nivel_en_recent": None,
            "posicion_nivel_cercano": None,
            "fuerza_nivel_cercano": None,
            "pips_nivel_arriba": None,
            "pips_nivel_abajo": None
        })

    # Convertir el campo 'last_touch' de cada nivel a datetime
    for nivel in recent_levels:
        nivel["last_touch_dt"] = datetime.datetime.strptime(nivel["last_touch"], "%Y-%m-%d %H:%M:%S")
    for nivel in closest_levels:
        nivel["last_touch_dt"] = datetime.datetime.strptime(nivel["last_touch"], "%Y-%m-%d %H:%M:%S")

    # 1. Variables usando los niveles 'recent'
    # Ordenamos por 'last_touch_dt' para obtener el nivel más reciente
    recent_ordenados = sorted(recent_levels, key=lambda x: x["last_touch_dt"])
    ultimo_recent = recent_ordenados[-1]

    # 1.1 Dirección tras el último toque:
    direccion = "hacia arriba" if precio_actual > ultimo_recent["price"] else "hacia abajo"

    # 1.2 Fuerza del último nivel (puedes combinar varias métricas o elegir una)
    fuerza_ultimo = ultimo_recent["importance_score"]

    # 1.3 Rebote vs. Ruptura: utilizando el último y el penúltimo nivel
    if len(recent_ordenados) >= 2:
        penultimo_recent = recent_ordenados[-2]
        if min(ultimo_recent["price"], penultimo_recent["price"]) < precio_actual < max(ultimo_recent["price"], penultimo_recent["price"]):
            comportamiento = "rebote"
        elif precio_actual > max(ultimo_recent["price"], penultimo_recent["price"]):
            comportamiento = "ruptura al alza"
        elif precio_actual < min(ultimo_recent["price"], penultimo_recent["price"]):
            comportamiento = "ruptura a la baja"
        else:
            comportamiento = "indeterminado"
    else:
        comportamiento = "indeterminado"

    # 1.4 Posición del precio respecto a los niveles recientes
    precios_recent = sorted([n["price"] for n in recent_levels])
    # Si la lista está ordenada de menor a mayor, la posición es:
    posicion = sum([1 for p in precios_recent if precio_actual > p]) + 1

    # 1.5 Distancia en pips al último nivel
    diff_ultimo = abs(precio_actual - ultimo_recent["price"])
    pips_ultimo = diff_ultimo / 0.0001

    # 1.6 Usar bars_since_touch directamente que ya está calculado
    minutos_ultimo = ultimo_recent["bars_since_touch"]

    # 1.7 Tendencia implícita en recent_levels: 
    precios_por_fecha = [n["price"] for n in recent_ordenados]
    if all(earlier >= later for earlier, later in zip(precios_por_fecha, precios_por_fecha[1:])):
        tendencia = "bajista"
    elif all(earlier <= later for earlier, later in zip(precios_por_fecha, precios_por_fecha[1:])):
        tendencia = "alcista"
    else:
        tendencia = "lateral"

    # 2. Variables usando los niveles 'closest'
    # 2.1 Seleccionar el nivel con la menor diferencia (distance)
    nivel_cercano = min(closest_levels, key=lambda x: x["distance"])
    pips_cercano = nivel_cercano["distance"] / 0.0001
    
    # 2.2 Usar bars_since_touch directamente que ya está calculado
    minutos_cercano = nivel_cercano["bars_since_touch"]

    # 2.3 Verificar si el nivel cercano aparece en recent_levels
    ids_recent = {n["id"] for n in recent_levels}
    en_recent = nivel_cercano["id"] in ids_recent

    # 2.4 Posición del nivel cercano respecto al precio
    posicion_nivel = "por encima" if nivel_cercano["price"] > precio_actual else "por debajo"

    # 2.5 Fuerza del nivel cercano (usando, por ejemplo, importance_score)
    fuerza_cercano = nivel_cercano["importance_score"]

    # 2.6 Nivel más cercano por encima del precio
    niveles_arriba = [n for n in closest_levels if n["price"] > precio_actual]
    pips_arriba = None
    if niveles_arriba:
        nivel_arriba = min(niveles_arriba, key=lambda x: x["price"] - precio_actual)
        pips_arriba = (nivel_arriba["price"] - precio_actual) / 0.0001

    # 2.7 Nivel más cercano por debajo del precio
    niveles_abajo = [n for n in closest_levels if n["price"] < precio_actual]
    pips_abajo = None
    if niveles_abajo:
        nivel_abajo = max(niveles_abajo, key=lambda x: x["price"])
        pips_abajo = (precio_actual - nivel_abajo["price"]) / 0.0001

    return pd.Series({
        "direccion_ultimo": direccion,
        "fuerza_ultimo": fuerza_ultimo,
        "comportamiento": comportamiento,
        "posicion_precio_recent": posicion,
        "pips_desde_ultimo": pips_ultimo,
        "minutos_desde_ultimo_touch": minutos_ultimo,
        "tendencia_recent": tendencia,
        "pips_cercano": pips_cercano,
        "minutos_desde_cercano": minutos_cercano,
        "nivel_en_recent": en_recent,
        "posicion_nivel_cercano": posicion_nivel,
        "fuerza_nivel_cercano": fuerza_cercano,
        "pips_nivel_arriba": pips_arriba,
        "pips_nivel_abajo": pips_abajo
    })

