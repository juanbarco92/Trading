import numpy as np
import pandas as pd 
from scipy.signal import find_peaks


# Funciones para análisis de soportes y resistencias
def calculate_mass(touch_count, revalidation_count, bonus_factor, last_touch_age, 
                  reaction_magnitude, prominence_mean, role_changes=0, role_change_bonus=1.5, 
                  gamma=5, delta=5, epsilon=0.5):
    """
    Calcula la masa del cluster usando suma ponderada de componentes, priorizando
    el número de toques y la antigüedad.
    """
    # Componente base: toques y revalidaciones (50% del peso total)
    base_component = gamma * (touch_count + (revalidation_count * bonus_factor))
    
    # Componente de antigüedad (30% del peso total)
    # Función exponencial que decae mucho más rápido con la antigüedad
    recency_factor = np.exp(-last_touch_age * 0.05)  # Decae muy rápidamente
    time_component = delta * recency_factor  
    
    # Componente de cambio de rol (10% del peso total)
    role_component = 1.0 * (role_changes * role_change_bonus)
    
    # Componente de reacción del mercado (5% del peso total)
    reaction_component = epsilon * reaction_magnitude * 100  # Escalamos para que sea significativo
    
    # Componente de prominencia (5% del peso total)
    prominence_component = 0.5 * prominence_mean * 100  # Escalamos para que sea significativo
    
    # Suma ponderada total
    mass = base_component + time_component + role_component + reaction_component + prominence_component
    
    return mass

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

def cluster_peaks_dynamic_center(peaks, df, pips_range=0.001, reaction_window=50, current_index=None, level_type='R', K_age=100, T_recency=50, bonus_factor=0.1, gamma=1.5, delta=0.001, epsilon=0.001):
    """
    Agrupa los picos en clusters con centro dinámico.
    Calcula `last_touch_age`, `reaction_magnitude` y `mass` dentro del mismo proceso.
    """
    if not peaks:
        return []
    
    revalidation_threshold = 50  # Número de velas para considerar revalidación
    
    peaks_sorted = sorted(peaks, key=lambda x: x[1])  # Ordenar por precio
    clusters = []
    current_cluster = [peaks_sorted[0]]
    cluster_center = peaks_sorted[0][1]
    last_peak_index = peaks_sorted[0][0]
    revalidation_count = 0
    
    for i in range(1, len(peaks_sorted)):
        idx, price, prom = peaks_sorted[i]
        dist = abs(price - cluster_center)
        
        if dist <= pips_range:
            if idx - last_peak_index > revalidation_threshold:
                revalidation_count += 1
            current_cluster.append((idx, price, prom))
            cluster_center = np.mean([p[1] for p in current_cluster])
            last_peak_index = idx
        else:
            indices = [p[0] for p in current_cluster]
            prices = [p[1] for p in current_cluster]
            prominences = [p[2] for p in current_cluster]
            touch_count = len(current_cluster)
            prominence_mean = np.mean(prominences)
            duration = max(indices) - min(indices)
            max_idx = max(indices)
            
            # Calcular last_touch_age dentro del mismo loop
            last_touch_age = current_index - max_idx if current_index else 0
            
            # Calcular reaction_magnitude dentro del mismo loop
            start = max_idx + 1
            end = min(start + reaction_window, len(df))
            if start >= len(df):
                reaction_magnitude = 0
            else:
                window_prices = df.iloc[start:end]['close'].values
                if level_type == 'R':
                    reaction_magnitude = cluster_center - np.min(window_prices)
                else:
                    reaction_magnitude = np.max(window_prices) - cluster_center
                reaction_magnitude = max(reaction_magnitude, 0)  # Evitar valores negativos
            
            # Calcular la masa dentro del mismo loop
            mass = calculate_mass(touch_count, revalidation_count, bonus_factor, last_touch_age,
                                  reaction_magnitude, prominence_mean, 0, 1.5,
                                  gamma, delta, epsilon)
            
            clusters.append({
                'cluster_center': float(np.mean(prices)),
                'touch_count': touch_count,
                'revalidation_count': revalidation_count,
                'duration': duration,
                'prominence_mean': prominence_mean,
                'reaction_magnitude': reaction_magnitude,
                'last_touch_age': last_touch_age,
                'mass': mass,  # Se guarda directamente
                'min_idx': min(indices),
                'max_idx': max_idx
            })
            
            current_cluster = [(idx, price, prom)]
            cluster_center = price
            last_peak_index = idx
            revalidation_count = 0
    
    if current_cluster:
        indices = [p[0] for p in current_cluster]
        prices = [p[1] for p in current_cluster]
        prominences = [p[2] for p in current_cluster]
        touch_count = len(current_cluster)
        prominence_mean = np.mean(prominences)
        duration = max(indices) - min(indices)
        max_idx = max(indices)
        last_touch_age = current_index - max_idx if current_index else 0

        start = max_idx + 1
        end = min(start + reaction_window, len(df))
        if start >= len(df):
            reaction_magnitude = 0
        else:
            window_prices = df.iloc[start:end]['close'].values
            if level_type == 'R':
                reaction_magnitude = cluster_center - np.min(window_prices)
            else:
                reaction_magnitude = np.max(window_prices) - cluster_center
            reaction_magnitude = max(reaction_magnitude, 0)

        mass = calculate_mass(touch_count, revalidation_count, bonus_factor, last_touch_age,
                              reaction_magnitude, prominence_mean, 0, 1.5,
                              gamma, delta, epsilon)

        clusters.append({
            'cluster_center': float(np.mean(prices)),
            'touch_count': touch_count,
            'revalidation_count': revalidation_count,
            'duration': duration,
            'prominence_mean': prominence_mean,
            'reaction_magnitude': reaction_magnitude,
            'last_touch_age': last_touch_age,
            'mass': mass,
            'min_idx': min(indices),
            'max_idx': max_idx
        })
    
    clusters.sort(key=lambda c: c['cluster_center'])
    return clusters

def cluster_all_points_dynamic(points, df, pips_range=0.001, reaction_window=50, current_index=None, 
                              bonus_factor=0.1, role_change_bonus=1.5, 
                              gamma=1.5, delta=0.001, epsilon=0.001):
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
                           role_changes, bonus_factor, role_change_bonus,
                           gamma, delta, epsilon, last_point_timestamp)
            
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
                       role_changes, bonus_factor, role_change_bonus,
                       gamma, delta, epsilon, last_point_timestamp)
    
    # Ordenar clusters por centro
    clusters.sort(key=lambda c: c['cluster_center'])
    return clusters

def process_cluster(current_cluster, clusters, df, current_index, reaction_window,
                   role_changes, bonus_factor, role_change_bonus,
                   gamma, delta, epsilon, last_point_timestamp=None):
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

def find_peaks_and_cluster_dynamic_center(
    df,
    max_rows=10000,
    prominence=0.0001,
    distance=5,
    pips_range=0.001,
    reaction_window=50,
    K_age=100,
    T_recency=50,
    bonus_factor=0.1,
    role_change_bonus=1.5,
    gamma=1.5,
    delta=0.001,
    epsilon=0.001
):
    """
    Procesa las primeras max_rows de df, detectando tanto soportes como resistencias.
    """
    df_chunk = df.iloc[:max_rows].copy()
    
    # Detectamos tanto picos como valles
    peaks_list = detect_peaks_in_close(df_chunk, prominence=prominence, distance=distance)
    valleys_list = detect_valleys_in_close(df_chunk, prominence=prominence, distance=distance)
    
    # Combinamos ambas listas y las ordenamos por índice
    all_points = peaks_list + valleys_list
    all_points.sort(key=lambda x: x[0])  # Ordenar por índice
    
    current_index = max_rows - 1
    
    # Procesamos clusters para resistencias y soportes
    clusters = cluster_all_points_dynamic(all_points, df_chunk, pips_range=pips_range, 
                                         reaction_window=reaction_window, current_index=current_index, 
                                         K_age=K_age, T_recency=T_recency, bonus_factor=bonus_factor, 
                                         role_change_bonus=role_change_bonus, gamma=gamma, delta=delta, epsilon=epsilon)
    
    # Convertimos a DataFrames
    df_points = pd.DataFrame(all_points, columns=['idx', 'price', 'prominence', 'type']).sort_values('idx').reset_index(drop=True)
    df_clusters = pd.DataFrame(clusters).reset_index(drop=True)
    
    return df_points, df_clusters

def calculate_progressive_support_resistance(
    df,
    initial_window=5000,
    step_size=100,
    lookback_window=10000,
    prominence=0.0010,
    distance=10,
    pips_range=0.001,
    reaction_window=50,
    bonus_factor=0.1,
    role_change_bonus=1.5,
    gamma=5,
    delta=5,
    epsilon=0.5,
    merge_threshold=0.0002
):
    """
    Calcula soportes y resistencias progresivamente usando una ventana deslizante,
    manteniendo todos los niveles activos en el historial.
    """
    # Configuración inicial y manejo temporal
    FAR_FUTURE_DATE = pd.Timestamp('2099-12-31 23:59:59')
    
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
        peaks = detect_peaks_in_close(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)
        valleys = detect_valleys_in_close(current_slice, prominence=prominence, distance=distance, time_reference=slice_time_reference)

        # Combinar y ordenar los puntos
        all_points = peaks + valleys
        all_points.sort(key=lambda x: x[0])
        # Ajustar el índice pero mantener el timestamp
        all_points = [(idx + start_idx, price, prom, type_, timestamp) for idx, price, prom, type_, timestamp in all_points]
        
        # Seguimiento de cambios en este ciclo
        updated_in_cycle = set()
        
        if all_points:
            # Clusterizar puntos
            local_clusters = cluster_all_points_dynamic(
                all_points, df_working, pips_range=pips_range, 
                reaction_window=reaction_window, current_index=actual_end-1,
                bonus_factor=bonus_factor, role_change_bonus=role_change_bonus,
                gamma=gamma, delta=delta, epsilon=epsilon
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

