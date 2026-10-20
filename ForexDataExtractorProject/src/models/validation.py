"""
validation.py - TICK-09: Purged & Embargoed Time-Series Cross-Validation.

Implementación de validación cruzada para finanzas cuantitativas (Marcos López de Prado - AFML):
1. PurgedKFold:
   - Particiona la línea de tiempo en K bloques secuenciales.
   - En cada fold k, el bloque k actúa como conjunto de prueba Out-of-Sample.
   - Purga (Purging): Elimina del conjunto de entrenamiento cualquier operación cuyo horizonte de vida
     se solape con el período de prueba (evita fuga de información hacia atrás y hacia adelante).
   - Embargo (Embargoing): Elimina un período posterior al conjunto de prueba para prevenir fuga
     por correlación serial / memoria autorregresiva de la volatilidad.
2. PurgedWalkForwardCV:
   - Ventana temporal expansiva (Walk-Forward).
   - Entrena con bloques pasados [0..k] y evalúa en el bloque futuro inmediato [k+1],
     purgando las operaciones del final del train que cruzan la frontera del test.
3. Compatible con scikit-learn (BaseCrossValidator).
"""

import numpy as np
import pandas as pd
from typing import Generator, List, Optional, Tuple, Union
from sklearn.model_selection import BaseCrossValidator


def compute_t1_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    exit_bar_col: str = "trade_exit_bar",
    ticker_col: Optional[str] = "ticker",
    default_horizon: int = 26,
) -> pd.Series:
    """
    Calcula la serie de fechas/horas exactas de salida (t1) para cada observación.
    Respeta la cronología individual de cada activo (saltos de fin de semana, feriados).

    Retorna:
    --------
    pd.Series con t1 (timestamp de cierre del evento) alineada al índice del DataFrame.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Columna '{timestamp_col}' no encontrada en el DataFrame.")

    timestamps = pd.to_datetime(df[timestamp_col])
    t1_series = pd.Series(index=df.index, dtype=timestamps.dtype)

    if ticker_col and ticker_col in df.columns:
        for ticker, group_idx in df.groupby(ticker_col).groups.items():
            g_ts = timestamps.loc[group_idx]
            n_g = len(g_ts)
            if exit_bar_col in df.columns:
                bars = df.loc[group_idx, exit_bar_col].fillna(default_horizon).to_numpy(dtype=int)
            else:
                bars = np.full(n_g, default_horizon, dtype=int)

            target_idx = np.clip(np.arange(n_g) + bars, 0, n_g - 1)
            t1_series.loc[group_idx] = g_ts.iloc[target_idx].to_numpy()
    else:
        n_g = len(timestamps)
        if exit_bar_col in df.columns:
            bars = df[exit_bar_col].fillna(default_horizon).to_numpy(dtype=int)
        else:
            bars = np.full(n_g, default_horizon, dtype=int)

        target_idx = np.clip(np.arange(n_g) + bars, 0, n_g - 1)
        t1_series = pd.Series(timestamps.iloc[target_idx].to_numpy(), index=df.index, dtype=timestamps.dtype)

    return t1_series


class PurgedKFold(BaseCrossValidator):
    """
    K-Fold Cross-Validation con Purga y Embargo (López de Prado).

    Parámetros:
    -----------
    n_splits : int
        Número de particiones temporales continuas (default 5).
    t1 : Optional[pd.Series]
        Serie con las fechas de fin de evento (exit timestamps) de cada trade.
    timestamps : Optional[pd.Series]
        Serie con las fechas de inicio (entry timestamps) de cada trade.
    pct_embargo : float
        Porcentaje del rango temporal global a descartar tras el bloque de test (default 0.01 = 1%).
    bars_embargo : Optional[int]
        Número explícito de barras de embargo tras el fin del bloque de test (ej. 26 barras = 4 días).
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
        bars_embargo: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits debe ser al menos 2, recibido: {n_splits}")
        self.n_splits = n_splits
        self.t1 = t1
        self.timestamps = timestamps
        self.pct_embargo = pct_embargo
        self.bars_embargo = bars_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _resolve_time_series(self, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Extrae o calcula t0 (entry) y t1 (exit) desde X si no fueron provistos explícitamente."""
        if self.timestamps is not None:
            t0 = self.timestamps
        elif isinstance(X, pd.DataFrame) and "timestamp" in X.columns:
            t0 = pd.to_datetime(X["timestamp"])
        else:
            raise ValueError("Se requiere columna 'timestamp' en X o serie 'timestamps' provista.")

        if self.t1 is not None:
            t1 = self.t1
        elif isinstance(X, pd.DataFrame) and "trade_exit_bar" in X.columns:
            t1 = compute_t1_series(X)
        else:
            # Si no hay t1, asumir t1 = t0 (sin solapamiento dinámico, pero con embargo temporal)
            t1 = t0.copy()

        return t0, t1

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Genera particiones (train_indices, test_indices) con CERO solapamiento temporal.
        """
        t0, t1 = self._resolve_time_series(X)
        n_samples = len(X)
        indices = np.arange(n_samples)

        # 1. Particionar sobre la línea de tiempo global única (fechas ordenadas sin duplicados)
        unique_times = np.sort(t0.unique())
        n_times = len(unique_times)

        if n_times < self.n_splits:
            raise ValueError(
                f"Número de fechas únicas ({n_times}) menor que n_splits ({self.n_splits})."
            )

        # Dividir las fechas únicas en n_splits bloques secuenciales
        split_points = np.linspace(0, n_times, self.n_splits + 1, dtype=int)

        # Calcular timedelta de embargo
        if self.bars_embargo is not None and self.bars_embargo > 0:
            # Estimar duración media por barra
            diffs = pd.Series(unique_times).diff().dropna()
            median_bar_dt = diffs.median() if len(diffs) > 0 else pd.Timedelta(hours=1)
            embargo_dt = median_bar_dt * self.bars_embargo
        elif self.pct_embargo > 0:
            total_dt = unique_times[-1] - unique_times[0]
            embargo_dt = total_dt * self.pct_embargo
        else:
            embargo_dt = pd.Timedelta(0)

        for fold in range(self.n_splits):
            # Límites temporales del conjunto de test
            test_start_idx = split_points[fold]
            test_end_idx = split_points[fold + 1] - 1

            test_times_fold = unique_times[test_start_idx : test_end_idx + 1]
            test_t0_bound = test_times_fold[0]
            test_entry_max = test_times_fold[-1]

            # Test set: Todas las observaciones cuya entrada ocurrió en el rango del fold
            test_mask = t0.isin(test_times_fold).values
            test_indices = indices[test_mask]

            if len(test_indices) == 0:
                continue

            # Límite superior del test: última salida de los trades del test
            test_t1_bound = t1.iloc[test_indices].max()

            # 2. Purga (Purging) en el conjunto de entrenamiento:
            # Candidatos a entrenamiento: todas las observaciones fuera de test
            train_mask = ~test_mask

            # Condición de solapamiento: un trade de entrenamiento se solapa si
            # empezó antes de que terminara el test Y terminó después de que empezó el test:
            # (t0 <= test_t1_bound) & (t1 >= test_t0_bound)
            overlap_mask = ((t0 <= test_t1_bound) & (t1 >= test_t0_bound)).to_numpy()
            # Eliminar observaciones con solapamiento del train
            train_mask = train_mask & (~overlap_mask)

            # 3. Embargo (Embargoing):
            # Eliminar trades de entrenamiento que empiezan inmediatamente después del test
            # dentro de la ventana de embargo
            if embargo_dt > pd.Timedelta(0):
                embargo_limit = test_t1_bound + embargo_dt
                in_embargo = ((t0 > test_t1_bound) & (t0 <= embargo_limit)).to_numpy()
                train_mask = train_mask & (~in_embargo)

            train_indices = indices[train_mask]

            yield train_indices, test_indices


class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation (Ventana Expansiva) con Purga y Embargo.

    Ideal para simulación estricta de backtest Out-of-Sample:
    - Fold 0: Train en Bloque 0 -> Test en Bloque 1
    - Fold 1: Train en Bloques 0..1 -> Test en Bloque 2
    - Fold k: Train en Bloques 0..k -> Test en Bloque k+1
    - CERO lookahead: los datos de entrenamiento siempre preceden cronológicamente al test.
    - Se purgan del final del entrenamiento los trades que aún estaban abiertos al iniciar el test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits debe ser al menos 2, recibido: {n_splits}")
        self.n_splits = n_splits
        self.t1 = t1
        self.timestamps = timestamps

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits - 1

    def _resolve_time_series(self, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        if self.timestamps is not None:
            t0 = self.timestamps
        elif isinstance(X, pd.DataFrame) and "timestamp" in X.columns:
            t0 = pd.to_datetime(X["timestamp"])
        else:
            raise ValueError("Se requiere columna 'timestamp' en X o serie 'timestamps' provista.")

        if self.t1 is not None:
            t1 = self.t1
        elif isinstance(X, pd.DataFrame) and "trade_exit_bar" in X.columns:
            t1 = compute_t1_series(X)
        else:
            t1 = t0.copy()

        return t0, t1

    def split(
        self, X: pd.DataFrame, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        t0, t1 = self._resolve_time_series(X)
        n_samples = len(X)
        indices = np.arange(n_samples)

        unique_times = np.sort(t0.unique())
        n_times = len(unique_times)
        split_points = np.linspace(0, n_times, self.n_splits + 1, dtype=int)

        for step in range(self.n_splits - 1):
            # Train abarca bloques 0 a step
            train_times = unique_times[0 : split_points[step + 1]]
            # Test abarca bloque step + 1
            test_times = unique_times[split_points[step + 1] : split_points[step + 2]]

            test_t0_bound = test_times[0]
            test_mask = t0.isin(test_times).values
            test_indices = indices[test_mask]

            if len(test_indices) == 0:
                continue

            # Candidatos a train (todos los que entraron antes del inicio del test)
            train_mask = t0.isin(train_times).values

            # Purga en Walk-Forward:
            # Eliminar cualquier trade de train cuyo evento concluyó después de que empezó el test (t1 >= test_t0_bound)
            leaking_trades = (t1 >= test_t0_bound).to_numpy()
            train_mask = train_mask & (~leaking_trades)

            train_indices = indices[train_mask]

            yield train_indices, test_indices


def verify_no_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    t0: pd.Series,
    t1: pd.Series,
) -> dict:
    """
    Función de auditoría cuantitativa: Verifica formalmente que no exista ningún solapamiento
    temporal ni fuga de datos entre el conjunto de entrenamiento y el de prueba.
    """
    test_t0_min = t0.iloc[test_indices].min()
    test_t1_max = t1.iloc[test_indices].max()

    train_t0 = t0.iloc[train_indices]
    train_t1 = t1.iloc[train_indices]

    # Comprobar solapamiento: t0_train <= test_t1_max & t1_train >= test_t0_min
    overlapping = ((train_t0 <= test_t1_max) & (train_t1 >= test_t0_min)).to_numpy()
    overlap_count = int(np.sum(overlapping))

    # Comprobar intersección de índices crudos
    index_intersection = len(np.intersect1d(train_indices, test_indices))

    return {
        "test_t0_min": test_t0_min,
        "test_t1_max": test_t1_max,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "index_intersection": index_intersection,
        "temporal_overlap_count": overlap_count,
        "is_leakage_free": (overlap_count == 0) and (index_intersection == 0),
    }
