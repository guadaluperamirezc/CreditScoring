"""
inference_preprocessing.py
══════════════════════════════════════════════════════════════════════════════
Este script reutiliza la clase Preprocessing EXISTENTE para correr
sobre un DataFrame en memoria (sin leer/escribir CSV).

No toca nada de preprocessing.py — solo importas y llamas transform().
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import sys
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # → .../src/s08
ROOT_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # → .../CreditScoring
src_path_07 = os.path.join(ROOT_DIR, "src", "s07")
sys.path.insert(0, src_path_07)

# ── Importa tus clases tal como están ────────────────────────────────────────
from preprocessing import (
    Preprocessing,
    EDUCATION_MAP,
    MARITAL_MAP,
    LOG_COLS,
    LEAKAGE_COLS,
)


def transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de preprocessing.py sobre un DataFrame
    en memoria (no lee CSV, no guarda nada a disco).

    Parámetros
    ----------
    df_raw : DataFrame con las columnas del JSON de entrada (nombres originales)

    Retorna
    -------
    DataFrame preprocesado, listo para feature engineering.
    """
    df = df_raw.copy()

    # Agrega columnas dummy que el pipeline original espera pero no vienen
    # en el payload de inferencia
    if "Application_Date" not in df.columns:
        df["Application_Date"] = pd.Timestamp("today").strftime("%Y-%m-%d")
    if "Customer_ID" not in df.columns:
        df["Customer_ID"] = range(len(df))
    if "Default" not in df.columns:
        df["Default"] = np.nan   # target desconocido en inferencia

    # ── Instancia el pipeline (sin rutas reales — no va a leer CSV) ──────────
    pp = Preprocessing(
        raw_data_dir    = ".",
        output_data_dir = ".",
        drop_leakage    = True,
    )
    pp.data = df  # inyecta el DataFrame directamente

    ## en caso sea cliente nuevo
    for col in ["Oldest_Trade_Open_Months", "Newest_Trade_Open_Months"]:
        if col not in pp.data.columns or pp.data[col].isnull().all():
            pp.data[col] = 0
    
    for col in ["Default", "default"]:
        if col in pp.data.columns:
            pp.data.drop(columns=[col], inplace=True)

    # ── Corre cada paso excepto load_data() y guardado ───────────────────────
    pp._drop_leakage()
    pp._drop_non_features()
    pp._fix_dtypes()
    pp._drop_duplicates()
    pp._handle_missing()
    pp._encode_categoricals()
    pp._log_transform()
    pp._standardise_columns()

    # Elimina columna target (no existe en inferencia)
    for col in ["default", "Default"]:
        if col in pp.data.columns:
            pp.data.drop(columns=[col], inplace=True)

    return pp.data