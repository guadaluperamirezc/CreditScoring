"""
inference_feature_engineering.py
══════════════════════════════════════════════════════════════════════════════
Este script reutiliza la clase FeatureEngineering EXISTENTE para
aplicar los WoE maps ya entrenados sobre datos nuevos en memoria.

No se toca nada del archivo feature_engineering.py — solo importas y llamas transform().
══════════════════════════════════════════════════════════════════════════════
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))  # → .../src/s08
ROOT_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # → .../CreditScoring
src_path_07 = os.path.join(ROOT_DIR, "src", "s07")
sys.path.insert(0, src_path_07)

# ── Importa tu clase tal como está ───────────────────────────────────────────
from feature_engineering import FeatureEngineering


def transform(df_preprocessed: pd.DataFrame, woe_maps_path: str) -> pd.DataFrame:
    """
    1. Calcula los derived features (ratios) igual que en entrenamiento.
    2. Aplica los WoE maps ya entrenados (cargados desde woe_maps.pkl).

    Parámetros
    ----------
    df_preprocessed : DataFrame ya preprocesado (salida de inference_preprocessing)
    woe_maps_path   : ruta al archivo woe_maps.pkl generado en entrenamiento

    Retorna
    -------
    DataFrame con columnas woe_* listas para el modelo.
    """
    # ── Carga WoE maps ────────────────────────────────────────────────────────
    with open(woe_maps_path, "rb") as f:
        woe_maps = pickle.load(f)

    # ── Instancia FeatureEngineering solo para usar sus métodos ──────────────
    fe = FeatureEngineering(input_dir=".", output_dir=".")
    fe.data  = df_preprocessed.copy()
    fe.train = fe.data   # en inferencia no hay split; todo es "train"
    fe.test  = pd.DataFrame()

    # ── Paso 1: derived features (misma lógica que en entrenamiento) ──────────
    fe._derived_features()

    df = fe.data.copy()

    # ── Paso 2: aplica WoE maps entrenados ───────────────────────────────────
    for col, artefact in woe_maps.items():
        if col not in df.columns:
            df[f"woe_{col}"] = 0.0   # fallback neutro si columna no existe
            continue

        bin_edges = artefact["bin_edges"]
        woe_map   = artefact["woe_map"]   # {str(Interval) → float}

        binned = pd.cut(df[col], bins=bin_edges, include_lowest=True)

        # Las claves del woe_map son strings → convierte Interval a string
        df[f"woe_{col}"] = (
            binned.astype(str).map(woe_map).fillna(0).astype(float)
        )

    return df