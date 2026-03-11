"""
main.py  —  Credit Scoring API
══════════════════════════════════════════════════════════════════════════════
Flujo por request:

  JSON payload
      │
      ▼
  inference_preprocessing.py   ← reutiliza tu Preprocessing class
      │
      ▼
  inference_feature_engineering.py  ← reutiliza tu FeatureEngineering class
      │                                + aplica woe_maps.pkl entrenado
      ▼
  model.pkl  →  predict_proba()
      │
      ▼
  Escala PDO  →  Score en puntos
      │
      ▼
  JSON response  { score, prob_default, decision, risk_segment }

Endpoints:
  POST /score        →  un cliente   (dict)
  POST /score/batch  →  N clientes   (list[dict])
  GET  /health       →  status check
══════════════════════════════════════════════════════════════════════════════
"""

import json
import pickle
import numpy as np
import pandas as pd
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# ── Tus adapters ──────────────────────────────────────────────────────────────
import inference_preprocessing       as prep_inf
import inference_feature_engineering as fe_inf

# ── Rutas a artefactos ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está main_api.py
MODEL_PATH = os.path.join(BASE_DIR, "../../models/s08/api/model.pkl")
WOE_PATH   = os.path.join(BASE_DIR, "../../models/s08/api/woe_maps.pkl")
CONFIG_PATH= os.path.join(BASE_DIR, "../../models/s08/api/config.json")

# ── Estado global (cargado una sola vez al arrancar) ─────────────────────────
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelo + config al arrancar; libera al apagar."""
    with open(MODEL_PATH, "rb") as f:
        STATE["model"] = pickle.load(f)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    STATE["features"] = cfg["features"]      # lista de columnas woe_*
    STATE["cutoff"]   = cfg["cutoff"]
    STATE["factor"]   = cfg["PDO"] / np.log(2)
    STATE["offset"]   = cfg["Score0"] - STATE["factor"] * np.log(cfg["Odds0"])

    print(f"✅  Modelo cargado | cutoff={STATE['cutoff']:.2f} | "
          f"features={len(STATE['features'])}")
    yield
    STATE.clear()


app = FastAPI(
    title       = "Credit Scoring API",
    description = "Credit Scoring Programa de Especialización 🚀",
    summary     = "scorecard logístico —> preprocessing → WoE → modelo → decisión",
    version     = "1.0.0",
    contact     = {'name': 'Enzo Infantes Zúñiga',
                   'email': 'enzo.infantes28@gmail.com',
                   'url': 'https://www.linkedin.com/in/enzo-infantes/'},
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL DE SCORING
# ─────────────────────────────────────────────────────────────────────────────
def _score_dataframe(raw_records: list[dict]) -> list[dict]:
    """
    Recibe lista de dicts crudos (del JSON), corre el pipeline completo
    y devuelve lista de resultados.
    """
    # 1. Preprocessing
    df_raw  = pd.DataFrame(raw_records)
    df_prep = prep_inf.transform(df_raw)

    # 2. Feature Engineering + WoE
    df_fe   = fe_inf.transform(df_prep, WOE_PATH)

    # 3. Selecciona solo las features que usó el modelo
    features = STATE["features"]
    X = df_fe.reindex(columns=features, fill_value=0.0)

    # 4. Predicción
    pds   = STATE["model"].predict_proba(X)[:, 1]        # P(Default=1)
    odds  = (1 - pds) / (pds + 1e-9)                    # Good:Bad
    score = STATE["offset"] + STATE["factor"] * np.log(odds)

    cutoff = STATE["cutoff"]

    # 5. Ensambla resultados
    results = []
    for i, (pd_, sc) in enumerate(zip(pds, score)):
        decision = "APPROVE" if sc >= cutoff else "REJECT"
        segment  = (
            "LOW"    if sc >= cutoff + 40 else
            "MEDIUM" if sc >= cutoff      else
            "HIGH"
        )
        results.append({
            "score"        : round(float(sc),  2),
            "prob_default" : round(float(pd_), 4),
            "log_odds"     : round(float(np.log(odds[i])), 4),
            "decision"     : decision,
            "risk_segment" : segment,
            "cutoff_used"  : cutoff,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["utils"])
def health():
    return {
        "status"      : "ok",
        "model_loaded": "model" in STATE,
        "n_features"  : len(STATE.get("features", [])),
        "cutoff"      : STATE.get("cutoff"),
    }


@app.post("/score", tags=["scoring"])
def score_one(applicant: dict):
    """Scorea un solo cliente. Recibe un JSON object {}."""
    try:
        return _score_dataframe([applicant])[0]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/score/batch", tags=["scoring"])
def score_batch(applicants: list[dict]):
    """
    Scorea una lista de clientes. Recibe un JSON array [{},...].
    Devuelve resultados individuales + resumen del portafolio.
    """
    if not applicants:
        raise HTTPException(400, "Lista vacía.")
    try:
        results  = _score_dataframe(applicants)
        approved = [r for r in results if r["decision"] == "APPROVE"]
        return {
            "results": results,
            "summary": {
                "total"        : len(results),
                "approved"     : len(approved),
                "rejected"     : len(results) - len(approved),
                "approval_rate": round(len(approved) / len(results), 4),
                "avg_score"    : round(np.mean([r["score"] for r in results]), 2),
                "avg_pd"       : round(np.mean([r["prob_default"] for r in results]), 4),
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))