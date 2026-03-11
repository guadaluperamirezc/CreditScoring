"""
export_artefacts.py
──────────────────────────────────────────────────────────────────────────────
Corre esto UNA SOLA VEZ al final de tu notebook de entrenamiento.
Exporta los 3 artefactos que necesita la API:

    models/
    └── s08/
        └── api/
            ├── model.pkl
            ├── woe_maps.pkl
            └── config.json
"""

import json
import pickle
from pathlib import Path

# ── Detecta el root del proyecto en base a la ubicación de este script ───────
try:
    _SCRIPT_PATH = Path(__file__).resolve()
    PROJECT_ROOT = _SCRIPT_PATH.parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().resolve().parent

# Ruta destino por defecto: <project_root>/models/s08/api
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "s08" / "api"
print(f"📂 Proyecto raíz detectado en: {PROJECT_ROOT}\n")
print(f"📂 Carpeta de salida por defecto: {DEFAULT_OUT_DIR}\n")

def export(
    model,
    fe,                   # instancia FeatureEngineering post run_all()
    features: list,       # columnas woe_* que entran al modelo
    cutoff: float,
    # parámetros PDO — deben coincidir con tu notebook de scoring
    PDO    : float = 20.0,
    Score0 : float = 600.0,
    Odds0  : float = (1 - 0.29) / 0.29,   # ≈ 2.448  (PD base = 29%)
    out_dir: Path | str = DEFAULT_OUT_DIR,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Modelo ─────────────────────────────────────────────────────────────
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"✅  model.pkl       → {out_dir / 'model.pkl'}")

    # ── 2. WoE maps ───────────────────────────────────────────────────────────
    # Convierte las claves Interval → str para serialización portátil
    woe_export = {}
    for col, art in fe._woe_maps.items():
        woe_export[col] = {
            "bin_edges": [float(x) for x in art["bin_edges"]],
            "woe_map"  : {str(k): float(v) for k, v in art["woe_map"].items()},
        }

    with open(out_dir / "woe_maps.pkl", "wb") as f:
        pickle.dump(woe_export, f)
    print(
        f"✅  woe_maps.pkl    → {out_dir / 'woe_maps.pkl'}  "
        f"({len(woe_export)} features)"
    )

    # ── 3. Config ─────────────────────────────────────────────────────────────
    config = {
        "cutoff"  : float(cutoff),
        "features": list(features),
        "PDO"     : float(PDO),
        "Score0"  : float(Score0),
        "Odds0"   : float(Odds0),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅  config.json     → {out_dir / 'config.json'}")

    print(f"\n   cutoff   = {cutoff:.4f}")
    print(f"   features = {len(features)}  →  {features[:4]} ...")
    print(f"   PDO={PDO} | Score0={Score0} | Odds0={Odds0:.4f}")
    print(f"\n🎯  Listo. Copia la carpeta '{out_dir}/' junto a main.py y levanta la API.")


if __name__ == "__main__":
    print("Importa y llama export() desde tu notebook de entrenamiento.")
    print("Ejemplo:")
    print("  from export_artefacts import export")
    print("  export(model=mdl.best_model, fe=fe, features=features, cutoff=cutoff_score)")
