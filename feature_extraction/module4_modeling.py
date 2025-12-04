from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Imports opcionales (solo si están instalados)
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - opcional
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - opcional
    LGBMRegressor = None

# Base de datos: usa outputs de Módulo 2 y 3
BASE_DIR: Path = Path("data") / "module1_ingestion" / "2024_Bahrain_Grand_Prix_R"
LAP_FEATURES_FILE = BASE_DIR / "lap_features_module2.csv"
PCA_SCORES_FILE = BASE_DIR / "pca_scores_module3.csv"
TELEMETRY_ENRICHED_FILE = BASE_DIR / "telemetry_time_10hz_enriched.csv"

TARGET = "LapTimeSeconds"
THROTTLE_ON_THRESH = 10.0  # %
BRAKE_ON_THRESH = 5.0  # %
JERK_THR = 5.0  # m/s^3 para contar eventos de jerk alto


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray


def load_dataset(
    lap_features_path: Path = LAP_FEATURES_FILE,
    pca_scores_path: Path = PCA_SCORES_FILE,
    telemetry_enriched_path: Path = TELEMETRY_ENRICHED_FILE,
    use_pcs: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga features de M2 y opcionalmente añade PC1-3 como variables."""
    lap_feat = pd.read_csv(lap_features_path)
    # Enriquecer con features adicionales derivados de la telemetría enriquecida
    if telemetry_enriched_path.exists():
        extra = compute_extra_features(telemetry_enriched_path)
        lap_feat = lap_feat.merge(extra, on=["Driver", "LapNumber"], how="left")

    if use_pcs and pca_scores_path.exists():
        pcs = pd.read_csv(pca_scores_path)[["PC1", "PC2", "PC3"]]
        lap_feat = pd.concat([lap_feat.reset_index(drop=True), pcs.reset_index(drop=True)], axis=1)

    y = lap_feat[TARGET]
    X = lap_feat.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Crea un preprocesador con escalado numérico y one-hot para categóricas."""
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_tf = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def get_models() -> Dict[str, object]:
    """Define modelos a comparar; incluye XGB si está disponible."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Lasso": Lasso(alpha=0.01, max_iter=5000, random_state=42),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
    return models


def evaluate_model_cv(
    name: str,
    model: object,
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Tuple[ModelResult, Pipeline]:
    """Entrena y evalúa con K-Fold; devuelve métricas promedio y último pipeline entrenado."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []
    preds_all: List[float] = []
    y_all: List[float] = []

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        maes.append(mean_absolute_error(y_val, y_pred))
        rmses.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
        r2s.append(r2_score(y_val, y_pred))
        preds_all.extend(y_pred.tolist())
        y_all.extend(y_val.tolist())

    # Entrenar una vez en todo el dataset para persistir el pipeline final
    pipe.fit(X, y)

    res = ModelResult(
        name=name,
        mae=float(np.mean(maes)),
        rmse=float(np.mean(rmses)),
        r2=float(np.mean(r2s)),
        y_true=np.array(y_all),
        y_pred=np.array(preds_all),
    )
    return res, pipe


def run_modeling(use_pcs: bool = True) -> Dict[str, float]:
    """Orquesta la carga, entrenamiento y evaluación (K-Fold) de múltiples modelos."""
    X, y = load_dataset(use_pcs=use_pcs)
    preprocessor, _, _ = build_preprocessor(X)

    models = get_models()
    results: List[ModelResult] = []
    trained_pipes: Dict[str, Pipeline] = {}

    for name, mdl in models.items():
        res, pipe = evaluate_model_cv(name, mdl, preprocessor, X, y, n_splits=5)
        results.append(res)
        trained_pipes[name] = pipe

    # Seleccionar el mejor por MAE
    best = min(results, key=lambda r: r.mae)
    best_pipe = trained_pipes[best.name]

    metrics = {
        r.name: {"MAE_mean": r.mae, "RMSE_mean": r.rmse, "R2_mean": r.r2} for r in results
    }
    metrics["best_model"] = {
        "name": best.name,
        "MAE_mean": best.mae,
        "RMSE_mean": best.rmse,
        "R2_mean": best.r2,
    }

    # Persistir métricas y modelo ganador
    out_dir = BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_metrics_module4.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(best_pipe, out_dir / "best_model_module4.pkl")

    # Guardar predicciones de validación para el notebook
    preds_df = pd.DataFrame(
        {"y_true": best.y_true, "y_pred": best.y_pred, "fold_order": list(range(len(best.y_true)))}
    ).reset_index(drop=True)
    preds_df.to_csv(out_dir / "val_predictions_module4.csv", index=False)

    return {
        "best_model": best.name,
        "best_mae": best.mae,
        "best_rmse": best.rmse,
        "best_r2": best.r2,
    }


def main() -> None:
    info = run_modeling(use_pcs=True)
    print("✓ Módulo 4 completado")
    print(f"  Mejor modelo: {info['best_model']}")
    print(f"  MAE (prom CV):  {info['best_mae']:.3f}")
    print(f"  RMSE (prom CV): {info['best_rmse']:.3f}")
    print(f"  R2 (prom CV):   {info['best_r2']:.3f}")


if __name__ == "__main__":
    main()
