from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# SHAP opcional; si no está instalado, lo omitimos.
try:
    import shap
except Exception:
    shap = None

# BentoML opcional (para evitar errores si no está instalado en entornos dev simples)
try:
    import bentoml
except Exception:
    bentoml = None

# Imports opcionales (solo si están instalados)
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - opcional
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - opcional
    LGBMRegressor = None

# Use the script directory so paths work whether the script is run from the
# project root or from inside the `feature_extraction/` folder.
_script_dir = Path(__file__).resolve().parent
# Base de datos: usa outputs de Módulo 2 y 3
BASE_DIR: Path = _script_dir / "data" / "module1_ingestion"
# Dataset global consolidado
GLOBAL_LAP_FEATURES = BASE_DIR / "all_lap_features.csv"
# PCs globales normalizados (opcional)
GLOBAL_PCS_NORM = BASE_DIR / "pca_scores_global_norm.csv"

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
    meta: pd.DataFrame | None = None
    model_params: Dict[str, Any] | None = None


def load_dataset(
    lap_features_path: Path = GLOBAL_LAP_FEATURES,
    pca_scores_path: Path = GLOBAL_PCS_NORM,
    use_pcs: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset global consolidado y añade PC1-3 normalizados si existen."""
    # Intentamos cargar local o desde rutas relativas comunes
    if not lap_features_path.exists():
        # Fallback a buscar en feature_extraction/ variable
        candidate = Path("feature_extraction") / lap_features_path
        if candidate.exists():
            lap_features_path = candidate
        else:
            # Fallback 2: buscar en data/ directamente
            candidate2 = Path("data") / "module1_ingestion" / "all_lap_features.csv"
            if candidate2.exists():
                lap_features_path = candidate2

    lap_feat = pd.read_csv(lap_features_path)

    if use_pcs:
        # Lógica de fallback para PCA
        if not pca_scores_path.exists():
             candidate = Path("feature_extraction") / pca_scores_path
             if candidate.exists():
                 pca_scores_path = candidate
             else:
                 candidate2 = Path("data") / "module1_ingestion" / "pca_scores_global_norm.csv"
                 if candidate2.exists():
                     pca_scores_path = candidate2

        if pca_scores_path.exists():
            pcs = pd.read_csv(pca_scores_path)[["PC1", "PC2", "PC3"]]
            # Asegurar alineación
            if len(pcs) == len(lap_feat):
                 lap_feat = pd.concat([lap_feat.reset_index(drop=True), pcs.reset_index(drop=True)], axis=1)

    # Eliminar filas con NaN en cualquier columna de entrada/target
    lap_feat = lap_feat.dropna().reset_index(drop=True)
    if TARGET not in lap_feat.columns:
        raise ValueError(f"Target '{TARGET}' no encontrado en dataset columns: {lap_feat.columns}")

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


def get_models(custom_params: Dict[str, Any] = None) -> Dict[str, object]:
    """Define modelos a comparar; incluye XGB si está disponible."""
    if custom_params is None:
        custom_params = {}
    
    # Random Forest Default
    rf_params = {
        "n_estimators": custom_params.get("n_estimators", 300),
        "max_depth": custom_params.get("max_depth", None),
        "random_state": 42,
        "n_jobs": -1
    }
    
    models = {
        "RandomForest": RandomForestRegressor(**rf_params),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "Lasso": Lasso(alpha=0.01, max_iter=5000, random_state=42),
    }

    # Si se pide XGBoost explícitamente o está disponible
    if XGBRegressor is not None:
        xgb_params = {
            "n_estimators": custom_params.get("n_estimators", 400),
            "learning_rate": custom_params.get("learning_rate", 0.05),
            "max_depth": custom_params.get("max_depth", 4),
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        # Solo sobreescribir si estamos 'tuneando' específicamente este modelo, 
        # o usar defaults razonables si es un run genérico.
        models["XGBoost"] = XGBRegressor(**xgb_params)

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
    meta_parts: List[pd.DataFrame] = []

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
        meta_parts.append(X_val.reset_index(drop=True))

    # Entrenar una vez en todo el dataset para persistir el pipeline final
    pipe.fit(X, y)

    res = ModelResult(
        name=name,
        mae=float(np.mean(maes)),
        rmse=float(np.mean(rmses)),
        r2=float(np.mean(r2s)),
        y_true=np.array(y_all),
        y_pred=np.array(preds_all),
        meta=pd.concat(meta_parts, ignore_index=True) if meta_parts else None,
        model_params=model.get_params() if hasattr(model, "get_params") else {}
    )
    return res, pipe


def run_modeling(
    use_pcs: bool = True, 
    save_bento: bool = True,
    bento_model_name: str = "f1_laptime_predictor"
) -> Dict[str, Any]:
    """Orquesta la carga global, entrenamiento y evaluación (K-Fold) de múltiples modelos."""
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

    # Persistir métricas y modelo ganador localmente
    out_dir = BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_metrics_module4.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(best_pipe, out_dir / "best_model_module4.pkl")

    # Guardar en BentoML
    if save_bento and bentoml is not None:
        try:
            # BentoML Sklearn soporta pipelines
            bento_model = bentoml.sklearn.save_model(
                bento_model_name, 
                best_pipe,
                signatures={"predict": {"batchable": False}}
            )
            print(f"Modelo guardado en BentoML: {bento_model.tag}")
            metrics["bento_tag"] = str(bento_model.tag)
        except Exception as e:
            print(f"Error guardando en BentoML: {e}")

    # Guardar predicciones de validación
    preds_df = pd.DataFrame(
        {"y_true": best.y_true, "y_pred": best.y_pred, "fold_order": list(range(len(best.y_true)))}
    ).reset_index(drop=True)
    if best.meta is not None:
        preds_df = pd.concat([best.meta.reset_index(drop=True), preds_df], axis=1)
    preds_df.to_csv(out_dir / "val_predictions_module4.csv", index=False)

    return metrics


def train_single_model(
    model_type: str,
    params: Dict[str, Any],
    use_pcs: bool = True
) -> ModelResult:
    """Función helper para Streamlit: entrena UN solo modelo con parámetros custom."""
    X, y = load_dataset(use_pcs=use_pcs)
    preprocessor, _, _ = build_preprocessor(X)
    
    # Instanciar modelo según tipo
    if model_type == "RandomForest":
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == "XGBoost" and XGBRegressor is not None:
        model = XGBRegressor(**params, objective="reg:squarederror", random_state=42, n_jobs=-1)
    elif model_type == "GradientBoosting":
        model = GradientBoostingRegressor(**params, random_state=42)
    else:
        # Fallback
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Evaluar con CV rápido (3 splits para feedback rápido en UI)
    res, pipe = evaluate_model_cv(model_type, model, preprocessor, X, y, n_splits=3)
    return res


def main() -> None:
    info = run_modeling(use_pcs=True)
    print("✓ Módulo 4 completado")
    print(f"  Mejor modelo: {info['best_model']}")
    print(f"  MAE (prom CV):  {info['best_model']['MAE_mean']:.3f}")
    
    if "bento_tag" in info:
        print(f"  BentoML Tag: {info['bento_tag']}")


if __name__ == "__main__":
    main()
