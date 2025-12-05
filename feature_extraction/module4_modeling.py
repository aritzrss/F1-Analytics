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

# SHAP opcional; si no está instalado, lo omitimos.
try:
    import shap
except Exception:
    shap = None

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


def load_dataset(
    lap_features_path: Path = GLOBAL_LAP_FEATURES,
    pca_scores_path: Path = GLOBAL_PCS_NORM,
    use_pcs: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset global consolidado y añade PC1-3 normalizados si existen."""
    lap_feat = pd.read_csv(lap_features_path)

    if use_pcs and pca_scores_path.exists():
        pcs = pd.read_csv(pca_scores_path)[["PC1", "PC2", "PC3"]]
        lap_feat = pd.concat([lap_feat.reset_index(drop=True), pcs.reset_index(drop=True)], axis=1)

    # Eliminar filas con NaN en cualquier columna de entrada/target
    lap_feat = lap_feat.dropna().reset_index(drop=True)
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
    )
    return res, pipe


def run_modeling(use_pcs: bool = True) -> Dict[str, float]:
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

    # Persistir métricas y modelo ganador
    out_dir = BASE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_metrics_module4.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(best_pipe, out_dir / "best_model_module4.pkl")

    # Guardar predicciones de validación para el notebook
    preds_df = pd.DataFrame(
        {"y_true": best.y_true, "y_pred": best.y_pred, "fold_order": list(range(len(best.y_true)))}
    ).reset_index(drop=True)
    if best.meta is not None:
        preds_df = pd.concat([best.meta.reset_index(drop=True), preds_df], axis=1)
    preds_df.to_csv(out_dir / "val_predictions_module4.csv", index=False)

    # SHAP: calcular valores y guardar resumen si la librería está disponible
    if shap is not None and hasattr(best_pipe, "predict"):
        try:
            # Tomamos una muestra para rapidez (sobre las features originales)
            X_sample_raw = X.sample(min(500, len(X)), random_state=42)
            # Transformamos con el preprocesador del pipeline (puede devolver sparse)
            preproc = best_pipe.named_steps["preprocess"]
            X_sample_tr = preproc.transform(X_sample_raw)
            # Asegurar denso para construir DataFrame
            if not hasattr(X_sample_tr, "toarray"):
                X_sample_dense = X_sample_tr
            else:
                X_sample_dense = X_sample_tr.toarray()
            feature_names = preproc.get_feature_names_out()

            model_step = best_pipe.named_steps["model"]
            # Explainer acorde al modelo; TreeExplainer cubre RF/GB/XGB/LGBM, para Lasso cae a Explainer genérico
            tree_models = {
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "XGBRegressor",
                "LGBMRegressor",
            }
            if hasattr(shap, "TreeExplainer") and model_step.__class__.__name__ in tree_models:
                explainer = shap.TreeExplainer(model_step)
                shap_values = explainer(X_sample_dense)
            else:
                explainer = shap.Explainer(model_step, X_sample_dense, feature_names=feature_names)
                shap_values = explainer(X_sample_dense)

            # Convertir a DataFrame usando nombres del espacio transformado
            shap_summary = pd.DataFrame(shap_values.values, columns=feature_names)

            shap_dir = out_dir / "shap"
            shap_dir.mkdir(exist_ok=True, parents=True)
            # Guardar valores SHAP en formato resumen (parquet)
            shap_summary.to_parquet(shap_dir / "shap_values.parquet", index=False)
            # Guardar la muestra de entrada original para contextualizar
            X_sample_raw.to_csv(shap_dir / "shap_input_sample_raw.csv", index=False)
            # Guardar nombres de features (post-encoder) para trazabilidad
            pd.Series(feature_names).to_csv(shap_dir / "shap_feature_names.csv", index=False, header=False)
            # Importancias medias absolutas
            mean_abs = shap_summary.abs().mean().sort_values(ascending=False)
            mean_abs.to_csv(shap_dir / "shap_mean_abs.csv")
        except Exception as exc:
            print(f"SHAP no se pudo calcular: {exc}")

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
