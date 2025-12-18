from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Base de datos: reutiliza outputs del Módulo 2 (Bahrain 2024 por defecto)
DEFAULT_BASE_DIR: Path = Path("data") / "module1_ingestion" / "2024_Bahrain_Grand_Prix_R"
DEFAULT_FEATURES: List[str] = [
    "LapTimeSeconds",
    "Energy_Index",
    "MeanAbs_Jerk_Long",
    "MeanAbs_Jerk_Lat",
    "Avg_Speed_mps",
    "Brake_Aggression",
    "Max_Lateral_g",
    "Max_Longitudinal_g",
    "TyreLife",
]
PCA_COMPONENTS: int = 3


@dataclass
class PCAArtifacts:
    scores_path: Path
    model_path: Path


def load_lap_features(base_dir: Path) -> pd.DataFrame:
    """Carga las features agregadas por vuelta (output de Módulo 2)."""
    return pd.read_csv(base_dir / "lap_features_module2.csv")


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye la matriz X (numérica) y conserva metadatos para interpretar PCs.
    One-hot para Compound; otras columnas numéricas definidas en DEFAULT_FEATURES.
    """
    missing = [c for c in DEFAULT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en lap_features_module2.csv: {missing}")

    meta_cols = ["Driver", "LapNumber", "Compound"]
    meta = df[meta_cols].copy()

    X_numeric = df[DEFAULT_FEATURES].copy()
    compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    X = pd.concat([X_numeric, compound_dummies], axis=1)
    return X, meta


def run_pca(X: pd.DataFrame, n_components: int = PCA_COMPONENTS) -> Tuple[pd.DataFrame, dict]:
    """Estandariza, aplica PCA y devuelve scores y metadatos del modelo."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    model_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "components": pca.components_.tolist(),  # loadings
        "feature_names": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return scores_df, model_info


def persist_pca_outputs(
    base_dir: Path,
    scores_df: pd.DataFrame,
    meta: pd.DataFrame,
    model_info: dict,
) -> PCAArtifacts:
    """Guarda scores con metadatos y el modelo PCA/Scaler en JSON."""
    scores_with_meta = pd.concat([meta.reset_index(drop=True), scores_df], axis=1)
    scores_path = base_dir / "pca_scores_module3.csv"
    model_path = base_dir / "pca_model_module3.json"

    scores_with_meta.to_csv(scores_path, index=False)
    model_path.write_text(json.dumps(model_info, indent=2))

    return PCAArtifacts(scores_path=scores_path, model_path=model_path)


def run_module3(
    base_dir: Path = DEFAULT_BASE_DIR,
    n_components: int = PCA_COMPONENTS,
) -> PCAArtifacts:
    """
    Orquestador del Módulo 3:
    - Lee lap_features_module2.csv
    - Construye matriz de features (incluye dummies de Compound)
    - Estandariza y aplica PCA
    - Persiste scores + metadatos
    """
    lap_feat = load_lap_features(base_dir)
    X, meta = build_feature_matrix(lap_feat)
    scores_df, model_info = run_pca(X, n_components=n_components)
    return persist_pca_outputs(base_dir, scores_df, meta, model_info)


def main() -> None:
    """
    Ejecuta PCA sobre Bahrain 2024 (artefactos de M2) y guarda scores/modelo.
    """
    artifacts = run_module3()
    print("✓ Módulo 3 completado")
    print(f"  Scores PCA: {artifacts.scores_path}")
    print(f"  Modelo PCA: {artifacts.model_path}")


if __name__ == "__main__":
    main()
