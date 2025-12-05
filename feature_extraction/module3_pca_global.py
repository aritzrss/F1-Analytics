from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GLOBAL_FEATURES: List[str] = [
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
N_COMPONENTS = 3

# Use script directory to ensure paths work whether script is run from
# project root or from inside the feature_extraction/ folder.
_script_dir = Path(__file__).resolve().parent
GLOBAL_INPUT: Path = _script_dir / "data" / "module1_ingestion" / "all_lap_features.csv"
GLOBAL_SCORES_OUT: Path = _script_dir / "data" / "module1_ingestion" / "pca_scores_global.csv"
GLOBAL_MODEL_OUT: Path = _script_dir / "data" / "module1_ingestion" / "pca_model_global.json"


@dataclass
class PCAArtifacts:
    scores_path: Path
    model_path: Path


def load_global_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in GLOBAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para PCA global: {missing}")
    return df


def build_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = ["Driver", "LapNumber", "Compound", "Year", "Event", "SessionType"]
    meta = df[meta_cols].copy()
    feats_raw = df[GLOBAL_FEATURES].copy()
    X = feats_raw.copy()
    # One-hot de Compound y Year para capturar efectos categóricos
    compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    year_dummies = pd.get_dummies(df["Year"].astype(str), prefix="Year")
    X = pd.concat([X, compound_dummies, year_dummies], axis=1)
    # Drop filas con NaN en cualquier feature para PCA
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)
    feats_raw = feats_raw.loc[mask].reset_index(drop=True)
    return X, meta, feats_raw


def run_pca_global(
    X: pd.DataFrame, n_components: int = N_COMPONENTS
) -> Tuple[pd.DataFrame, dict]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    model_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "feature_names": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return scores_df, model_info


def persist_pca(
    scores_df: pd.DataFrame, meta: pd.DataFrame, feats_raw: pd.DataFrame, model_info: dict
) -> PCAArtifacts:
    scores_with_meta = pd.concat([meta.reset_index(drop=True), feats_raw.reset_index(drop=True), scores_df], axis=1)
    GLOBAL_SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores_with_meta.to_csv(GLOBAL_SCORES_OUT, index=False)
    GLOBAL_MODEL_OUT.write_text(json.dumps(model_info, indent=2))
    return PCAArtifacts(scores_path=GLOBAL_SCORES_OUT, model_path=GLOBAL_MODEL_OUT)


def main() -> None:
    df = load_global_dataset(GLOBAL_INPUT)
    X, meta, feats_raw = build_matrix(df)
    scores_df, model_info = run_pca_global(X)
    artifacts = persist_pca(scores_df, meta, feats_raw, model_info)
    var = model_info["explained_variance_ratio"]
    print("✓ PCA global completado")
    print(f"  Varianza explicada: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}% PC3={var[2]*100:.1f}%")
    print(f"  Scores: {artifacts.scores_path}")
    print(f"  Modelo: {artifacts.model_path}")


if __name__ == "__main__":
    main()
