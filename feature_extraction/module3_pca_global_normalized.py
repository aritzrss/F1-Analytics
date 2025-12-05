from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Use script directory to ensure paths work whether script is run from
# project root or from inside the feature_extraction/ folder.
_script_dir = Path(__file__).resolve().parent
GLOBAL_INPUT: Path = _script_dir / "data" / "module1_ingestion" / "all_lap_features.csv"
SCORES_OUT: Path = _script_dir / "data" / "module1_ingestion" / "pca_scores_global_norm.csv"
MODEL_OUT: Path = _script_dir / "data" / "module1_ingestion" / "pca_model_global_norm.json"

BASE_FEATURES: List[str] = [
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


@dataclass
class PCAArtifacts:
    scores_path: Path
    model_path: Path


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para PCA global normalizado: {missing}")
    return df


def normalize_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza por evento para eliminar efecto pista:
    - LapTimeSeconds -> z-score dentro de cada (Year, Event)
    - Energy_Index -> dividir por longitud media de vuelta (aprox usando LapTime y Avg_Speed)
    - Avg_Speed_mps -> z-score por evento
    - Jerk y g: z-score por evento
    """
    df = df.copy()
    # z-score por evento para LapTime, Speed, Jerk, g, TyreLife
    for col in ["LapTimeSeconds", "Avg_Speed_mps", "MeanAbs_Jerk_Long", "MeanAbs_Jerk_Lat", "Max_Lateral_g", "Max_Longitudinal_g", "Brake_Aggression"]:
        df[f"{col}_norm"] = df.groupby(["Year", "Event"])[col].transform(lambda s: (s - s.mean()) / (s.std() if s.std() not in (0, None) else 1))
    # Energy_Index: aproximación por longitud de vuelta (Speed * LapTime)
    df["Lap_Distance_est"] = df["Avg_Speed_mps"] * df["LapTimeSeconds"]
    df["Energy_per_m"] = df["Energy_Index"] / df["Lap_Distance_est"].replace({0: pd.NA})
    df["Energy_per_m_norm"] = df.groupby(["Year", "Event"])["Energy_per_m"].transform(lambda s: (s - s.mean()) / (s.std() if s.std() not in (0, None) else 1))
    # TyreLife: opcionalmente z-score
    df["TyreLife_norm"] = df.groupby(["Year", "Event"])["TyreLife"].transform(lambda s: (s - s.mean()) / (s.std() if s.std() not in (0, None) else 1))
    return df


def build_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = ["Driver", "LapNumber", "Compound", "Year", "Event", "SessionType"]
    meta = df[meta_cols].copy()
    feat_cols = [
        "LapTimeSeconds_norm",
        "Energy_per_m_norm",
        "MeanAbs_Jerk_Long_norm",
        "MeanAbs_Jerk_Lat_norm",
        "Avg_Speed_mps_norm",
        "Brake_Aggression_norm",
        "Max_Lateral_g_norm",
        "Max_Longitudinal_g_norm",
        "TyreLife_norm",
    ]
    feats_norm = df[feat_cols].copy()
    # Conservar columnas crudas para los scores (no para PCA)
    feats_raw = df[["LapTimeSeconds", "Energy_per_m"]].copy()
    X = feats_norm.copy()
    # One-hot de Compound (Year ya está absorbido en la normalización por evento)
    compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound")
    X = pd.concat([X, compound_dummies], axis=1)
    # Drop filas con NaN
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)
    feats_raw = feats_raw.loc[mask].reset_index(drop=True)
    return X, meta, feats_raw


def run_pca(X: pd.DataFrame, n_components: int = N_COMPONENTS) -> Tuple[pd.DataFrame, dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(Xs)
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    model_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "feature_names": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return scores_df, model_info


def persist(scores_df: pd.DataFrame, meta: pd.DataFrame, feats_raw: pd.DataFrame, model_info: dict) -> PCAArtifacts:
    scores_with_meta = pd.concat([meta.reset_index(drop=True), feats_raw.reset_index(drop=True), scores_df], axis=1)
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores_with_meta.to_csv(SCORES_OUT, index=False)
    MODEL_OUT.write_text(json.dumps(model_info, indent=2))
    return PCAArtifacts(scores_path=SCORES_OUT, model_path=MODEL_OUT)


def main() -> None:
    df = load_dataset(GLOBAL_INPUT)
    df_norm = normalize_by_event(df)
    X, meta, feats_raw = build_matrix(df_norm)
    scores_df, model_info = run_pca(X)
    artifacts = persist(scores_df, meta, feats_raw, model_info)
    var = model_info["explained_variance_ratio"]
    print("✓ PCA global normalizado completado")
    print(f"  Varianza explicada: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}% PC3={var[2]*100:.1f}%")
    print(f"  Scores: {artifacts.scores_path}")
    print(f"  Modelo: {artifacts.model_path}")


if __name__ == "__main__":
    main()
