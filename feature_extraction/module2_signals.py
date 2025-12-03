from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Configuración base (alineada con Módulo 1)
TIME_SAMPLE_RATE_HZ: float = 10.0
DEFAULT_SAVGOL_WINDOW: int = 11  # Debe ser impar y mayor que el orden
DEFAULT_SAVGOL_POLY: int = 2
DEFAULT_BASE_DIR: Path = Path("data") / "module1_ingestion" / "2024_Bahrain_Grand_Prix_R"


@dataclass
class Module1Artifacts:
    telemetry_time: pd.DataFrame
    telemetry_distance: pd.DataFrame
    metadata: pd.DataFrame


def load_module1_artifacts(base_dir: Path) -> Module1Artifacts:
    """Carga los artefactos del Módulo 1 desde el directorio dado."""
    telemetry_time = pd.read_csv(base_dir / "telemetry_time_10hz.csv")
    telemetry_distance = pd.read_csv(base_dir / "telemetry_distance_aligned.csv")
    metadata = pd.read_csv(base_dir / "laps_metadata.csv")
    return Module1Artifacts(
        telemetry_time=telemetry_time,
        telemetry_distance=telemetry_distance,
        metadata=metadata,
    )


def _savgol(series: pd.Series, window: int, poly: int) -> np.ndarray:
    """
    Aplica Savitzky-Golay a una serie 1D.
    Ajusta dinámicamente la ventana para que sea impar y no exceda el tamaño de la serie.
    """
    if series.size == 0:
        return series.to_numpy()

    window = min(window, series.size if series.size % 2 == 1 else series.size - 1)
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)

    return savgol_filter(series.to_numpy(), window_length=window, polyorder=poly, mode="interp")


def _compute_dynamics_group(
    group: pd.DataFrame,
    sample_rate_hz: float,
    window: int,
    poly: int,
) -> pd.DataFrame:
    """
    Calcula aceleraciones y jerks para un grupo (Driver, LapNumber).
    Se usa velocidad y posición suavizadas para obtener derivadas estables.
    """
    df = group.sort_values("RelativeTime_s").copy()
    dt = 1.0 / sample_rate_hz
    eps = 1e-6

    # Suavizamos velocidad y posición para minimizar ruido antes de derivar
    speed_mps = _savgol(df["Speed"] * (1000 / 3600), window, poly)
    x_smooth = _savgol(df["X"], window, poly)
    y_smooth = _savgol(df["Y"], window, poly)

    # Velocidades en el plano XY
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)

    # Aceleraciones (derivada de velocidad en cada eje)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)

    speed_norm = np.clip(np.sqrt(vx**2 + vy**2), eps, None)

    # Descomposición en tangencial (longitudinal) y normal (lateral) respecto a la trayectoria
    a_t = (ax * vx + ay * vy) / speed_norm  # proyección en la dirección de la velocidad (longitudinal)
    a_n = (ax * vy - ay * vx) / speed_norm  # componente perpendicular (lateral); signo indica giro

    # Jerk = derivada de la aceleración
    jerk_t = np.gradient(a_t, dt)
    jerk_n = np.gradient(a_n, dt)

    # Proxy de energía del neumático: potencia ~ F · v => a * v; integramos |a|*v para tener un índice acumulado
    power_proxy = (np.abs(a_t) + np.abs(a_n)) * speed_norm
    energy_proxy = np.cumsum(power_proxy * dt)

    df["Speed_mps"] = speed_mps
    df["VX"] = vx
    df["VY"] = vy
    df["AX_long"] = a_t
    df["AY_lat"] = a_n
    df["Jerk_long"] = jerk_t
    df["Jerk_lat"] = jerk_n
    df["TireEnergyProxy"] = energy_proxy

    return df


def enrich_telemetry_time(
    telemetry_time: pd.DataFrame,
    sample_rate_hz: float = TIME_SAMPLE_RATE_HZ,
    window: int = DEFAULT_SAVGOL_WINDOW,
    poly: int = DEFAULT_SAVGOL_POLY,
) -> pd.DataFrame:
    """
    Enriquecer telemetría temporal con dinámicas (aceleraciones, jerks, energía neumático).
    Procesa por piloto y vuelta para evitar saltos entre vueltas distintas.
    """
    required_cols = {"Speed", "X", "Y", "RelativeTime_s", "Driver", "LapNumber"}
    missing = required_cols - set(telemetry_time.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias en telemetría temporal: {missing}")

    enriched_frames: List[pd.DataFrame] = []
    for (driver, lap), group in telemetry_time.groupby(["Driver", "LapNumber"]):
        if len(group) < max(window, poly + 2):
            # Si hay muy pocos puntos, evitamos Savitzky-Golay; usamos gradientes directos
            enriched_frames.append(
                _compute_dynamics_group(group, sample_rate_hz, window=max(5, poly + 2), poly=poly)
            )
        else:
            enriched_frames.append(_compute_dynamics_group(group, sample_rate_hz, window, poly))

    return pd.concat(enriched_frames, ignore_index=True)


def compute_lap_features(enriched: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por vuelta usando la telemetría enriquecida y metadatos originales.
    Incluye suavidad (jerk), agresividad de freno, y energía acumulada.
    """
    if enriched.empty:
        return pd.DataFrame()

    agg_rows: List[Dict[str, float]] = []

    # Indexar metadatos por Driver/LapNumber para acceso rápido
    meta_idx = metadata.set_index(["Driver", "LapNumber"])

    for (driver, lap), group in enriched.groupby(["Driver", "LapNumber"]):
        meta = meta_idx.loc[(driver, lap)] if (driver, lap) in meta_idx.index else {}
        energy_idx = group["TireEnergyProxy"].iloc[-1] if "TireEnergyProxy" in group else np.nan

        agg_rows.append(
            {
                "Driver": driver,
                "LapNumber": lap,
                "LapTimeSeconds": meta.get("LapTimeSeconds", np.nan),
                "Compound": meta.get("Compound", ""),
                "TyreLife": meta.get("TyreLife", np.nan),
                "Avg_Speed_mps": group["Speed_mps"].mean(),
                "Avg_Throttle": group["Throttle"].mean() if "Throttle" in group else np.nan,
                "Brake_Aggression": group["Brake"].std() if "Brake" in group else np.nan,
                "MeanAbs_Jerk_Long": group["Jerk_long"].abs().mean(),
                "MeanAbs_Jerk_Lat": group["Jerk_lat"].abs().mean(),
                "Max_Lateral_g": group["AY_lat"].abs().max() / 9.81,
                "Max_Longitudinal_g": group["AX_long"].abs().max() / 9.81,
                "Energy_Index": energy_idx,
            }
        )

    return pd.DataFrame(agg_rows)


def persist_module2_outputs(
    base_dir: Path,
    enriched_time: pd.DataFrame,
    lap_features: pd.DataFrame,
) -> Tuple[Path, Path]:
    """
    Guarda los artefactos enriquecidos:
    - telemetría temporal con dinámicas
    - features agregadas por vuelta
    """
    output_dir = base_dir
    enriched_path = output_dir / "telemetry_time_10hz_enriched.csv"
    lap_features_path = output_dir / "lap_features_module2.csv"

    enriched_time.to_csv(enriched_path, index=False)
    lap_features.to_csv(lap_features_path, index=False)

    return enriched_path, lap_features_path


def run_module2(
    base_dir: Path = DEFAULT_BASE_DIR,
    sample_rate_hz: float = TIME_SAMPLE_RATE_HZ,
    window: int = DEFAULT_SAVGOL_WINDOW,
    poly: int = DEFAULT_SAVGOL_POLY,
) -> Tuple[Path, Path]:
    """
    Orquestador del Módulo 2:
    - Carga artefactos del Módulo 1
    - Suaviza velocidad/posiciones con Savitzky-Golay
    - Calcula aceleraciones, jerk longitudinal/lateral y energía proxy de neumáticos
    - Agrega features por vuelta y persiste resultados
    """
    artifacts = load_module1_artifacts(base_dir)
    enriched_time = enrich_telemetry_time(
        artifacts.telemetry_time,
        sample_rate_hz=sample_rate_hz,
        window=window,
        poly=poly,
    )
    lap_features = compute_lap_features(enriched_time, artifacts.metadata)
    return persist_module2_outputs(base_dir, enriched_time, lap_features)


def main() -> None:
    """
    Ejemplo reproducible con Bahrein 2024 (usa los outputs del Módulo 1 en data/module1_ingestion/2024_Bahrain_Grand_Prix_R).
    """
    enriched_path, lap_features_path = run_module2()
    print("✓ Módulo 2 completado")
    print(f"  Telemetría enriquecida: {enriched_path}")
    print(f"  Features por vuelta:   {lap_features_path}")


if __name__ == "__main__":
    main()
