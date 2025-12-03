from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import fastf1
import numpy as np
import pandas as pd

# Configuración base del módulo
CACHE_DIR: Path = Path(".fastf1-cache")
OUTPUT_DIR: Path = Path("data") / "module1_ingestion"
TIME_SAMPLE_RATE_HZ: float = 10.0  # Frecuencia objetivo para sincronizar sensores
DISTANCE_STEP_METERS: float = 1.0  # Resolución espacial para comparar vueltas

# Columnas típicas en telemetría que nos interesan para sincronización/espacio
DEFAULT_TELEMETRY_COLUMNS: Tuple[str, ...] = (
    "Speed",
    "Throttle",
    "Brake",
    "nGear",
    "RPM",
    "DRS",
    "Distance",
    "RelativeDistance",
    "X",
    "Y",
)


@dataclass
class LapTelemetryAligned:
    """Artefactos alineados de una vuelta específica."""

    driver_code: str
    lap_number: int
    compound: str
    tyre_life: Optional[int]
    lap_time_seconds: Optional[float]
    time_resampled: pd.DataFrame
    distance_aligned: pd.DataFrame


def configure_fastf1_cache(cache_dir: Path = CACHE_DIR) -> None:
    """Habilita la caché local de FastF1 para evitar descargas repetitivas."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session(
    year: int, grand_prix: Union[int, str], session_type: str = "R"
) -> fastf1.core.Session:
    """
    Carga una sesión de FastF1 con laps, telemetría y clima habilitados.
    """
    try:
        session = fastf1.get_session(year, grand_prix, session_type)
        session.load(laps=True, telemetry=True, weather=True)
    except Exception as exc:  # pragma: no cover - dependiente de la API externa
        raise RuntimeError(
            f"FastF1 no pudo cargar la sesión {year} {grand_prix} {session_type}: {exc}"
        ) from exc

    return session


def _select_numeric_columns(
    telemetry: pd.DataFrame, candidate_columns: Sequence[str]
) -> pd.DataFrame:
    """
    Devuelve solo las columnas numéricas existentes dentro de candidate_columns.
    Si ninguna coincide, toma todas las columnas numéricas disponibles.
    """
    existing_columns = [col for col in candidate_columns if col in telemetry.columns]
    if not existing_columns:
        existing_columns = [
            col for col in telemetry.columns if pd.api.types.is_numeric_dtype(telemetry[col])
        ]

    numeric_df = telemetry[existing_columns].apply(pd.to_numeric, errors="coerce")
    # Convertimos explícitamente a DataFrame plano (evita warnings de downcast en subclass Telemetry)
    numeric_df = pd.DataFrame(numeric_df).infer_objects(copy=False)
    return numeric_df


def resample_telemetry_by_time(
    lap_telemetry: pd.DataFrame,
    frequency_hz: float = TIME_SAMPLE_RATE_HZ,
    columns: Sequence[str] = DEFAULT_TELEMETRY_COLUMNS,
) -> pd.DataFrame:
    """
    Sincroniza señales en el dominio temporal a una frecuencia fija (ej. 10 Hz).
    Usamos una escala temporal relativa al inicio de la vuelta para poder solapar
    múltiples sensores y comparar vueltas independientemente del momento en la sesión.
    """
    if "SessionTime" not in lap_telemetry.columns:
        raise ValueError("La telemetría no contiene 'SessionTime' para sincronizar.")

    telemetry = lap_telemetry.sort_values("SessionTime").copy()
    telemetry["RelativeTime"] = telemetry["SessionTime"] - telemetry["SessionTime"].iloc[0]
    telemetry["RelativeTime_s"] = telemetry["RelativeTime"].dt.total_seconds()

    numeric_df = _select_numeric_columns(telemetry, columns)
    if numeric_df.empty:
        raise ValueError("No se encontraron columnas numéricas para remuestreo temporal.")

    numeric_df.index = pd.to_timedelta(telemetry["RelativeTime_s"], unit="s")
    step = pd.Timedelta(seconds=1.0 / frequency_hz)

    # Resample sobre el índice de tiempo relativo y extrae el índice resampleado como columna explícita
    resampled = (
        numeric_df.resample(step)
        .ffill()
        .bfill()
        .interpolate()
        .infer_objects(copy=False)
    )
    relative_time = resampled.index.to_series().dt.total_seconds().reset_index(drop=True)
    resampled = resampled.reset_index(drop=True)
    resampled.insert(0, "RelativeTime_s", relative_time)

    return resampled


def interpolate_telemetry_by_distance(
    lap_telemetry: pd.DataFrame,
    step_meters: float = DISTANCE_STEP_METERS,
    columns: Sequence[str] = DEFAULT_TELEMETRY_COLUMNS,
) -> pd.DataFrame:
    """
    Interpola señales sobre el espacio (distancia recorrida) para comparar vueltas
    en el mismo punto de pista. Esto permite hacer "ghost laps" consistentes.
    """
    if "Distance" not in lap_telemetry.columns:
        raise ValueError("La telemetría no contiene 'Distance' para interpolar.")

    telemetry = lap_telemetry.sort_values("Distance").copy()
    telemetry["Distance"] = telemetry["Distance"] - telemetry["Distance"].iloc[0]
    telemetry = telemetry.dropna(subset=["Distance"])

    if telemetry.empty:
        raise ValueError("No hay puntos de distancia válidos en la vuelta.")

    distance_values = telemetry["Distance"].to_numpy()
    unique_distance, unique_idx = np.unique(distance_values, return_index=True)
    telemetry_unique = telemetry.iloc[unique_idx]

    numeric_df = _select_numeric_columns(telemetry_unique, columns)
    if numeric_df.empty:
        raise ValueError("No se encontraron columnas numéricas para interpolar por distancia.")

    numeric_df = numeric_df.ffill().bfill().infer_objects(copy=False)

    max_distance = float(unique_distance.max()) if unique_distance.size else 0.0
    target_distance = np.arange(0.0, max_distance + step_meters, step_meters)

    interpolated: Dict[str, np.ndarray] = {}
    for column in numeric_df.columns:
        interpolated[column] = np.interp(
            target_distance, unique_distance, numeric_df[column].to_numpy()
        )

    distance_df = pd.DataFrame(interpolated)
    distance_df.insert(0, "Distance_m", target_distance)
    distance_df["Distance_pct"] = distance_df["Distance_m"] / max_distance if max_distance else 0.0

    return distance_df


def extract_laps_and_weather(
    session: fastf1.core.Session,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve dataframes con las vueltas y el clima, listos para persistir.
    Convierte campos temporales a segundos para análisis posteriores.
    """
    laps_df = session.laps.copy()
    if "LapTime" in laps_df.columns:
        laps_df["LapTimeSeconds"] = laps_df["LapTime"].dt.total_seconds()

    weather_df = session.weather_data.copy() if hasattr(session, "weather_data") else pd.DataFrame()
    if not weather_df.empty and "Time" in weather_df.columns:
        weather_df["TimeSeconds"] = weather_df["Time"].dt.total_seconds()

    return laps_df, weather_df


def build_driver_fastest_lap_alignment(
    session: fastf1.core.Session,
    driver_code: str,
    frequency_hz: float = TIME_SAMPLE_RATE_HZ,
    distance_step_m: float = DISTANCE_STEP_METERS,
) -> Optional[LapTelemetryAligned]:
    """
    Obtiene la vuelta más rápida del piloto y genera las versiones alineadas
    en tiempo (10 Hz) y espacio (malla de distancia fija).
    """
    driver_laps = session.laps.pick_drivers([driver_code])
    if driver_laps.empty:
        return None

    fastest_lap = driver_laps.pick_fastest()
    if fastest_lap is None or fastest_lap.empty:
        return None

    telemetry = fastest_lap.get_telemetry()
    if telemetry is None or telemetry.empty:
        return None

    time_resampled = resample_telemetry_by_time(telemetry, frequency_hz=frequency_hz)
    distance_aligned = interpolate_telemetry_by_distance(
        telemetry, step_meters=distance_step_m
    )

    lap_time = (
        float(fastest_lap["LapTime"].total_seconds())
        if pd.notna(fastest_lap.get("LapTime"))
        else None
    )

    return LapTelemetryAligned(
        driver_code=str(driver_code),
        lap_number=int(fastest_lap.get("LapNumber", 0)),
        compound=str(fastest_lap.get("Compound", "UNKNOWN")),
        tyre_life=int(fastest_lap.get("TyreLife", 0)) if pd.notna(fastest_lap.get("TyreLife")) else None,
        lap_time_seconds=lap_time,
        time_resampled=time_resampled,
        distance_aligned=distance_aligned,
    )


def persist_ingestion_outputs(
    session_label: str,
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    time_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    base_output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Guarda los artefactos de ingesta en CSV para facilitar el EDA posterior."""
    session_dir = base_output_dir / session_label
    session_dir.mkdir(parents=True, exist_ok=True)

    laps_df.to_csv(session_dir / "laps.csv", index=False)
    weather_df.to_csv(session_dir / "weather.csv", index=False)
    time_df.to_csv(session_dir / "telemetry_time_10hz.csv", index=False)
    distance_df.to_csv(session_dir / "telemetry_distance_aligned.csv", index=False)
    metadata_df.to_csv(session_dir / "laps_metadata.csv", index=False)

    return session_dir


def ingest_session(
    year: int,
    grand_prix: Union[int, str],
    session_type: str = "R",
    frequency_hz: float = TIME_SAMPLE_RATE_HZ,
    distance_step_m: float = DISTANCE_STEP_METERS,
    drivers: Optional[Iterable[str]] = None,
) -> Path:
    """
    Orquestador del MÓDULO 1:
    - Configura caché
    - Carga sesión (laps, clima, telemetría)
    - Sincroniza telemetría (tiempo y distancia) para la vuelta más rápida de cada piloto
    - Persiste artefactos listos para EDA/feature engineering
    """
    configure_fastf1_cache()
    session = load_session(year, grand_prix, session_type=session_type)

    laps_df, weather_df = extract_laps_and_weather(session)
    driver_list = list(drivers) if drivers is not None else list(session.drivers)

    aligned_laps: List[LapTelemetryAligned] = []
    for driver_no in driver_list:
        # session.drivers usa números, convertimos a código corto (VER/LEC/etc.)
        code = (
            session.get_driver(driver_no)["Abbreviation"]
            if isinstance(driver_no, (int, np.integer))
            else str(driver_no)
        )
        alignment = build_driver_fastest_lap_alignment(
            session, code, frequency_hz=frequency_hz, distance_step_m=distance_step_m
        )
        if alignment is not None:
            aligned_laps.append(alignment)

    if not aligned_laps:
        raise RuntimeError("No se pudieron alinear vueltas para ningún piloto.")

    time_frames: List[pd.DataFrame] = []
    distance_frames: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, Union[str, int, float, None]]] = []

    for aligned in aligned_laps:
        time_df = aligned.time_resampled.copy()
        time_df["Driver"] = aligned.driver_code
        time_df["LapNumber"] = aligned.lap_number
        time_df["Compound"] = aligned.compound
        time_df["TyreLife"] = aligned.tyre_life
        time_frames.append(time_df)

        distance_df = aligned.distance_aligned.copy()
        distance_df["Driver"] = aligned.driver_code
        distance_df["LapNumber"] = aligned.lap_number
        distance_df["Compound"] = aligned.compound
        distance_df["TyreLife"] = aligned.tyre_life
        distance_frames.append(distance_df)

        metadata_rows.append(
            {
                "Driver": aligned.driver_code,
                "LapNumber": aligned.lap_number,
                "Compound": aligned.compound,
                "TyreLife": aligned.tyre_life,
                "LapTimeSeconds": aligned.lap_time_seconds,
            }
        )

    telemetry_time_df = (
        pd.concat(time_frames, ignore_index=True) if time_frames else pd.DataFrame()
    )
    telemetry_distance_df = (
        pd.concat(distance_frames, ignore_index=True) if distance_frames else pd.DataFrame()
    )
    metadata_df = pd.DataFrame(metadata_rows)

    event_name = str(session.event.get("EventName", grand_prix)).replace(" ", "_")
    session_label = f"{year}_{event_name}_{session_type}"

    output_path = persist_ingestion_outputs(
        session_label=session_label,
        laps_df=laps_df,
        weather_df=weather_df,
        time_df=telemetry_time_df,
        distance_df=telemetry_distance_df,
        metadata_df=metadata_df,
    )

    return output_path


def main() -> None:
    """
    Ejemplo reproducible con Bahrein 2024 (Round 1, Carrera):
    - Sincroniza telemetría a 10 Hz
    - Interpola por distancia para comparar vueltas espacialmente
    """
    # Default alineado con los datos que ya recopila el equipo (temporada 2024)
    output_dir = ingest_session(year=2024, grand_prix=1, session_type="R")
    print(f"Artefactos del MÓDULO 1 guardados en: {output_dir}")


if __name__ == "__main__":
    main()
