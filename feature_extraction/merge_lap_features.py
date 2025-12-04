from __future__ import annotations

"""
Merge de artefactos del Módulo 2 (lap_features_module2.csv) a un dataset multiaño/multi-GP.

Uso:
    uv run python feature_extraction/merge_lap_features.py

Lee todos los directorios bajo data/module1_ingestion/* y concatena los lap_features_module2.csv
añadiendo columnas Year, Event, SessionType inferidas del nombre de carpeta.
Guarda:
    data/module1_ingestion/all_lap_features.csv
"""

import re
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path("data") / "module1_ingestion"
OUTPUT_FILE = BASE_DIR / "all_lap_features.csv"


def parse_folder_name(folder: Path):
    """
    Espera nombres tipo '2024_Bahrain_Grand_Prix_R'
    Devuelve year, event_name, session_type
    """
    name = folder.name
    parts = name.split("_")
    if len(parts) < 2:
        return None
    year = parts[0]
    session_type = parts[-1]
    event_name = "_".join(parts[1:-1]) if len(parts) > 2 else "UnknownEvent"
    if not year.isdigit():
        return None
    return int(year), event_name, session_type


def main() -> None:
    frames: List[pd.DataFrame] = []
    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        parsed = parse_folder_name(folder)
        if parsed is None:
            continue
        year, event_name, session_type = parsed
        lap_file = folder / "lap_features_module2.csv"
        if lap_file.exists():
            df = pd.read_csv(lap_file)
            df["Year"] = year
            df["Event"] = event_name
            df["SessionType"] = session_type
            frames.append(df)

    if not frames:
        print("No se encontraron lap_features_module2.csv")
        return

    all_df = pd.concat(frames, ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Merge completado: {OUTPUT_FILE} ({len(all_df)} filas)")


if __name__ == "__main__":
    main()
