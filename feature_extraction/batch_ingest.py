from __future__ import annotations

"""
Batch ingestión de múltiples eventos/sesiones con el Módulo 1.

Uso:
    uv run python feature_extraction/batch_ingest.py

Obtiene el calendario real de FastF1 para cada año, recorre los eventos existentes
y aplica reintentos/pausas para evitar rate limits.
"""

import time
from pathlib import Path
from typing import List

import fastf1

# Try package import first (when run as module from project root),
# otherwise fall back to local import so the script can be run
# directly from the `feature_extraction/` directory.
try:
    from feature_extraction.module1_ingestion import ingest_session
except ModuleNotFoundError:
    from module1_ingestion import ingest_session  # type: ignore

# Configuración: ajustar según necesidad
YEARS: List[int] = [2021, 2022, 2023, 2024]
SESSION_TYPES: List[str] = ["R"]  # 'R' (Race), 'S' (Sprint)
SLEEP_BETWEEN_CALLS = 2.0  # segundos entre sesiones para evitar rate limit
MAX_RETRIES = 2  # reintentos por sesión
SLEEP_BETWEEN_RETRIES = 10.0  # segundos de espera antes de reintentar


def fetch_rounds_for_year(year: int) -> List[dict]:
    """Devuelve una lista de eventos (RoundNumber, EventName) usando el calendario real."""
    try:
        schedule = fastf1.get_event_schedule(year)
        rounds = []
        for _, row in schedule.iterrows():
            rnd = int(row["RoundNumber"])
            name = row["EventName"]
            # Filtrar rondas inválidas (ej. 0 = test de pretemporada) y eventos sin GP
            if rnd <= 0 or "Test" in str(name) or "Pre-Season" in str(name):
                continue
            rounds.append({"round": rnd, "event": name})
        return rounds
    except Exception as exc:
        print(f"✗ No se pudo obtener calendario {year}: {exc}")
        return []


def session_already_ingested(year: int, event_name: str, session_type: str) -> bool:
    """Devuelve True si ya existe la carpeta de salida para esa sesión."""
    folder_name = f"{year}_{event_name.replace(' ', '_')}_{session_type}"
    out_dir = Path("data") / "module1_ingestion" / folder_name
    return out_dir.exists() and any(out_dir.iterdir())


def main() -> None:
    for year in YEARS:
        rounds = fetch_rounds_for_year(year)
        if not rounds:
            continue
        for evt in rounds:
            round_number = evt["round"]
            event_name = evt["event"]
            for session_type in SESSION_TYPES:
                if session_already_ingested(year, event_name, session_type):
                    print(f"✓ Saltando {year} {event_name} {session_type} (ya existe)")
                    continue
                attempts = 0
                while attempts <= MAX_RETRIES:
                    try:
                        print(f"\n>>> Ingestando {year} Round {round_number} Session {session_type} ({event_name})")
                        ingest_session(year=year, grand_prix=round_number, session_type=session_type)
                        time.sleep(SLEEP_BETWEEN_CALLS)
                        break
                    except Exception as exc:
                        attempts += 1
                        print(f"⚠️  Falló {year} Round {round_number} {session_type} (intento {attempts}/{MAX_RETRIES+1}): {exc}")
                        if attempts > MAX_RETRIES:
                            print(f"✗ Se omite {year} Round {round_number} {session_type}")
                            break
                        time.sleep(SLEEP_BETWEEN_RETRIES)


if __name__ == "__main__":
    main()
