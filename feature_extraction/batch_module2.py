from __future__ import annotations

"""
Batch para ejecutar el Módulo 2 (señales) sobre todos los artefactos del Módulo 1.

Uso:
    uv run python feature_extraction/batch_module2.py

Recorre los subdirectorios de data/module1_ingestion/* y, si no existe
`lap_features_module2.csv`, ejecuta run_module2(base_dir=...). Esto genera
telemetry_time_10hz_enriched.csv y lap_features_module2.csv por sesión.
"""

from pathlib import Path

# Import run_module2 either via package import (when running from repo root)
# or fallback to local import (when running from inside feature_extraction/).
try:
    from feature_extraction.module2_signals import run_module2
except ModuleNotFoundError:
    from module2_signals import run_module2  # type: ignore


def main() -> None:
    # Use the script's directory to locate the data folder reliably whether
    # the script is executed from the project root or from inside
    # the `feature_extraction/` folder.
    script_dir = Path(__file__).resolve().parent
    base_root = script_dir / "data" / "module1_ingestion"
    for folder in base_root.iterdir():
        if not folder.is_dir():
            continue
        lap_features = folder / "lap_features_module2.csv"
        telemetry_time = folder / "telemetry_time_10hz.csv"
        if lap_features.exists():
            print(f"✓ Saltando {folder.name} (lap_features_module2.csv ya existe)")
            continue
        if not telemetry_time.exists():
            print(f"✗ Saltando {folder.name} (no hay telemetry_time_10hz.csv)")
            continue
        try:
            print(f">>> Ejecutando Módulo 2 en {folder.name}")
            run_module2(base_dir=folder)
        except Exception as exc:
            print(f"⚠️  Falló {folder.name}: {exc}")
            continue


if __name__ == "__main__":
    main()
