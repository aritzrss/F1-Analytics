import bentoml
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Definir esquema de entrada con Pydantic
class LapFeaturesInput(BaseModel):
    SessionType: str = "R"
    Compound: str = "MEDIUM"
    TyreLife: int = 5
    Avg_Speed_mps: float = 55.0
    Avg_Throttle: float = 40.0
    Brake_Aggression: float = 5.0
    MeanAbs_Jerk_Long: float = 2.0
    MeanAbs_Jerk_Lat: float = 2.0
    Max_Lateral_g: float = 3.0
    Max_Longitudinal_g: float = 2.0
    Energy_Index: float = 500.0
    PC2: float = 0.0
    PC3: float = 0.0
    
    # Contexto Extra (Necesario para el pipeline, aunque sea dummy)
    Year: int = 2024
    Event: str = "Bahrain Grand Prix"
    Driver: str = "11"
    LapNumber: int = 1

@bentoml.service(
    name="f1_laptime_service",
    traffic={"timeout": 10},
    resources={"cpu": "1"}, 
)
class F1LaptimeService:
    def __init__(self):
        # En la nueva API, cargamos el modelo directamente o usamos bentoml.models.get
        # sklearn.load_model carga el modelo en memoria (objeto python)
        try:
            self.model = bentoml.sklearn.load_model("f1_laptime_predictor:latest")
        except Exception as e:
            raise RuntimeError(f"FATAL: No se encontró el modelo 'f1_laptime_predictor' en BentoML. \n"
                               f"Ejecuta: `uv run feature_extraction/module4_modeling.py`\n"
                               f"Error: {e}")

    @bentoml.api
    def predict_laptime(self, input_data: LapFeaturesInput) -> dict:
        # Pydantic v2 usa model_dump(), v1 usaba dict(). BentoML 1.4 suele usar Pydantic v2 si está installed.
        # Probaremos getattr o try/except para compatibilidad, o .dict() si es v1
        if hasattr(input_data, "model_dump"):
            data_dict = input_data.model_dump()
        else:
            data_dict = input_data.dict()

        df = pd.DataFrame([data_dict])
        
        # Añadir columnas dummy que el preprocesador espera pero que no afectan la predicción individual
        # (o que asumimos valores por defecto para inferencia)
        if "Year" not in df.columns:
            df["Year"] = 2024
        if "Event" not in df.columns:
            df["Event"] = "Bahrain Grand Prix"
        if "Driver" not in df.columns:
            df["Driver"] = "11"
        if "LapNumber" not in df.columns:
            df["LapNumber"] = 1
        
        # Predecir usando el método predict estándar de sklearn
        prediction = self.model.predict(df)
        
        return {"predicted_laptime_s": float(prediction[0])}
