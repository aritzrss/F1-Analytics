# F1-Analytics

Proyecto universitario de analítica de rendimiento en Fórmula 1 usando FastF1, con un pipeline completo orientado a Industria 4.0: ingesta robusta, procesamiento de señales, ingeniería de características, modelado predictivo y visualización interactiva (Gemelo Digital).

## Estado actual y scripts principales

- `feature_extraction/module1_ingestion.py` (Módulo 1): Ingesta orientada a análisis avanzado. Configura caché, carga una sesión específica, y **alinea la telemetría en tiempo (10 Hz) y en espacio (malla de distancia)**. Persistimos artefactos limpios.
- `feature_extraction/module2_signals.py` (Módulo 2): Procesamiento de señales. Aplica filtros Savitzky-Golay, calcula derivadas físicas (Jerk, G-forces) y genera proxies de energía de neumático.
- `feature_extraction/module3_pca_global_normalized.py` (Módulo 3): Reducción de dimensionalidad. Normaliza datos por evento y extrae componentes principales (PC1/PC2) para identificar estilos de conducción.
- `feature_extraction/module4_modeling.py` (Módulo 4): Modelado predictivo. Entrena modelos de Machine Learning (RandomForest/XGBoost) para predecir tiempos de vuelta y explica las variables con SHAP.
- **`app.py` (Módulo 5)**: **Nuevo Dashboard interactivo**. Aplicación Streamlit que consume los artefactos generados (offline) para visualizar telemetría comparada, física vehicular y explicabilidad del modelo AI.

## Cuándo usar cada script (Pipeline)

1. **Ingesta:** Ejecuta `module1_ingestion.py` (o `batch_ingest.py`) para descargar y alinear datos crudos.
2. **Física:** Ejecuta `module2_signals.py` para calcular derivadas, energías y suavizado.
3. **Features & AI:** Ejecuta `merge_lap_features.py`, luego los scripts de PCA (Módulo 3) y finalmente `module4_modeling.py` para entrenar la IA.
4. **Visualización:** Ejecuta `streamlit run app.py` para explorar los datos, comparar pilotos (Ghost Car) y validar el modelo.

---

## Módulo 1 – Ingesta, sincronización y alineación espacial

Archivo: `feature_extraction/module1_ingestion.py`

### Qué hace
- Habilita caché local (`.fastf1-cache`) para no re-descargar datos.
- Carga una sesión FastF1 con laps, telemetría y clima.
- **Alineación Temporal (10 Hz)**: Remuestreo uniforme para sincronizar sensores (RPM, Speed, Throttle, Brake, DRS).
- **Alineación Espacial (Distancia)**: Reinicia la distancia a 0 por vuelta e interpola las señales sobre una malla de 1 m. Esto permite comparar dos vueltas en el mismo punto físico de la pista, base fundamental para el "Ghost Car" y estudios de dinámica vehicular.
- Persiste artefactos en `data/module1_ingestion/<year>_<event>_<session_type>/`.

---

## Módulo 2 – Procesamiento de señales y métricas físicas

Archivo: `feature_extraction/module2_signals.py`

### Qué hace
- **Suavizado Savitzky–Golay**: Elimina ruido de alta frecuencia en velocidad y posición sin introducir retardo de fase, permitiendo derivadas limpias.
- **Dinámica Vehicular**:
  - Descompone la aceleración en **Longitudinal** (frenada/tracción) y **Lateral** (curva).
  - Calcula **Jerk** (derivada de la aceleración): mide la brusquedad de los inputs del piloto (pedales/volante).
- **Proxy de Energía**: Integra `(|a_lat| + |a_long|) * velocidad` para estimar la carga disipada en el neumático (J/kg), útil para analizar degradación.

---

## Módulo 3 – Ingeniería de features y PCA

Archivo: `feature_extraction/module3_pca_global_normalized.py`

### Qué hace
- **Normalización por Evento**: Aplica Z-score por grupo (Año, Evento) a las features físicas. Esto elimina el sesgo del circuito (ej. Mónaco es lento, Monza es rápido) para que el PCA se centre en el estilo y la estrategia.
- **PCA (Principal Component Analysis)**: Reduce la dimensionalidad.
  - **PC1**: Generalmente correlaciona con el **Ritmo y Gestión** (Energy Index, Velocidad media).
  - **PC2**: Generalmente captura el **Estilo de Conducción** (Agresividad en freno, Jerk, Picos de G).

---

## Módulo 4 – Modelado predictivo (LapTime)

Archivo: `feature_extraction/module4_modeling.py`

### Qué hace
- Carga el dataset consolidado (`all_lap_features.csv`) y los scores del PCA.
- Entrena modelos de regresión (RandomForest, XGBoost, etc.) para predecir el `LapTimeSeconds`.
- **Explicabilidad (SHAP)**: Calcula valores Shapley para cuantificar qué variables físicas (ej. Energía, Jerk, Temperatura) influyen más en el tiempo de vuelta, ofreciendo insights de ingeniería.
- Genera métricas de error (MAE, R2) y gráficas de residuos para validar la robustez del modelo.

---

## Módulo 5 – Visualización y Gemelo Digital (Dashboard)

Archivo: `app.py`

### Descripción
Interfaz gráfica desarrollada en **Streamlit** que actúa como la capa de usuario final ("Industry 4.0 Dashboard"). No descarga datos en tiempo real, sino que explota los "artefactos" generados por los módulos anteriores, garantizando velocidad y disponibilidad offline.

### Funcionalidades por Pestaña

1.  **📊 Telemetría & Ghost Car**:
    *   **Ghost Car Delta**: Reconstruye matemáticamente el tiempo a partir de la distancia y velocidad ($t = \int v^{-1} dx$) para mostrar la ganancia/pérdida de tiempo metro a metro entre dos pilotos.
    *   **Comparativa de Velocidad**: Gráfica alineada espacialmente para detectar diferencias en puntos de frenada y velocidad mínima en curva.

2.  **🧪 Física & Neumáticos**:
    *   **Energía Acumulada**: Visualiza la curva de `TireEnergyProxy` a lo largo de la vuelta para comparar gestión de gomas.
    *   **Círculo de Fricción (G-G Diagram)**: Scatter plot de Aceleración Lateral vs Longitudinal para evaluar el uso del grip disponible.
    *   **Mapa de Jerk**: Identifica en el circuito (mapa X/Y) dónde el piloto es más brusco con los controles.

3.  **🧬 Estilo de Conducción (PCA)**:
    *   Visualización del **Espacio Latente (PC1 vs PC2)**. Permite ver clusters de pilotos, diferencias entre compuestos y la evolución del estilo a lo largo de la carrera.
    *   Tooltips interactivos con metadatos de vuelta.

4.  **🤖 Modelado AI (M4)**:
    *   **Feature Importance**: Gráfico de barras con valores SHAP (qué física importa más para el modelo).
    *   **Evaluación**: Scatter plot de *Predicho vs Real* para auditar la precisión de la Inteligencia Artificial.

### Cómo ejecutar
Asegúrate de tener instaladas las librerías necesarias:
```bash
pip install streamlit plotly pandas numpy
```
Ejecuta la aplicación desde la raíz del proyecto:
```bash
streamlit run app.py
```
*Nota: La aplicación requiere que hayas ejecutado previamente los Módulos 1 y 2 para al menos un evento.*

---

## Batch y consolidación de datos

Para escalar el dataset a múltiples años y carreras:

1.  **Ingesta Masiva**: `python feature_extraction/batch_ingest.py`
    Recorre años y rondas definidos, generando carpetas con telemetría alineada.
2.  **Procesamiento Masivo**: `python feature_extraction/batch_module2.py`
    Calcula la física para todas las sesiones descargadas.
3.  **Consolidación**: `python feature_extraction/merge_lap_features.py`
    Concatena todos los `lap_features` en un único CSV maestro (`all_lap_features.csv`).
4.  **PCA Global**: `python feature_extraction/module3_pca_global_normalized.py`
    Recalcula el PCA con el dataset histórico completo.
5.  **Reentrenamiento**: `python feature_extraction/module4_modeling.py`
    Genera un nuevo modelo predictivo con más datos.

---

## Features — Física, Matemáticas y Cálculo (Detallado)

Esta sección documenta con rigor las features calculadas en el pipeline.

- **Preprocesado y derivadas**: Se aplica un suavizado **Savitzky–Golay** sobre las series de posición y velocidad para reducir ruido sin introducir retardo de fase.
  - Velocidades: $v_x = \mathrm{d}x/\mathrm{d}t$, $v_y = \mathrm{d}y/\mathrm{d}t$.
  - Aceleraciones: $a_x = \mathrm{d}v_x/\mathrm{d}t$, $a_y = \mathrm{d}v_y/\mathrm{d}t$.

- **Aceleraciones descompuestas (tangencial / lateral)**
  - Aceleración tangencial (longitudinal) $a_t$: Proyección del vector aceleración sobre la velocidad. Representa frenada y tracción.
    $$a_t = \frac{a_x v_x + a_y v_y}{\|v\|}$$
  - Aceleración lateral (normal) $a_n$: Componente perpendicular que explica la curva.
    $$a_n = \frac{a_x v_y - a_y v_x}{\|v\|}$$

- **Jerk (Tasa de cambio de aceleración)**
  - is the rate of change of an object's acceleration over time
  - $j_t = \mathrm{d}a_t/\mathrm{d}t$ (m/s³). Picos altos indican transiciones bruscas en pedales o volante.
  - `MeanAbs_Jerk`: Indicador agregado de suavidad de conducción.

- **Proxy de Energía del Neumático (TireEnergyProxy)**
  - Integral aproximada de la potencia específica demandada al neumático.
    $$E' \approx \sum (|a_t| + |a_n|) \cdot \|v\| \cdot \Delta t$$
  - Unidades: J/kg (aprox). Permite comparar la demanda física impuesta a la goma entre distintos pilotos y estrategias.

---

## 🚀 Despliegue de Modelos (BentoML)


### 1. Servir el Modelo (API)
Abre una terminal nueva y ejecuta:
```bash
bentoml serve service:svc --reload
```
Esto iniciará un servidor en `http://localhost:3000`. Puedes probar el Swagger UI en esa URL o usar el comando `curl`.

### 2. Consumir desde el Dashboard
1. Ve a la pestaña **"🛠️ Lab de IA & Despliegue"** en la aplicación Streamlit.
2. Introduce los valores de telemetría (Vida neumático, Velocidad, Ajustes de PCA).
3. Pulsa "Enviar a API" para recibir la predicción del servidor BentoML.

## 🛠️ Entrenamiento Interactivo
En la misma pestaña del Lab, puedes:
- Modificar hiperparámetros (n_estimators, max_depth).
- Re-entrenar el modelo Random Forest en vivo.
- Ver cómo mejora (o empeora) el MAE/R2 instantáneamente.
