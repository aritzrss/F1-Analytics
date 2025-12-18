# Análisis Integral y Defensa de Proyecto: F1-Analytics

Este documento contiene la auditoría técnica, justificación teórica, documentación académica y preparación para la defensa del proyecto "Análisis del Rendimiento en la Fórmula 1 mediante Datos de Telemetría".

---

## PASO 1: ANÁLISIS DE ARQUITECTURA Y "CLEAN CODE"

### 1. Flujo de Datos (Pipeline ETL)
El proyecto implementa una arquitectura secuencial modular, alineada con los principios de pipelines de datos modernos (ELT/ETL).

*   **Extracción (Extract):**
    *   **Ubicación:** `feature_extraction/module1_ingestion.py`
    *   **Mecanismo:** La clase `fastf1` gestiona la conexión con la API oficial. La función `ingest_session` (L255) actúa como orquestador, descargando datos de vueltas (`session.laps`), telemetría y clima.
    *   **Caching:** Se implementa específicamente en `configure_fastf1_cache` (L45) usando un directorio local `.fastf1-cache` para evitar latencia de red en ejecuciones sucesivas, vital para iteraciones rápidas en desarrollo.

*   **Transformación (Transform):**
    *   **Sincronización:** En `module1_ingestion.py`, las funciones `resample_telemetry_by_time` (L87) e `interpolate_telemetry_by_distance` (L126) normalizan los datos heterogéneos de los sensores.
        *   *Temporal:* Resampling a 10Hz (constante `TIME_SAMPLE_RATE_HZ`).
        *   *Espacial:* Interpolación lineal sobre una malla de distancia fija (`DISTANCE_STEP_METERS = 1.0`), esencial para comparar telemetría entre vueltas con velocidades distintas ("Ghost Car").
    *   **Enriquecimiento Físico:** `feature_extraction/module2_signals.py` calcula derivadas físicas (velocidad $\rightarrow$ aceleración $\rightarrow$ jerk) y métricas de ingeniería (`_compute_dynamics_group`, L51).

*   **Carga y Visualización (Load & Visualize):**
    *   **Persistencia:** Los datos procesados se guardan en formato CSV estructurado por sesión/evento en `data/module1_ingestion`, desacoplando el procesamiento de la visualización.
    *   **Consumo:** `app.py` lee estos "Feature Stores" locales. Usa `st.cache_data` (L35, L44) para cargar los DataFrames en memoria de forma eficiente, permitiendo interactividad en tiempo real con Streamlit y Plotly.

### 2. Calidad del Código (Clean Code & Buenas Prácticas)
El código demuestra madurez técnica superior a la media académica:

*   **Modularidad y Separación de Responsabilidades (SRP):**
    *   Cada script tiene un propósito único: `module1` para ingesta, `module2` para física, `module4` para ML.
    *   Uso de funciones pequeñas y atómicas (ej: `_select_numeric_columns`, `_savgol`), facilitando testabilidad.
*   **Tipado Estático (Type Hinting):**
    *   Uso extensivo de `typing` (List, Tuple, Optional) y `dataclasses`.
    *   Ejemplo en `module1_ingestion.py`:
        ```python
        def resample_telemetry_by_time(
            lap_telemetry: pd.DataFrame,
            frequency_hz: float = TIME_SAMPLE_RATE_HZ,
            ...
        ) -> pd.DataFrame:
        ```
    *   Esto mejora la legibilidad y permite análisis estático con herramientas como `mypy`.
*   **Manejo de Robustez:**
    *   Verificaciones defensivas: "Si hay muy pocos puntos, evitamos Savitzky-Golay" (`module2_signals.py`, L121).
    *   Manejo de excepciones en cálculos físicos en `app.py` (L66-92) para evitar que la UI colapse si faltan datos.
*   **Reproducibilidad:**
    *   Uso de semilla aleatoria (`random_state=42`) en `module4_modeling.py` (L103, L141) para garantizar que los modelos ML sean deterministas.

### 3. Justificación de Escalabilidad (Entorno Producción)
Tu implementación simula un **Feature Store** básico.
*   **Desacoplamiento:** Al guardar los CSVs intermedios, separas el coste computacional "caro" (descarga API, resampling 10Hz, derivadas) de la latencia "barata" de lectura en el dashboard. En producción real, estos CSVs serían tablas en BigQuery o archivos en S3/Parquet.
*   **Paralelismo Implícito:** El diseño por "Sesión/Evento" permite que futuras iteraciones procesen múltiples Grandes Premios en paralelo (Airflow/Dagster) sin conflictos de estado compartido.

---

## PASO 2: CONEXIÓN TEÓRICA (FÍSICA Y MATEMÁTICAS)

### 1. Manipulación de Datos
*   **Velocidad y Posición:** Extraídas crudas de API (`Speed`, `X`, `Y`) en `module1`.
*   **Tiempo Relativo:** Calculado en `resample_telemetry_by_time` como $t - t_{inicio}$.
*   **Aceleración (G-Force):** No viene directa del GPS con precisión. Tu código la deriva en `module2_signals.py`:
    *   $\vec{v} = (\dot{x}, \dot{y})$
    *   $\vec{a} = (\ddot{x}, \ddot{y})$

### 2. Matemática Aplicada

#### A. Derivación Numérica y Suavizado (Calculus)
Las señales de sensores tienen ruido de alta frecuencia. Derivar ruido amplifica el error.
*   **Solución:** Usas **Filtro Savitzky-Golay** en `module2_signals.py` (`_savgol`, L37).
    *   Matemáticamente, ajusta un polinomio local de grado `poly` (2) en una ventana `window` (11 muestras) usando mínimos cuadrados.
    *   Esto preserva mejor los picos de la señal (frenadas bruscas) que una media móvil simple.
*   **Cinemática:**
    *   Calculas gradientes numéricos (`np.gradient`) para obtener aceleración:
        $$ a_x[n] = \frac{v_x[n+1] - v_x[n-1]}{2\Delta t} $$

#### B. Descomposición de Vectores (Álgebra Lineal)
En `_compute_dynamics_group` (L80-82), proyectas la aceleración cartesiana ($a_x, a_y$) al marco de referencia del coche (Frenett-Serret frame local):
*   **Longitudinal ($a_t$):** Aceleración/Frenada. Producto escalar con el vector velocidad unitario.
    $$ a_{long} = \vec{a} \cdot \hat{v} = \frac{a_x v_x + a_y v_y}{|\vec{v}|} $$
*   **Lateral ($a_n$):** Fuerza centrífuga en curva. Producto cruz 2D.
    $$ a_{lat} = \frac{a_x v_y - a_y v_x}{|\vec{v}|} $$

#### C. Diagrama G-G y Círculo de Fricción
El gráfico "Círculo de Fricción" en `app.py` representa la envolvente de rendimiento del neumático.
*   Físicamente: $\sqrt{a_{lat}^2 + a_{long}^2} \leq \mu g$.
*   Tu código normaliza por gravedad ($9.81 m/s^2$) para visualizar Gs. Puntos fuera del círculo teórico indican peralte, aerodinámica (Downforce) o error de sensor.

#### D. Energía de Neumático (Proxy)
Calculas `TireEnergyProxy` integrando la potencia de las fuerzas de fricción:
$$ E \approx \int (|a_{lat}| + |a_{long}|) \cdot v \cdot dt $$
Esto modela el trabajo realizado por los neumáticos, correlacionando directamente con la temperatura y la degradación térmica (Energy Management).

---

## PASO 3: REDACCIÓN DE DOCUMENTACIÓN TÉCNICA

### Apartado: Metodología e Implementación

#### 3.1 Arquitectura del Sistema y Procesamiento de Señales
El sistema se ha diseñado bajo los principios de **Industria 4.0**, implementando un gemelo digital (*Digital Twin*) del comportamiento dinámico del monoplaza. El núcleo de procesamiento consta de un pipeline dividido en cuatro etapas modulares:

1.  **Ingesta y Sincronización Espacio-Temporal:** Se utiliza la librería `FastF1` como interfaz de extracción. Debido a la naturaleza heterogénea de los sensores (GPS a baja frecuencia vs. ECU), se aplica un algoritmo de *resampling* a 10 Hz para homogeneizar el dominio temporal. Adicionalmente, se implementa una interpolación espacial lineal (`module1_ingestion.py`), remapeando las series temporales a un dominio de distancia fija ($\Delta d = 1m$). Esto permite la comparación directa de telemetría entre vueltas ("Ghost Car Analysis") independientemente de las diferencias en velocidad de paso por curva.

2.  **Modelado Dinámico y Filtrado de Ruido:** Para reconstruir las fuerzas G (no disponibles con precisión en la fuente pública), se aplica cálculo diferencial numérico sobre los vectores de posición y velocidad. Para mitigar el ruido de cuantización inherente al GPS, se implementa un filtro **Savitzky-Golay** (polinomio de orden 2, ventana de 11 muestras), que suaviza la señal preservando los momentos de alta frecuencia característicos de las frenadas en F1. Las aceleraciones resultantes se descomponen vectorialmente en componentes longitudinales (tracción/frenado) y laterales (viraje) mediante proyección sobre el vector velocidad instantáneo.

3.  **Caracterización del Estilo de Conducción:** Se utiliza Análisis de Componentes Principales (**PCA**) sobre el espacio de características agregadas por vuelta (Jerk medio, agresividad en freno, Gs máximos) para reducir la dimensionalidad y detectar patrones latentes (suavidad vs. agresividad) que diferencian a los pilotos.

4.  **Predicción de Rendimiento (ML):** Se implementa un modelo de regresión (Gradient Boosting/Random Forest) validado mediante *K-Fold Cross Validation* para predecir tiempos de vuelta basándose en parámetros físicos y de gestión de neumáticos, logrando cuantificar el impacto de cada variable mediante valores SHAP.

#### 3.2 Justificación Tecnológica
*   **Pandas & NumPy:** Elegidos por su capacidad de vectorización (SIMD), permitiendo realizar cálculos cinemáticos sobre millones de puntos de datos de telemetría en milisegundos, frente a bucles iterativos convencionales.
*   **FastF1:** Proporciona acceso democratizado a datos de nivel industrial (Timing y Telemetría básica), sirviendo como capa de abstracción sobre la API oficial de la F1.
*   **Streamlit:** Facilita la creación rápida de dashboards interactivos, vital para la visualización exploratoria de datos complejos sin el overhead de desarrollo frontend tradicional.

---

## PASO 4: SIMULACIÓN DE DEFENSA (EL TRIBUNAL)

### Pregunta 1: Sincronización y Frecuencia de Muestreo
**Tribunal:** "Usted hace un resampling a 10 Hz. Sin embargo, los datos de GPS de la F1 suelen venir a 1-5 Hz y la telemetría del coche a mucho más. ¿No está inventando datos al interpolar a 10 Hz? ¿Cómo afecta esto a la precisión de sus derivadas de aceleración?"

**Respuesta Señor:**
"Es una excelente observación sobre la teoría de señales. Efectivamente, al hacer upsampling a 10 Hz desde una fuente GPS de menor frecuencia, estoy introduciendo puntos interpolados que no existen en la medición original. Sin embargo, esto es una decisión de diseño deliberada por dos razones:
1.  **Sincronización:** Necesito una base temporal común para cruzar datos de variables que vienen a frecuencias dispares (RPM vs Posición).
2.  **Suavizado Implícito:** La interpolación lineal actúa como un filtro paso bajo rudimentario.
Para mitigar el error en las derivadas (falsos picos de aceleración), no derivo directamente los datos interpolados crudos. Aplico el filtro **Savitzky-Golay** *después* del resampling y *antes* de la derivación. Este filtro utiliza un ajuste polinómico local que es robusto frente a los artefactos generados por la interpolación lineal, permitiendo estimar la tendencia dinámica real del vehículo (la 'física') separándola del ruido de muestreo."

### Pregunta 2: Veracidad y Ruido de Datos
**Tribunal:** "Sus gráficos muestran picos de 5G en frenada. ¿Cómo valida que esos datos son reales y no ruido numérico, dado que usa datos públicos y no telemetría propietaria del equipo?"

**Respuesta Senior:**
"La validación absoluta es imposible sin acceso a la telemetría interna del equipo (Atlas/McLaren). Sin embargo, realizo una validación **por rango físico y consistencia**.
1.  **Rango Físico:** Los monoplazas de F1 tienen límites físicos conocidos (~5-6G en frenada, ~1.2G en tracción). Si mi cálculo de derivadas arrojara 15G, sabría que es ruido. Mis resultados se mantienen dentro de la envolvente de rendimiento conocida de un F1 actual.
2.  **Consistencia Espacial:** Comparo vueltas consecutivas. Si un pico de 'G' aparece sistemáticamente en la misma coordenada de la pista (ej. curva 1) vuelta tras vuelta, es una característica de la pista/conducción. Si fuera ruido aleatorio, aparecería estocásticamente en diferentes puntos. Mis mapas de calor de Jerk muestran patrones consistentes en los vértices de curva, lo que valida fenomenológicamente el algoritmo."

### Pregunta 3: Complejidad y Rendimiento
**Tribunal:** "Su cálculo de 'Ghost Car' requiere alinear dos series temporales distintas. Si tuviera que procesar en tiempo real las 20 coches durante la carrera, ¿su implementación actual basada en Pandas aguantaría? ¿Qué cambiaría?"

**Respuesta Senior:**
"Mi implementación actual en Pandas es **Batch**, optimizada para análisis post-carrera. Funciona íntegramente en memoria (RAM). Para 20 coches en tiempo real, el cuello de botella sería la re-interpolación constante de DataFrames crecientes.
Para escalar a tiempo real (Streaming):
1.  Cambiaría la estructura de datos: Dejaría de usar DataFrames monolíticos para usar **Ventanas Deslizantes** o Buffers Circulares (ej. últimas 2 vueltas).
2.  Optimización Algorítmica: La interpolación lineal (`np.interp`) es $O(N)$. Podría pre-calcular la malla espacial y usar *Look-up Tables* para reducir el coste de CPU.
3.  **Infraestructura:** Pasaría de un script Python secuencial a un motor de procesamiento de streams como **Apache Flink** o **Spark Streaming**, donde cada coche es un flujo de eventos independiente procesado en paralelo."

### Pregunta 4: MLOps y Despliegue (BentoML)
**Tribunal:** "Ha implementado una API con BentoML. ¿Por qué BentoML y no simplemente Flask o FastAPI? ¿Y qué sentido tiene permitir reentrenamiento desde la UI?"

**Respuesta Señor:**
"La elección de **BentoML** responde a la necesidad de estandarizar el ciclo de vida del modelo (MLOps) más allá de un simple servidor web:
1.  **Empaquetado Unificado (Bento):** BentoML crea un contenedor autosuficiente que incluye no solo el código de la API, sino también el modelo serializado, las dependencias exactas de Python y la configuración del entorno. Esto elimina el problema de 'funciona en mi máquina' al pasar a producción (Docker/Kubernetes).
2.  **Optimización de Inferencia:** A diferencia de un Flask plano, BentoML gestiona automáticamente colas de peticiones y *micro-batching* (Adaptive Batching), optimizando el throughput si recibiéramos ráfagas de telemetría de 20 coches a la vez.

Sobre el **Reentrenamiento Interactivo**:
Implementa el concepto de **'Human-in-the-loop'**. Los modelos de F1 degradan rápido porque las condiciones de pista y coche cambian cada fin de semana (Drift). Permitir que el ingeniero ajuste los hiperparámetros y reentrene el modelo *in-situ* desde el Dashboard (sin tocar código) acelera la adaptación del modelo a las nuevas condiciones del asfalto o actualizaciones aerodinámicas del coche, democratizando el uso de la IA para los ingenieros de pista que no son expertos en Python."

---

## ANEXO: Actualización MLOps (Implementación Final)

### 5. Despliegue y Servicio de Inferencia (Nuevo Módulo)
Se ha integrado **BentoML** para cumplir con el requisito de "Despliegue de API Real-time".
*   **Servicio (`service.py`):** Define una clase `F1LaptimeService` decorada con `@bentoml.service`. Implementa validación estricta de tipos con **Pydantic** (`LapFeaturesInput`) para asegurar que la API rechaza peticiones mal formadas antes de llegar al modelo.
*   **Robustez:** El servicio incluye lógica de *fallback* e inyección de valores por defecto (ej. Driver ID, Año) para garantizar que el preprocesador `scikit-learn` recibe siempre la estructura de columnas exacta con la que fue entrenado, previniendo errores en tiempo de ejecución (`ValueErrors`) típicos en producción.
*   **Interoperabilidad:** El Dashboard de Streamlit (Pestaña 5 "Lab de IA") actúa ahora como cliente HTTP, consumiendo esta API. Esto demuestra una arquitectura desacoplada: el Dashboard (Frontend) podría estar en una tablet en el muro de boxes, mientras que el Modelo (Backend BentoML) corre en un servidor potente en la fábrica, comunicándose vía REST.
