# Análisis Integral y Defensa de Proyecto: F1-Analytics

Este documento contiene la auditoría técnica, justificación teórica, documentación académica y preparación para la defensa del proyecto "Análisis del Rendimiento en la Fórmula 1 mediante Datos de Telemetría".

---

## CAPÍTULO 1: ARQUITECTURA DE DATOS Y FLUJO ETL

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

### 2. Justificación de Escalabilidad (Entorno Producción)
Nuestra implementación simula un **Feature Store** básico.
*   **Desacoplamiento:** Al guardar los CSVs intermedios, separamos el coste computacional "caro" (descarga API, resampling 10Hz, derivadas) de la latencia "barata" de lectura en el dashboard. En producción real, estos CSVs serían tablas en BigQuery o archivos en S3/Parquet.
*   **Paralelismo Implícito:** El diseño por "Sesión/Evento" permite que futuras iteraciones procesen múltiples Grandes Premios en paralelo (Airflow/Dagster) sin conflictos de estado compartido.

---

## CAPÍTULO 2: ASUNCIONES TEÓRICAS Y LIMITACIONES DEL MODELO

Esta sección detalla las simplificaciones adoptadas en nuestro gemelo digital, cruciales para entender el alcance y validez de las conclusiones.

### 2.1 Asunciones en la Física Vehicular
1.  **Modelo de Masa Puntual y Plano Horizontal:**
    *   **Asunción:** Tratamos el coche como un punto de masa constante moviéndose en un plano 2D perfecto (`z=0`).
    *   **Justificación:** FastF1 no proporciona datos fiables de elevación (eje Z con ruido excesivo) ni de la masa del combustible en tiempo real.
    *   **Impacto/Gap:** Ignoramos el efecto de la gravedad en subidas/bajadas (fuerza normal variable) y la reducción de peso por quema de combustible (~100kg a 0kg).
    *   **Mitigación:** El modelo de ML (Random Forest) aprende implícitamente la reducción de peso a través de la variable `LapNumber`, que correlaciona linealmente con el consumo.

2.  **Validez de la Interpolación (Ghost Car):**
    *   **Asunción:** La "Distancia de Vuelta" es una métrica absoluta comparable entre pilotos.
    *   **Justificación:** Para comparar telemetría espacialmente, necesitamos un eje común.
    *   **Limitación:** Pilotos con líneas de carrera diferentes (uno abierto, uno cerrado) recorren distancias reales distintas. Al proyectar todo a un eje de "Distancia de Vuelta Estandarizada" (Centerline), introducimos un pequeño error de alineación espacial.
    *   **Defensa:** Este error es despreciable (< 5 metros) comparado con la longitud del circuito (5000m) y es el estándar de la industria para análisis comparativo rápido.

### 2.2 Limitaciones de los Datos de Entrada
*   **Fuente Pública (API):** Trabajamos con datos de GPS de baja precisión (~1 metro, 5-10Hz) y telemetría pública. No tenemos acceso a sensores internos como presión de freno (Brake Pressure) o ángulo de volante real (Steering Angle) con alta fidelidad.
*   **Variables Ocultas:** No conocemos el mapa motor (Engine Mode), carga de batería (ERS deploy) ni el setup aerodinámico (Ala).
*   **Justificación del Modelo:** Por esto usamos variables "Proxy" (derivadas): Jerk como proxy de agresividad, y `TireEnergy` como proxy de degradación.

### 2.3 Alcance del Dataset (Prototipo)
*   **Selección de "Vuelta Rápida":**
    *   **Limitación:** El pipeline actual procesa únicamente la **Vuelta Más Rápida** de cada piloto por sesión, descartando vueltas de entrada/salida y *long runs*.
    *   **Impacto:** El dataset de entrenamiento es pequeño (~20 muestras por carrera) y está sesgado hacia el rendimiento máximo (Qualifying pace).
    *   **Justificación:** Decisión de diseño para reducir costes computacionales y de almacenamiento en esta fase de prototipado. El sistema demuestra la capacidad de procesar telemetría compleja, escalar a todas las vueltas es trivial (bucle `for`) pero costoso en tiempo de ejecución.
*   **Variables Climáticas:**
    *   **Limitación:** Aunque descargamos datos climáticos (`weather.csv`), no los integramos en la predicción final.
    *   **Justificación:** La variabilidad climática intra-sesión en nuestros datos de prueba fue baja. En un modelo de producción real, cruzaríamos `SessionTime` con la temperatura de pista para ajustar las predicciones de degradación.

---

## CAPÍTULO 3: FUNDAMENTOS FÍSICOS Y MATEMÁTICOS

### 1. Manipulación de Datos
*   **Velocidad y Posición:** Extraídas crudas de API (`Speed`, `X`, `Y`) en `module1`.
*   **Tiempo Relativo:** Calculado en `resample_telemetry_by_time` como $t - t_{inicio}$.
*   **Aceleración (G-Force):** No viene directa del GPS con precisión. Nuestro código la deriva en `module2_signals.py`:
    *   $\vec{v} = (\dot{x}, \dot{y})$
    *   $\vec{a} = (\ddot{x}, \ddot{y})$

### 2. Matemática Aplicada

#### A. Derivación Numérica y Suavizado (Calculus)
Las señales de sensores tienen ruido de alta frecuencia. Derivar ruido amplifica el error.
*   **Solución:** Utilizamos **Filtro Savitzky-Golay** en `module2_signals.py` (`_savgol`, L37).
    *   Matemáticamente, ajusta un polinomio local de grado `poly` (2) en una ventana `window` (11 muestras) usando mínimos cuadrados.
    *   Esto preserva mejor los picos de la señal (frenadas bruscas) que una media móvil simple.
*   **Cinemática:**
    *   Calculamos gradientes numéricos (`np.gradient`) para obtener aceleración:
        $$ a_x[n] = \frac{v_x[n+1] - v_x[n-1]}{2\Delta t} $$

#### B. Descomposición de Vectores (Álgebra Lineal)
En `_compute_dynamics_group` (L80-82), proyectamos la aceleración cartesiana ($a_x, a_y$) al marco de referencia del coche (Frenett-Serret frame local):
*   **Longitudinal ($a_t$):** Aceleración/Frenada. Producto escalar con el vector velocidad unitario.
    $$ a_{long} = \vec{a} \cdot \hat{v} = \frac{a_x v_x + a_y v_y}{|\vec{v}|} $$
*   **Lateral ($a_n$):** Fuerza centrífuga en curva. Producto cruz 2D.
    $$ a_{lat} = \frac{a_x v_y - a_y v_x}{|\vec{v}|} $$

#### C. Diagrama G-G y Círculo de Fricción
El gráfico "Círculo de Fricción" en `app.py` representa la envolvente de rendimiento del neumático.
*   Físicamente: $\sqrt{a_{lat}^2 + a_{long}^2} \leq \mu g$.
*   Nuestro código normaliza por gravedad ($9.81 m/s^2$) para visualizar Gs. Puntos fuera del círculo teórico indican peralte, aerodinámica (Downforce) o error de sensor.

#### D. Energía de Neumático (Proxy)
Calculamos `TireEnergyProxy` integrando la potencia de las fuerzas de fricción:
$$ E \approx \int (|a_{lat}| + |a_{long}|) \cdot v \cdot dt $$
Esto modela el trabajo realizado por los neumáticos, correlacionando directamente con la temperatura y la degradación térmica (Energy Management).

---

## CAPÍTULO 4: METODOLOGÍA DE INGENIERÍA

### Apartado: Metodología e Implementación

#### 4.1 Arquitectura del Sistema y Procesamiento de Señales
El sistema se ha diseñado bajo los principios de **Industria 4.0**, implementando un gemelo digital (*Digital Twin*) del comportamiento dinámico del monoplaza. El núcleo de procesamiento consta de un pipeline dividido en cuatro etapas modulares:

1.  **Ingesta y Sincronización Espacio-Temporal:** Se utiliza la librería `FastF1` como interfaz de extracción. Debido a la naturaleza heterogénea de los sensores (GPS a baja frecuencia vs. ECU), se aplica un algoritmo de *resampling* a 10 Hz para homogeneizar el dominio temporal. Adicionalmente, se implementa una interpolación espacial lineal (`module1_ingestion.py`), remapeando las series temporales a un dominio de distancia fija ($\Delta d = 1m$). Esto permite la comparación directa de telemetría entre vueltas ("Ghost Car Analysis") independientemente de las diferencias en velocidad de paso por curva.

2.  **Modelado Dinámico y Filtrado de Ruido:** Para reconstruir las fuerzas G (no disponibles con precisión en la fuente pública), se aplica cálculo diferencial numérico sobre los vectores de posición y velocidad. Para mitigar el ruido de cuantización inherente al GPS, se implementa un filtro **Savitzky-Golay** (polinomio de orden 2, ventana de 11 muestras), que suaviza la señal preservando los momentos de alta frecuencia característicos de las frenadas en F1. Las aceleraciones resultantes se descomponen vectorialmente en componentes longitudinales (tracción/frenado) y laterales (viraje) mediante proyección sobre el vector velocidad instantáneo.

3.  **Caracterización del Estilo de Conducción:** Se utiliza Análisis de Componentes Principales (**PCA**) sobre el espacio de características agregadas por vuelta (Jerk medio, agresividad en freno, Gs máximos) para reducir la dimensionalidad y detectar patrones latentes (suavidad vs. agresividad) que diferencian a los pilotos.

4.  **Predicción de Rendimiento (ML):** Se implementa un modelo de regresión (Gradient Boosting/Random Forest) validado mediante *K-Fold Cross Validation* para predecir tiempos de vuelta basándose en parámetros físicos y de gestión de neumáticos, logrando cuantificar el impacto de cada variable mediante valores SHAP.

#### 4.2 Justificación Tecnológica
*   **Pandas & NumPy:** Elegidos por su capacidad de vectorización (SIMD), permitiendo realizar cálculos cinemáticos sobre millones de puntos de datos de telemetría en milisegundos, frente a bucles iterativos convencionales.
*   **FastF1:** Proporciona acceso democratizado a datos de nivel industrial (Timing y Telemetría básica), sirviendo como capa de abstracción sobre la API oficial de la F1.
*   **Streamlit:** Facilita la creación rápida de dashboards interactivos, vital para la visualización exploratoria de datos complejos sin el overhead de desarrollo frontend tradicional.

---

## ANEXO: Actualización MLOps (Implementación Final)

### 5. Despliegue y Servicio de Inferencia (Nuevo Módulo)
Se ha integrado **BentoML** para cumplir con el requisito de "Despliegue de API Real-time".
*   **Servicio (`service.py`):** Define una clase `F1LaptimeService` decorada con `@bentoml.service`. Implementa validación estricta de tipos con **Pydantic** (`LapFeaturesInput`) para asegurar que la API rechaza peticiones mal formadas antes de llegar al modelo.
*   **Robustez:** El servicio incluye lógica de *fallback* e inyección de valores por defecto (ej. Driver ID, Año) para garantizar que el preprocesador `scikit-learn` recibe siempre la estructura de columnas exacta con la que fue entrenado, previniendo errores en tiempo de ejecución (`ValueErrors`) típicos en producción.

*   **Ejecucion:** Para ejecutar el bentoml, ejecuta '''uv run feature_extraction/module4_modeling.py''' si estás usando linux, para cargar el modelo bentoml en la lista de modelos. Después, ejecuta '''bentoml serve service.py:F1LaptimeService --reload'''
---

## ANEXO 2: FUNDAMENTACIÓN TEÓRICA Y MATEMÁTICA DETALLADA

Esta sección profundiza en la lógica interna de los algoritmos de procesamiento de señal (Módulos 1 y 2), justificada para la defensa técnica.

### 1. Sincronización e Interpolación (Módulo 1)

#### 1.1 Sincronización Temporal (El "Reloj Maestro" a 10 Hz)
**El Problema de la Asincronía:**
Un coche de F1 es un sistema distribuido de sensores. El ECU reporta RPM a ~100Hz, mientras que el GPS público reporta posición a ~5Hz. Si intentáramos unir estos datos en una tabla simple, tendríamos miles de huecos (NaNs) porque los sensores no "hablan" al mismo tiempo.

**Nuestra Solución: Resampling a 10Hz**
Creamos un "Reloj Maestro" virtual que hace *tic* cada 0.1 segundos.
*   **¿Por qué 10 Hz?**
    *   **Teorema de Nyquist-Shannon:** La fuente de datos más lenta y crítica es el GPS (~5 Hz). Para reconstruirla sin aliasing, necesitamos muestrear al menos al doble de su frecuencia (10 Hz).
    *   **Eficiencia vs Precisión:** Subir a 100 Hz (como la ECU) sería inútil para la posición, pues estaríamos interpolando (inventando) 19 de cada 20 puntos, multiplicando el tamaño el dataset por 10 sin ganar información real espacial. 10 Hz es el punto óptimo (Sweet Spot) entre fidelidad a la fuente GPS y carga computacional.

**Algoritmo:**
1.  **Tiempo Relativo ($t_{rel}$):** $t_{rel} = t_{actual} - t_{inicio\_vuelta}$.
2.  **Grilla Temporal:** Vector monotónico `[0.0, 0.1, 0.2, ...]`.
3.  **Interpolación Lineal:** Trazamos rectas entre puntos conocidos.

#### 1.2 Interpolación Espacial (Dominio $d$)
**Por qué el "Ghost Car" falla por tiempo (Comparación Timestamp vs Distancia):**
*   *Timestamp:* Comparar a Hamilton en $t=50s$ contra Verstappen en $t=50s$ es matemáticamente válido pero físicamente absurdo si Hamilton va 2 segundos más rápido. Estaríamos comparando una recta (Hamilton) con una curva (Verstappen).
*   *Distancia:* Comparar a Hamilton en el metro 3000 contra Verstappen en el metro 3000 asegura que ambos enfrentan la **misma geometría de pista** (misma curva 8).

**La Malla de Distancia Fija (Mesh):**
*   **Definición:** Un vector estático de distancia `[0m, 1m, 2m, ..., L_pista]`.
*   **¿Por qué 1 metro de resolución?**
    *   La precisión del GPS civil/público oscila entre 1-3 metros.
    *   *Hipótesis de Falsa Precisión:* Si bajáramos la malla a **0.5m o 0.1m**, estaríamos modelando "ruido de GPS" como si fuera trayectoria real. 1 metro es el límite de confianza de la fuente de datos.
*   **Remapeo de Variable ($t \to d$):**
    *   Invertimos la relación cinemática. En lugar de preguntar "¿Dónde estoy en el segundo 5?", preguntamos "¿A qué velocidad iba en el metro 500?". Usamos `np.interp` para proyectar todas las señales sobre esta cinta métrica universal.
    *   **Manejo de Duplicados (`np.unique`):**
        *   *Problema:* El GPS a veces reporta la misma distancia en dos timestamps consecutivos (coche parado o error de redondeo). Esto rompe la matemática de interpolación (falla la biyectividad).
        *   *Solución:* Usamos `np.unique` para filtrar duplicados y garantizar que la distancia sea estrictamente monotónica creciente antes de interpolar.
    *   **Analogía Visual (La Regla Gigante):**
        *   Imagina una regla gigante en la pista con marcas exactas cada 1 metro: `[0m, 1m, 2m, ...]`.
        *   **Input (Datos Reales):** Tenemos lecturas irregulares (ej. el coche pasó por 2.5m a 100km/h).
        *   **Output (Ghost Car):** Descartamos la distancia original y calculamos: "Si pasó por 2.5m a 100km/h, ¿a qué velocidad debió pasar por la marca exacta de **2.0m** y **3.0m**?". Esto normaliza el eje X para todos los pilotos.

---

### 2. Procesamiento de Señales (Módulo 2)

#### 2.1 Filtro Savitzky-Golay (La solución al "Ruido Derivativo")
**El Problema Matemático:**
La diferenciación numérica (calcular aceleración restando velocidades) actúa como un **filtro pasa-altos**.
*   Si la señal de velocidad tiene un pequeño "temblor" (jitter) de 1 km/h por error de GPS, al dividirlo por un $\Delta t$ muy pequeño (0.1s), el error se multiplica por 10 en la aceleración.
*   Además, nuestra interpolación lineal (Capítulo 1) crea "picos" no diferenciables (esquinas) en los puntos de unión de las rectas. Derivar una esquina genera una aceleración infinita (Impulso de Dirac teórico).

**La Solución Savitzky-Golay:**
Ajustamos un polinomio local (parábola) sobre una ventana de 11 puntos ($1.1s$).
*   **¿Qué resuelve?**
    1.  **Suaviza el Jitter:** Promedia el ruido aleatorio del sensor.
    2.  **Restaura la Diferenciabilidad:** Convierte las "esquinas" de la interpolación lineal en curvas suaves (Polinomio $C^1$ continuo), permitiendo calcular derivadas físicas realistas.
    3.  **Preserva Picos:** A diferencia de una media móvil (que aplana todo), el polinomio respeta los picos reales de frenada (transiciones bruscas pero físicas).

#### 2.2 Dinámica Vehicular y Vectores
*   **Proyección de Vectores (Global a Local):**
    *   Transformamos del Global Frame ($x, y$ GPS) al Local/Frenet Frame ($a_t, a_n$). Al piloto no le importa si acelera "al Norte", sino si acelera "hacia adelante".
    *   **Aceleración Tangencial ($a_t$ - Longitudinal):** Responsable del cambio de rapidez (Frenado/Tracción). Se calcula como la proyección escalar (Producto Punto) de la aceleración sobre la velocidad:
        $$ a_t = \frac{\vec{a} \cdot \vec{v}}{||\vec{v}||} = \frac{a_x v_x + a_y v_y}{\sqrt{v_x^2 + v_y^2}} $$
    *   **Aceleración Normal ($a_n$ - Lateral):** Responsable del cambio de dirección (Viraje). Se calcula usando el "Producto Cruz 2D" (determinante):
        $$ a_n = \frac{\vec{a} \times_{2D} \vec{v}}{||\vec{v}||} = \frac{a_x v_y - a_y v_x}{\sqrt{v_x^2 + v_y^2}} $$
    *   *Nota de Implementación:* El signo de $a_n$ nos indica la dirección del giro (Izquierda vs Derecha), vital para analizar el comportamiento en curvas.

#### 2.2 bis. Energy Proxy (Definición Física)
He definido el **Índice de Energía del Neumático** basándome en el concepto físico de Trabajo y Potencia.
*   **Potencia Instantánea ($P$):** Fuerza $\times$ Velocidad.
    *   $F \propto$ Masa $\times$ Aceleración G combinada ($|G_{lat}| + |G_{long}|$).
*   **Energía Total ($J$):** Integral de la Potencia en el tiempo.
    $$ EnergyProxy = \int_{0}^{T} (|G_{lat}(t)| + |G_{long}(t)|) \cdot v(t) \cdot dt $$
*   **Significado Físico:** Representa la cantidad total de "sufrimiento" (Joules mecánicos) que la carcasa del neumático ha tenido que gestionar. Correlaciona directamente con la temperatura generada y el desgaste abrasivo.

#### 2.3 `enrich_telemetry_time`
Es la función orquestadora que:
1.  Verifica que existan las columnas base (`Speed`, `X`, `Y`).
2.  Agrupa por `Driver` y `LapNumber`. **¿Por qué?** Para no calcular derivadas entre el final de la vuelta 1 y el inicio de la vuelta 2 (habría un salto discontinuo enorme en posición que parecería una velocidad infinita).
3.  Aplica el cálculo físico (`_compute_dynamics_group`) a cada grupo de forma aislada.
---

#### 2.4 Conceptos Físicos Adicionales

**A. Ruido de Alta Frecuencia**
Cuando decimos "ruido de alta frecuencia" en este contexto, nos referimos a variaciones rápidas y aleatorias en la señal que no corresponden a movimientos reales del coche (el coche tiene inercia, no puede teletransportarse 10cm a la izquierda en 0.01s).
*   **Fuentes**: Vibración del motor (15,000 RPM = 250Hz) afectando al acelerómetro, error de precisión del GPS ("jitter"), interferencia electrónica.
*   **Por qué eliminarlo**: Si derivamos ruido rápido, obtenemos valores de aceleración infinitos (la derivada de un pico instantáneo es enorme). El filtro Savitzky-Golay lo suaviza.

**B. Marco de Referencia de Frenet-Serret (Local Frame)**
Es un sistema de coordenadas móvil que viaja con el coche.
*   **Vectores**:
    *   **Tangente ($\hat{T}$)**: Apunta hacia adelante, en la dirección de la velocidad.
    *   **Normal ($\hat{N}$)**: Apunta hacia el centro de la curva (perpendicular a $\hat{T}$).
*   **Por qué lo usamos**: El GPS nos da coordenadas "Globales" (Latitud, Longitud) o Cartesianas fijas ($X, Y$ respecto al centro del mapa). Al piloto no le sirve saber "estoy acelerando hacia el Norte". Le sirve saber "estoy frenando ($-a_t$)" o "estoy girando a la derecha ($a_n$)". Proyectar la aceleración global en el marco Frenet-Serret nos da esa visión "desde el asiento del piloto".

**C. Tiempo Relativo (`RelativeTime`)**
```python
telemetry["RelativeTime"] = telemetry["SessionTime"] - telemetry["SessionTime"].iloc[0]
```
*   **Explicación**: El `SessionTime` es el reloj oficial de la sesión (ej. "14:05:32 PM"). Si comparamos la vuelta 5 de Max con la vuelta 20 de Lewis, sus `SessionTimes` son totalmente distintos. Al restar el tiempo del *inicio* de esa vuelta específica (`iloc[0]`), ponemos el cronómetro a cero ($t=0$). Así podemos superponer graficas: "En el segundo 5.3 de *su respectiva vuelta*, ambos estaban en la curva 1".

---

### 3. Caracterización de Estilo (Módulo 3: PCA Normalizado)

**Objetivo**: Saber "cómo conduce" un piloto, independientemente del coche o la pista.

#### 3.1 Normalización por Evento
**Problema**: Mónaco es lento (avg speed 150 km/h) y Silverstone es rápido (240 km/h). Si metemos estos datos crudos al PCA, la Componente Principal 1 sería simplemente "Circuito Rápido vs Lento", no "Piloto Agresivo vs Suave".
**Solución**: Normalización Z-Score **por Evento**.
$$ x'_{i} = \frac{x_{i} - \mu_{carrera}}{\sigma_{carrera}} $$
*   Para cada métrica (ej. Agresividad de Freno), calculamos la media y desviación estándar **de ese Gran Premio**.
*   Si Max frena con agresividad 8 en una pista donde la media es 5, su score es +3 sigmas.
*   Si en otra pista frena con 8 pero la media es 8, su score es 0.
*   **Resultado**: Eliminamos el "Efecto Pista". Solo queda cuánto se desvía el piloto del promedio de la parrilla ese día.

#### 3.2 Interpretación de Componentes Principales
Para interpretar el PCA con rigor científico y no simplemente "adivinar", inspeccionamos la matriz de **Eigenvectors (Loadings)**:
1.  **PC1 (Correlación con Velocidad):** Vemos que las cargas más altas positivas son `Avg_Speed` y negativas `LapTime`. Esto define matemáticamente al PC1 como el eje de **"Ritmo Puro"**.
2.  **PC2 (Correlación con Derivadas):** Las cargas dominantes son `MeanAbs_Jerk_Long` y `Brake_Aggression`. Esto no es subjetivo; el algoritmo nos dice que este componente varía cuando varían las fuerzas bruscas. Por tanto, es el eje de **"Agresividad Longitudinal"**.
3.  **Evidencia:** Al plotear a los pilotos, Verstappen (rápido y agresivo) puntúa alto en ambos, mientras que pilotos conservadores puntúan bajo en PC2.

---

---

### 4. Predicción de Tiempos (Módulo 4: Modelado)

**Objetivo**: Predecir el `LapTimeSeconds` basándose en la telemetría agregada y el estilo del piloto.

#### 4.1 Arquitectura del Pipeline (`sklearn.pipeline`)
Hemos diseñado un pipeline robusto a prueba de fallos en producción:
1.  **Preprocesamiento Universal (`ColumnTransformer`)**:
    *   **Numéricos (`StandardScaler`)**: Normaliza inputs como `TyreLife` o `FuelLoad` para que tengan media 0 y varianza 1. Vital para modelos lineales (Lasso) y ayuda a la convergencia en redes neuronales (si se usaran).
    *   **Categóricos (`OneHotEncoder`)**: Transforma `Compound` (SOFT, MEDIUM, HARD) y `Team` en vectores binarios. Usamos `handle_unknown='ignore'` para que si mañana aparece un compuesto nuevo (ej. "HYPERSOFT"), el modelo no rompa en producción (simplemente lo ignora).

#### 4.2 Validación Cruzada (Cross-Validation 5-Fold)
**El Concepto Teórico:**
En lugar de hacer un solo examen final (Train/Test split 80/20), sometemos al modelo a **5 exámenes diferentes** rotando los datos.
1.  Dividimos el dataset en 5 partes iguales (Folds).
2.  Entrenamos con 4 partes y testeamos con la 1 restante.
3.  Repetimos el proceso 5 veces, cambiando la parte de test cada vez.
4.  El resultado final es el **promedio** de los 5 errores.

**¿Por qué lo usamos?**
*   **Robustez Estadística:** Evita que tengamos "suerte" eligiendo un test set fácil. Si el modelo funciona bien en los 5 folds, es estable.
*   **Detección de Overfitting:** Si el modelo memoriza datos, fallará estrepitosamente en al menos uno de los folds.

**Aplicación en el Proyecto (`KFold(shuffle=True)`):**
En `module4_modeling.py`, usamos `shuffle=True`.
*   *Nota Técnica:* Aunque la F1 es temporal, aquí tratamos cada vuelta como un **"Experimento Físico Independiente"**.
*   Queremos que el modelo aprenda: *"Si Neumático=Viejo y Gasolina=Baja -> Tiempo=X"*.
*   Al barajar (shuffle), validamos que el modelo aprende la **física del coche** (la relación variables-tiempo) y no simplemente la secuencia cronológica de las vueltas.

#### 4.3 Selección de Modelos (Benchmark)
Comparamos múltiples familias de algoritmos:
1.  **Lasso (Baseline Lineal)**: Nos dice cuánto podemos explicar con relaciones simples. Si $R^2$ es bajo, confirma que el problema es no-lineal.
2.  **Random Forest / XGBoost (No-Lineales)**:
    *   **Por qué ganan**: Capturan interacciones complejas automáticamente. Ejemplo: Un neumático blando (`SOFT`) degrada rápido (`TyreLife` alto = tiempo lento), pero un neumático duro (`HARD`) es más constante. Un modelo lineal sumaría `Beta_TyreLife` + `Beta_Compound`, pero el árbol puede hacer *splits*: "SI Compound=SOFT Y TyreLife>10 ENTONCES Lento".
    *   **Robustez**: Son menos sensibles a outliers que las redes neuronales simples.

#### 4.3 Métrica de Éxito: MAE (Mean Absolute Error)
Elegimos MAE sobre MSE/RMSE por **interpretabilidad**.
*   Decirle a un ingeniero de carrera: "El error cuadrático medio es 0.04" no significa nada intuitivo.
*   Decirle: "El modelo se equivoca en promedio **±0.2 segundos** por vuelta" es información accionable.

#### 4.4 MLOps: Integración con BentoML
Al finalizar `run_modeling`, no solo guardamos un archivo `.pkl`.
1.  **Empaquetado**: Usamos `bentoml.sklearn.save_model`. Esto guarda el modelo + versión de scikit-learn + metadatos (MAE, R2) en un "Bento" inmutable.
2.  **Trazabilidad**: Podemos saber exactamente qué versión del código entrenó el modelo que está corriendo hoy en el GP de Bahrein.

#### 4.5 Exploración Visual (Notebook Insights)
Como se observa en `notebooks/tests.ipynb`, validamos el modelo gráficamente:
*   **Actual vs Predicted**: Buscamos una línea perfecta $y=x$. Desviaciones sistemáticas (nube curva) indicarían que nos falta una feature no-lineal (ej. carga de combustible cuadrática).
*   **Residuos**: Verificamos que los errores sean aleatorios y centrados en cero. Si vemos patrones (ej. siempre erramos en la vuelta de salida de boxes), sabríamos dónde mejorar la limpieza de datos (Módulo 1).

---

## CAPÍTULO 6: CONCLUSIONES Y RESULTADOS FINALES

El proyecto **F1-Analytics** ha evolucionado de un script de análisis básico a una plataforma de **MLOps "End-to-End"** que integra Ingeniería de Datos, Física Vehicular y Machine Learning en tiempo real.

### 5.1 Resultados del Modelo (Quantitative)
Tras la evaluación de múltiples algoritmos, **Random Forest** ha sido seleccionado como el modelo campeón para producción.
*   **MAE (Mean Absolute Error)**: **0.86 segundos**. Esto significa que nuestro modelo predice el tiempo de vuelta con una precisión de menos de 1 segundo en promedio, lo cual es sobresaliente considerando las variables climáticas y de tráfico no modeladas.
*   **R² (Varianza Explicada)**: **0.97**. El 97% de la variabilidad en los tiempos de vuelta es explicada por nuestras features físicas (`Energy_Index`, `G-Forces`, `TyreLife`).
*   **Comparativa**:
    *   Random Forest (MAE 0.86) > Lasso Lineal (MAE 1.03) -> Confirma la naturaleza no lineal del problema (degradación neumáticos).
    *   Random Forest > Gradient Boosting (MAE 1.86) -> RF demostró ser más robusto a outliers sin necesidad de *hyperparameter tuning* excesivo.

### 5.2 Aprendizajes Clave
1.  **"Garbage In, Garbage Out" es real**: El 70% del esfuerzo se invirtió en el **Módulo 1** (sincronización de telemetría a 10Hz e interpolación espacial). Sin una base de datos limpia y alineada espacialmente, ningún modelo predeciría nada útil.
2.  **La Física importa**: Las features derivadas de principios físicos (Energía de neumático, Fuerzas G proyectadas en Frenet-Serret) resultaron ser los predictores más potentes (PC1 y PC3), superando a métricas crudas como "RPM" o "Speed".
3.  **MLOps desbloquea valor**: La transición de notebooks estáticos a una API **BentoML** consumida por **Streamlit** transformó un "experimento científico" en un "producto de software" utilizable por ingenieros en pista.

---

## ANEXO 4: FLUJO DE DATOS Y ARTEFACTOS (DATA LINEAGE)

Este diagrama detalla cómo se transforman los datos desde la API hasta el modelo final, garantizando trazabilidad total.

### 1. Ingesta y Sincronización (Módulo 1)
**Script**: `module1_ingestion.py`
*   **Fuente**: API FastF1 (Cacheada en `.fastf1-cache/`)
*   **Output por Sesión** (ej. `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/`):
    *   `telemetry_time_10hz.csv`: Telemetría cruda resampleada a 10Hz (eje temporal común).
    *   `telemetry_distance_aligned.csv`: Telemetría interpolada por metros (eje espacial).
    *   `laps_metadata.csv`: Tiempos de vuelta oficiales y compuestos de neumáticos.
    *   `weather.csv`: Condiciones climáticas.

### 2. Enriquecimiento de Señales (Módulo 2)
**Script**: `module2_signals.py`
*   **Input**: Lee los CSVs del Módulo 1 carpeta por carpeta.
*   **Proceso**: Calcula derivadas (Jerk, Fuerzas G) y aplica suavizado Savitzky-Golay.
*   **Output por Sesión** (en la misma carpeta):
    *   `telemetry_time_10hz_enriched.csv`: Incluye nuevas columnas como `Speed_mps`, `AX_long`, `AY_lat`, `TireEnergyProxy`.
    *   `lap_features_module2.csv`: **Dataset Agregado**. Una fila por vuelta con métricas resumen (ej. `MeanAbs_Jerk_Long`, `Energy_Index`).

### 3. Consolidación y Estilo (Módulo 3)
**Script**: `merge_lap_features.py` y `module3_pca_global_normalized.py`
*   **Paso A (Merge)**: `merge_lap_features.py` itera todas las carpetas de sesiones, lee `lap_features_module2.csv` y concatena todo en:
    *   **Output**: `data/module1_ingestion/all_lap_features.csv` (Dataset Maestro).
*   **Paso B (PCA)**: `module3_pca_global_normalized.py` lee el Dataset Maestro.
    *   **Proceso**: Normaliza métricas por evento (Z-Score por carrera) y aplica PCA.
    *   **Output**:
        *   `pca_scores_global_norm.csv`: Las 3 Componentes Principales (Estilo) para cada vuelta.
        *   `pca_model_global_norm.json`: Pesos del modelo PCA para transformar nuevos datos.

### 4. Modelado Predictivo (Módulo 4)
**Script**: `module4_modeling.py`
*   **Input**: Une `all_lap_features.csv` (Features físicas) con `pca_scores_global_norm.csv` (Features latentes).
*   **Proceso**: Entrena modelos (Random Forest, XGBoost) con Cross-Validation.
*   **Output**:
    *   `best_model_module4.pkl`: Pipeline scikit-learn entrenado (Scaler + Encoder + Modelo).
    *   `model_metrics_module4.json`: Resultados de MAE/R2.
    *   `val_predictions_module4.csv`: Predicciones vs Realidad para validación.
*   **BentoML**: Guarda el modelo en el store de BentoML (`f1_laptime_predictor:latest`) para ser servido por la API.

---


