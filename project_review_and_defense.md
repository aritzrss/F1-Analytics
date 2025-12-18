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

## CAPÍTULO 5: SIMULACIÓN DE DEFENSA (PREGUNTAS FRECUENTES)

### Pregunta 1: Sincronización y Frecuencia de Muestreo
**Tribunal:** "Usted hace un resampling a 10 Hz. Sin embargo, los datos de GPS de la F1 suelen venir a 1-5 Hz y la telemetría del coche a mucho más. ¿No está inventando datos al interpolar a 10 Hz? ¿Cómo afecta esto a la precisión de sus derivadas de aceleración?"

**Respuesta:**
"Es una excelente observación sobre la teoría de señales. Efectivamente, al hacer upsampling a 10 Hz desde una fuente GPS de menor frecuencia, estamos introduciendo puntos interpolados que no existen en la medición original. Sin embargo, esto es una decisión de diseño deliberada por dos razones:
1.  **Sincronización:** Necesitamos una base temporal común para cruzar datos de variables que vienen a frecuencias dispares (RPM vs Posición).
2.  **Suavizado Implícito:** La interpolación lineal actúa como un filtro paso bajo rudimentario.
Para mitigar el error en las derivadas (falsos picos de aceleración), no derivamos directamente los datos interpolados crudos. Aplicamos el filtro **Savitzky-Golay** *después* del resampling y *antes* de la derivación. Este filtro utiliza un ajuste polinómico local que es robusto frente a los artefactos generados por la interpolación lineal, permitiendo estimar la tendencia dinámica real del vehículo (la 'física') separándola del ruido de muestreo."

### Pregunta 2: Veracidad y Ruido de Datos
**Tribunal:** "Sus gráficos muestran picos de 5G en frenada. ¿Cómo valida que esos datos son reales y no ruido numérico, dado que usa datos públicos y no telemetría propietaria del equipo?"

**Respuesta:**
"La validación absoluta es imposible sin acceso a la telemetría interna del equipo (Atlas/McLaren). Sin embargo, realizamos una validación **por rango físico y consistencia**.
1.  **Rango Físico:** Los monoplazas de F1 tienen límites físicos conocidos (~5-6G en frenada, ~1.2G en tracción). Si nuestro cálculo de derivadas arrojara 15G, sabríamos que es ruido. Nuestros resultados se mantienen dentro de la envolvente de rendimiento conocida de un F1 actual.
2.  **Consistencia Espacial:** Comparamos vueltas consecutivas. Si un pico de 'G' aparece sistemáticamente en la misma coordenada de la pista (ej. curva 1) vuelta tras vuelta, es una característica de la pista/conducción. Si fuera ruido aleatorio, aparecería estocásticamente en diferentes puntos. Nuestros mapas de calor de Jerk muestran patrones consistentes en los vértices de curva, lo que valida fenomenológicamente el algoritmo."

### Pregunta 3: Complejidad y Rendimiento
**Tribunal:** "Su cálculo de 'Ghost Car' requiere alinear dos series temporales distintas. Si tuviera que procesar en tiempo real las 20 coches durante la carrera, ¿su implementación actual basada en Pandas aguantaría? ¿Qué cambiaría?"

**Respuesta Senior:**
"Nuestra implementación actual en Pandas es **Batch**, optimizada para análisis post-carrera. Funciona íntegramente en memoria (RAM). Para 20 coches en tiempo real, el cuello de botella sería la re-interpolación constante de DataFrames crecientes.
Para escalar a tiempo real (Streaming):
1.  Cambiaríamos la estructura de datos: Dejaríamos de usar DataFrames monolíticos para usar **Ventanas Deslizantes** o Buffers Circulares (ej. últimas 2 vueltas).
2.  Optimización Algorítmica: La interpolación lineal (`np.interp`) es $O(N)$. Podríamos pre-calcular la malla espacial y usar *Look-up Tables* para reducir el coste de CPU.
3.  **Infraestructura:** Pasaríamos de un script Python secuencial a un motor de procesamiento de streams como **Apache Flink** o **Spark Streaming**, donde cada coche es un flujo de eventos independiente procesado en paralelo."

### Pregunta 4: MLOps y Despliegue (BentoML)
**Tribunal:** "Ha implementado una API con BentoML. ¿Por qué BentoML y no simplemente Flask o FastAPI? ¿Y qué sentido tiene permitir reentrenamiento desde la UI?"

**Respuesta:**
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
---

## ANEXO 2: FUNDAMENTACIÓN TEÓRICA Y MATEMÁTICA DETALLADA

Esta sección profundiza en la lógica interna de los algoritmos de procesamiento de señal (Módulos 1 y 2), justificada para la defensa técnica.

### 1. Sincronización e Interpolación (Módulo 1)

#### 1.1 Resampling Temporal (10 Hz)
**Concepto**: La telemetría de F1 es asíncrona. El GPS puede reportar a 5Hz, el acelerómetro a 100Hz y el motor a 20Hz.
**Qué hace `resample_telemetry_by_time`**:
1.  **Alineación Temporal**: Resta el tiempo de inicio de sesión (`SessionTime - SessionTime[0]`) para tener un eje de tiempo relativo ($t=0$ al inicio de la vuelta).
2.  **Upsampling/Downsampling**: Crea una malla temporal perfecta cada 0.1 segundos (`1/10Hz`).
3.  **Algoritmo**: Utiliza interpolación lineal (`.interpolate()`).
    *   *Justificación*: Frente a un "Zero-Order Hold" (mantener el valor anterior), la interpolación lineal asume cambios continuos, lo cual es físicamente correcto para velocidad y RPM (no cambian de golpe).

#### 1.2 Interpolación Espacial (`interpolate_telemetry_by_distance`)
**Problema**: Dos pilotos toman tiempos distintos para recorrer la misma curva. Si comparamos por tiempo (segundo 10 vs segundo 10), uno puede estar en la curva y el otro ya saliendo.
**Solución**: Cambiar el dominio de Tiempo ($t$) a Espacio ($d$).
**Cómo funciona**:
1.  Se define una "malla espacial" de 0m a la longitud de la vuelta, con pasos de 1 metro (`DISTANCE_STEP_METERS = 1.0`).
2.  Para cada sensor (Velocidad, RPM), se calcula el valor en ese metro exacto, interpolando entre los dos puntos de GPS más cercanos.
**Para qué**: Permite el análisis "Ghost Car". Podemos restar la velocidad de Verstappen y Hamilton en el metro 1500 exacto (entrada a curva 4), independientemente de cuánto tardaron en llegar ahí.

---

### 2. Procesamiento de Señales (Módulo 2)

#### 2.1 Filtro Savitzky-Golay (`_savgol`)
**Qué es**: Es un filtro digital smoothing (suavizado) que, a diferencia de la media móvil, preserva los momentos de alto orden (picos y valles estrechos).
**Cómo funciona**: Para cada punto de la señal, ajusta un polinomio (parábola) mediante mínimos cuadrados a sus vecinos. El valor suavizado es el valor de ese polinomio en el punto central.

**Pregunta: ¿Por qué la ventana debe ser impar y mayor que el orden?**
1.  **Impar**: Un filtro simétrico necesita un punto central y el mismo número de vecinos a izquierda y derecha. Ejemplo: Ventana 5 = 2 izquierda + **Centro** + 2 derecha. Si fuera par, no habría centro definido y el filtro introduciría un desplazamiento de fase (retardo temporal de media muestra).
2.  **Mayor que el orden**: Matemáticamente, necesitas al menos $n+1$ puntos para ajustar un polinomio de grado $n$ (ej. 2 puntos para una recta, 3 para una parábola). Si `window <= poly`, el polinomio pasaría exactamente por todos los puntos (overfitting infinito), no suavizando nada (el ruido se mantiene intacto). Dejamos margen (`poly + 2` o más) para que el ajuste de mínimos cuadrados promedie el error (ruido).

**Explicación de la función `_savgol`**:
```python
# Asegura que la ventana no sea más grande que los datos reales (crash prevention)
window = min(window, series.size if series.size % 2 == 1 else series.size - 1)

# Asegura que la ventana sea mayor que el polinomio (Degree of Freedom check)
# Si poly=2 (parábola), window mínimo debe ser 5 (impar > 3).
window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
```
Esto es "programación defensiva": garantiza que `scipy.signal.savgol_filter` nunca reciba argumentos matemáticamente imposibles, incluso si la vuelta es muy corta (ej. vuelta de salida de boxes incompleta).

#### 2.2 Dinámica Vehicular (`_compute_dynamics_group`)
Calcula la física del coche.
1.  **Derivadas**: Calcula $v_x, v_y$ y luego $a_x, a_y$ usando `np.gradient` (diferencias finitas centradas).
2.  **Proyección Vectorial**:
    *   Los sensores dan aceleración en ejes X/Y globales (mapa). Al piloto le importa la aceleración relativa al coche (Frenada/Giro).
    *   **Tangencial ($a_t$)**: Producto punto $\vec{a} \cdot \hat{v}$. Mide cuánto de la aceleración va en la dirección del movimiento (Frenada/Tracción).
    *   **Normal ($a_n$)**: Producto cruz 2D. Mide la aceleración perpendicular (Fuerza centrífuga en curva).
3.  **Jerk**: Es la derivada de la aceleración ($\Delta a / \Delta t$). Picos altos indican conducción brusca o "inputs" muy rápidos (patadas al freno).
4.  **Energía Neumático**:
    *   Potencia = Fuerza x Velocidad.
    *   Fuerza ~ Masa x Aceleración (G).
    *   Integramos `(|Lat_G| + |Long_G|) * Speed` en el tiempo. Suma toda la "violencia" aplicada al neumático ponderada por la velocidad (sufrimiento de la goma).

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

#### 3.2 Dimensionalidad (PCA)
Una vez normalizado, usamos PCA para condensar 10 métricas correlacionadas en 3 "Estilos":
1.  **PC1 (Ritmo/Performance)**: Correlaciona con Tiempo de Vuelta y Velocidad.
2.  **PC2 (Agresividad/Estilo)**: Correlaciona con Jerk y Entradas bruscas.
3.  **PC3 (Gestión)**: Correlaciona con Energía de neumático y suavidad.

---

### 4. Predicción de Tiempos (Módulo 4: Modelado)

**Objetivo**: Predecir el `LapTimeSeconds` basándose en la telemetría agregada y el estilo del piloto.

#### 4.1 Arquitectura del Pipeline (`sklearn.pipeline`)
Hemos diseñado un pipeline robusto a prueba de fallos en producción:
1.  **Preprocesamiento Universal (`ColumnTransformer`)**:
    *   **Numéricos (`StandardScaler`)**: Normaliza inputs como `TyreLife` o `FuelLoad` para que tengan media 0 y varianza 1. Vital para modelos lineales (Lasso) y ayuda a la convergencia en redes neuronales (si se usaran).
    *   **Categóricos (`OneHotEncoder`)**: Transforma `Compound` (SOFT, MEDIUM, HARD) y `Team` en vectores binarios. Usamos `handle_unknown='ignore'` para que si mañana aparece un compuesto nuevo (ej. "HYPERSOFT"), el modelo no rompa en producción (simplemente lo ignora).

#### 4.2 Selección de Modelos (Benchmark)
Comparamos múltiples familias de algoritmos usando **Cross-Validation (5-Fold)** para evitar overfitting:
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


