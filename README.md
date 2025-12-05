# F1-Analytics

Proyecto universitario de analítica de rendimiento en Fórmula 1 usando FastF1, con un pipeline completo orientado a Industria 4.0: ingesta robusta, procesamiento de señales, ingeniería de características y visualización interactiva.

## Estado actual y scripts principales

- `feature_extraction/main.py` (extracción masiva 2024): descarga calendario, vueltas, neumáticos, clima y telemetría (submuestreo) de todas las sesiones 2024. Es útil para tener un “data lake” rápido, pero **no sincroniza señales** (ni temporal ni espacialmente) y limita la telemetría a las 3 mejores vueltas por piloto.
- `feature_extraction/module1_ingestion.py` (nuevo Módulo 1): ingesta orientada a análisis avanzado. Configura caché, carga una sesión específica (laps + telemetría + clima), y **alinea la telemetría en tiempo (10 Hz) y en espacio (malla de distancia)** para habilitar comparativas tipo “ghost car” y estudios de dinámica vehicular. Persistimos artefactos limpios listos para EDA/feature engineering.
- `src/f1_data.py` + `src/arcade_replay.py`: pipeline previo para replays tipo arcade; remuestrea x/y y distancia a una línea de tiempo común para animaciones.

## Cuándo usar cada script

- Si necesitas un volcado grande de datos crudos de 2024 para exploración general o dashboards rápidos: usa `feature_extraction/main.py`. Los datos son válidos, pero no traen alineación temporal/espacial ni suavizado; también submuestrean la telemetría y toman solo las mejores vueltas.
- Si necesitas señal preparada para análisis físico, comparativas vuelta a vuelta, PCA o modelado: usa `feature_extraction/module1_ingestion.py`, que entrega telemetría resampleada a 10 Hz y alineada por distancia con metadatos completos.

## Módulo 1 – Ingesta, sincronización y alineación espacial

Archivo: `feature_extraction/module1_ingestion.py`

### Qué hace
- Habilita caché local (`.fastf1-cache`) para no re-descargar datos.
- Carga una sesión FastF1 con laps, telemetría y clima (`session.load(laps=True, telemetry=True, weather=True)`).
- Selecciona la vuelta más rápida de cada piloto (pick_fastest), porque es la mejor referencia de rendimiento y suele venir con menos ruido operativo. y genera dos alineaciones:
  - **Temporal (10 Hz)**: remuestreo uniforme sobre tiempo relativo al inicio de vuelta (`SessionTime` → `RelativeTime_s`), calculamos RelativeTime_s desde el inicio de la vuelta y remuestreamos todas las señales al mismo grid temporal con interpolación + ffill/bfill para alinear sensores (Speed, RPM, Throttle, Brake, nGear, DRS, X/Y, etc.).
  - **Espacial (distancia)**: reiniciamos la distancia de la vuelta a 0, eliminamos duplicados y interpolamos sobre una malla uniforme de 1 m. Esto permite comparar dos vueltas en el mismo punto físico de pista (ghost laps) y separar estilo de conducción del tiempo absoluto. Normaliza la distancia de la vuelta a 0, elimina duplicados y interpola sobre una malla regular (paso por defecto 1 m). Así puedes comparar dos vueltas en el mismo punto de pista.
- Persiste artefactos en `data/module1_ingestion/<year>_<event>_<session_type>/`:
  - `laps.csv`: vueltas con `LapTimeSeconds`.
  - `weather.csv`: clima con `TimeSeconds`.
  - `telemetry_time_10hz.csv`: telemetría remuestreada en tiempo.
  - `telemetry_distance_aligned.csv`: telemetría alineada por distancia.
  - `laps_metadata.csv`: metadatos por vuelta rápida (Driver, LapNumber, Compound, TyreLife, LapTimeSeconds).

### Física y propósito
- **Alineación temporal**: asegura que señales de distinta cadencia (p.ej. RPM vs. DRS) se comparen en el mismo grid temporal, evitando aliasing en cálculos de derivadas.
- **Alineación espacial**: es clave para “ghost laps” y comparación de estilos porque la referencia es la posición sobre la pista, no el reloj de sesión; elimina desfases por lift-and-coast o safety car.

### Uso rápido (ejemplo Bahrein 2024, carrera)
Ejecuta el módulo directamente (usa la caché de FastF1 si ya descargaste datos):
```bash
python feature_extraction/module1_ingestion.py
```
Si trabajas con `uv`, el equivalente es:
```bash
uv run python feature_extraction/module1_ingestion.py
```
Por defecto carga Bahrein 2024 (Round 1, Carrera) para alinear con el dataset existente y guarda los artefactos en `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/`.

Parámetros clave (vía función `ingest_session`):
- `year`, `grand_prix`, `session_type` (`'R'`, `'Q'`, `'FP1'`, etc.).
- `frequency_hz` (10 Hz por defecto) para el remuestreo temporal.
- `distance_step_m` (1 m por defecto) para la malla espacial.
- `drivers` (opcional) para limitar a ciertos pilotos/códigos.

### Guía para modificar rápidamente (defensa/prototipos)
- Cambiar evento por defecto: en `main()` de `feature_extraction/module1_ingestion.py` ajusta `ingest_session(year=..., grand_prix=..., session_type=...)`.
- Resolución temporal: cambia `TIME_SAMPLE_RATE_HZ` o pasa `frequency_hz` a `ingest_session`; afecta el grid de 10 Hz donde se alinean RPM, Speed, Throttle, Brake, etc.
- Resolución espacial: cambia `DISTANCE_STEP_METERS` o pasa `distance_step_m` a `ingest_session`; reduce/aumenta el paso de la malla de distancia (ghost laps más finas vs. peso de datos).
- Señales a interpolar: edita `DEFAULT_TELEMETRY_COLUMNS`; el módulo toma solo las columnas numéricas existentes, por lo que puedes añadir SpeedST, ERSDeploy si existen en la sesión.

### Explicación detallada de la alineación (para defensa/prototipos)
1) **Caché y carga**: se habilita `.fastf1-cache` y se llama `session.load(laps=True, telemetry=True, weather=True)` para obtener vueltas, telemetría cruda y clima en un solo paso.  
2) **Selección de vuelta**: por piloto se toma la vuelta más rápida (`pick_fastest`) porque suele ser la más representativa y con menor ruido operativo; es el insumo para comparativas y modelado.  
3) **Alineación temporal 10 Hz**: se define `RelativeTime_s` desde el inicio de la vuelta para que todas las señales compartan un origen común. Se remuestrea con paso fijo (`1/frequency_hz`) usando `ffill/bfill` + interpolación en DataFrame plano (evita warnings y asegura tipos numéricos), lo que evita aliasing entre canales de distinta cadencia (RPM vs. DRS) y deja listas las señales para derivadas suaves en el Módulo 2.  
4) **Alineación espacial por distancia**: se reinicia la distancia al cero de la vuelta, se eliminan duplicados y se interpola sobre una malla regular (1 m por defecto) con `ffill/bfill` numérico. Esto permite comparar dos vueltas en el mismo punto físico de pista (ghost car) y separar efectos de estilo de conducción de los efectos de tiempo absoluto.  
5) **Persistencia de artefactos**: se guardan las vueltas (`laps.csv` con `LapTimeSeconds`), el clima (`weather.csv` con `TimeSeconds`), la telemetría temporal (`telemetry_time_10hz.csv`), la telemetría espacial (`telemetry_distance_aligned.csv`) y metadatos de la vuelta rápida (`laps_metadata.csv` con compuesto, vida de neumático, lap time). Todo queda en `data/module1_ingestion/<event>/` para que Módulos 2–4 consuman formatos consistentes sin depender de FastF1 en tiempo de presentación.  

### Datos recolectados (estado actual)
- Evento: Bahrain Grand Prix 2024, sesión de Carrera (`session_type='R'`, Round 1).
- Pilotos: 20 entradas (números oficiales 2024): 1 (VER), 11 (PER), 55 (SAI), 16 (LEC), 63 (RUS), 4 (NOR), 44 (HAM), 81 (PIA), 14 (ALO), 18 (STR), 24 (ZHO), 20 (MAG), 3 (RIC), 22 (TSU), 23 (ALB), 27 (HUL), 31 (OCO), 10 (GAS), 77 (BOT), 2 (SAR).
- Artefactos en `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/`:
  - `laps.csv`: todas las vueltas con tiempos (`LapTimeSeconds`), compuestos, vida de neumático y marcas de mejor personal.
  - `weather.csv`: clima por timestamp (temperatura pista/aire, viento, humedad) con `TimeSeconds`.
  - `telemetry_time_10hz.csv`: telemetría remuestreada a 10 Hz por vuelta rápida de cada piloto (Speed, Throttle, Brake, nGear, RPM, DRS, X, Y, Distance, RelativeTime_s, Driver, LapNumber, Compound, TyreLife).
  - `telemetry_distance_aligned.csv`: mismas señales en malla de distancia de 1 m (Distance_m, Distance_pct) para comparativas espaciales.
  - `laps_metadata.csv`: metadatos de cada vuelta rápida usada para telemetría (Driver, LapNumber, Compound, TyreLife, LapTimeSeconds).
- Módulo 2 añade sobre el mismo evento:
  - `telemetry_time_10hz_enriched.csv`: dinámica vehicular (aceleraciones, jerks, energía proxy).
  - `lap_features_module2.csv`: features agregadas por vuelta (suavidad, g máximas, energía, etc.).

### Fundamento teórico (análisis y transformaciones)
- **Remuestreo temporal (10 Hz)**: se define `t_rel = t - t_ini_vuelta` y se remuestrea cada señal con paso Δt = 0.1 s. Interpolar + ffill/bfill evita aliasing entre canales con distinta cadencia (ej. RPM más rápido que DRS). Esto genera un grid uniforme para derivar numéricamente (gradiente).
- **Remuestreo espacial (1 m)**: se normaliza la distancia de vuelta a 0 y se interpola cada canal sobre una malla Δs = 1 m. Las comparaciones “ghost” se hacen a igual punto de pista (no al mismo tiempo), separando estilo de conducción de condiciones temporales (lift-and-coast, SC, etc.).
- **Savitzky–Golay (SG)**:
  - Objetivo: suavizar sin retraso de fase y obtener derivadas robustas. SG ajusta un polinomio local de orden `p` sobre una ventana impar `W` centrada; el valor suavizado es la evaluación del polinomio en el centro. Matemáticamente, para cada punto i se resuelve por mínimos cuadrados el polinomio que mejor ajusta la ventana `[i - (W-1)/2, i + (W-1)/2]`. Este polinomio actúa como filtro pasa-bajas con mejor preservación de forma que un promedio móvil.
  - Derivadas: SG permite obtener derivadas analíticas del polinomio ajustado; aquí usamos SG para suavizar primero y luego derivar con gradiente discreto estable. Ventana mayor = más suavidad (menos ruido, menos respuesta a picos); orden mayor = más capacidad de seguir curvatura pero más riesgo de sobreajuste local.
  - Convolución y variable independiente: SG es equivalente a una convolución con un kernel fijo obtenido por mínimos cuadrados. Para `W = 2k+1`, definimos coordenadas locales `u ∈ {-k, ..., k}` (variable independiente). Ajustamos `p(u) = Σ_{m=0..p} a_m u^m` minimizando Σ (y_{i+u} − p(u))²; las ecuaciones normales producen coeficientes `c_smooth` tales que `y_suav(i) = Σ c_smooth[u] · y_{i+u}`. En nuestro caso `u` representa tiempo relativo con paso Δt = 0.1 s (remuestreo temporal); si se aplica en espacio sería `u*Δs` con Δs = 1 m.
  - Derivadas vía kernel: los coeficientes para derivada `c_deriv` aproximan `dp/du` en el centro y se escalan por `1/Δt` (o `1/Δs`). Al usar ventana centrada y coeficientes simétricos no hay retardo de fase: los picos de frenada/DRS quedan alineados entre señales.
  - Qué es `x` y qué es `y` en nuestro uso: `x` es el índice regularizado que construimos en Módulo 1 (tiempo relativo a la vuelta en pasos de Δt = 0.1 s). `y` son las series que suavizamos: (1) `Speed` convertida a m/s, y (2) las posiciones `X`, `Y` en metros de la telemetría. Aplicamos SG a estas `y` porque luego derivamos respecto a `x` para obtener velocidades/acceleraciones/jerks con menos ruido. No aplicamos SG a señales binarias (DRS) ni categóricas.
- **Dinámica vehicular (coordenadas tangencial/normal)**:
  - Posición en plano: `(x(t), y(t))` suavizados.
  - Velocidad: `v = (vx, vy) = d(x,y)/dt`; `|v| = sqrt(vx^2 + vy^2)`.
  - Aceleración: `a = (ax, ay) = d(vx, vy)/dt`.
  - Descomposición: aceleración longitudinal (tangencial) `a_t = (a · v)/|v|` explica cambios de módulo de velocidad; aceleración lateral (normal) `a_n = (ax*vy - ay*vx)/|v|` explica el cambio de dirección (curvatura). Signo de `a_n` indica giro.
  - Jerk: `j = da/dt` (derivada de aceleración). `j_long`, `j_lat` miden rapidez de cambio de inputs de pedales y volante. Magnitud alta → transiciones bruscas; magnitud baja → conducción suave.
- **Proxy de energía de neumático**:
  - Potencia de deslizamiento aproximada `P ~ |a| * v` (asumiendo masa unitaria y que fuerza ~ aceleración). Se integra en el tiempo para un índice acumulado: `E = ∑ (|a_t| + |a_n|) * |v| * Δt`. No son Joules físicos, pero el índice permite comparar demanda de energía/disipación en la goma entre vueltas y pilotos.
- **Agregados por vuelta (lap features)**:
  - Suavidad: `MeanAbs_Jerk_Long`, `MeanAbs_Jerk_Lat` (menor = más suave).
  - Agresividad de freno: desviación estándar de `Brake`.
  - Carga pico: `Max_Lateral_g`, `Max_Longitudinal_g` (aceleraciones normalizadas por 9.81).
  - Ritmo y energía: `Avg_Speed_mps`, `Energy_Index` (valor final de E), `LapTimeSeconds`, compuesto y vida de neumático como contexto.

## Módulo 2 – Procesamiento de señales y métricas físicas

Archivo: `feature_extraction/module2_signals.py`

### Qué hace (pipeline detallado)
- **Entrada**: reutiliza los artefactos del Módulo 1 (`telemetry_time_10hz.csv`, `telemetry_distance_aligned.csv`, `laps_metadata.csv`) desde `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/` por defecto.
- **Suavizado Savitzky–Golay**: aplica filtro SG sobre velocidad y posición XY (ventana impar, orden 2) para suavizar sin desfase de fase, manteniendo la forma de la señal. Esto reduce ruido antes de derivar.
- **Cálculo de dinámica vehicular** por piloto y vuelta (evitamos saltos entre vueltas):
  - Velocidades en plano: se derivan las posiciones suavizadas `X`, `Y` respecto al tiempo (dt = 0.1 s a 10 Hz).
  - Aceleraciones: derivada de las velocidades (`AX`, `AY`).
  - Descomposición física: aceleración longitudinal (tangencial) `AX_long` = proyección de la aceleración en la dirección de la velocidad (explica cambios de velocidad) y aceleración lateral `AY_lat` = componente normal (explica giro/curvatura).  
  - Jerk longitudinal/lateral: derivada de cada aceleración. Esto mide la agresividad o suavidad en la transición de pedales y volante: valores altos = inputs bruscos, valores bajos = conducción suave.
  - Proxy de energía de neumático: potencia ~ `F * v` ≈ `|a| * v` (considerando masa unitaria). Integramos en el tiempo `( |a_long| + |a_lat| ) * velocidad` para obtener un índice acumulado (`TireEnergyProxy`) que se interpreta como carga/energía disipada en el neumático a lo largo de la vuelta (no está en Joules reales, pero es comparativo entre vueltas).
- **Features por vuelta**: agrega métricas lap-level combinando telemetría enriquecida y metadatos:
  - `Avg_Speed_mps`, `Avg_Throttle`
  - `Brake_Aggression` = std de la señal de freno
  - `MeanAbs_Jerk_Long`, `MeanAbs_Jerk_Lat` = suavidad de pedales/volante
  - `Max_Lateral_g`, `Max_Longitudinal_g` (aceleraciones normalizadas por g)
- `Energy_Index` = valor final de `TireEnergyProxy` en la vuelta
- Metadatos de contexto: `LapTimeSeconds`, `Compound`, `TyreLife`

### Interpretación y próximos pasos (Módulo 2)
- Qué mirar: curvas de `Speed` vs `RelativeTime_s`/`Distance_m`, jerk longitudinal/lateral (picos = inputs bruscos), y la curva acumulada de `TireEnergyProxy` (más pendiente = más carga sobre el neumático). En `lap_features_module2.csv` compara `MeanAbs_Jerk` vs `LapTimeSeconds` para ver si la suavidad correlaciona con ritmo.
- Gráficos sugeridos (añadidos en `notebooks/tests.ipynb`):
  - Velocidad vs tiempo (vuelta más rápida) y vs distancia (top 2) para “ghost laps”.
  - Throttle/Brake vs tiempo para la vuelta más rápida.
  - Jerk_long/Jerk_lat vs tiempo para la vuelta más rápida (picos → transiciones agresivas).
- Curva de `TireEnergyProxy` a lo largo de la vuelta para los dos más rápidos.
- Dispersión `MeanAbs_Jerk_Long` vs `LapTimeSeconds` (suavidad vs performance).
- Justificación: estas visuales permiten validar que el suavizado no introduce fase, que las derivadas son estables y que el proxy de energía refleja zonas de alta demanda (curvas rápidas, frenadas fuertes). Si detectas ruido residual, ajusta la ventana/orden SG en `module2_signals.py`.
- Siguiente módulo: usar `lap_features_module2.csv` y/o `telemetry_time_10hz_enriched.csv` para Ingeniería de Características/PCA (Módulo 3) y preparar dataset de modelado (Módulo 4).

### EDA recomendado (ya en `notebooks/tests.ipynb`)
- Estadísticos básicos (`describe`) de `LapTimeSeconds`, `MeanAbs_Jerk_*`, `Energy_Index`, `Avg_Speed_mps`, `Brake_Aggression`, `Max_*_g`.
- Correlaciones con `LapTimeSeconds` para ver qué métricas físicas se asocian más al ritmo.
- Scatter matrix reducido (`LapTimeSeconds`, `Avg_Speed_mps`, `Energy_Index`, `MeanAbs_Jerk_*`) para observar relaciones no lineales.
- Boxplots por `Compound` (LapTime, Energy_Index, Jerk_long) para evidenciar diferencias por neumático.
- Dispersión `TyreLife` vs `LapTimeSeconds` y vs `Energy_Index` para evaluar efecto de desgaste.
- Visuales añadidas para intuición espacial/temporal: heatmap de correlaciones anotadas; velocidad vs distancia coloreada por throttle; mapa XY coloreado por velocidad; scatter Throttle vs Brake coloreado por tiempo; histogramas de jerk por compuesto; energía de neumático acumulada en el tiempo para top 2.
- Estas celdas están agregadas en la sección de EDA estadístico del notebook.

### Interpretación de lo explorado en el notebook
- **Ritmo vs suavidad**: la dispersión `MeanAbs_Jerk_Long` vs `LapTimeSeconds` suele mostrar que menor jerk longitudinal (inputs más suaves de acelerador/freno) tiende a correlacionar con mejores tiempos, aunque no siempre es lineal; la scatter matrix ayuda a detectar outliers o relaciones no lineales.
- **Energía de neumático**: la curva de `TireEnergyProxy` a lo largo de la vuelta señala zonas de alta demanda (pendiente pronunciada en frenadas/curvas rápidas). El `Energy_Index` final, comparado entre pilotos/top 2, sugiere quién “gasta” más neumático para lograr su tiempo.
- **Throttle/Brake vs tiempo y vs distancia**: al colorear velocidad por throttle en función de la distancia se identifican zonas de lift-and-coast o aplicación parcial de acelerador; el scatter Throttle vs Brake coloreado por tiempo muestra secuencias de inputs y si hay solapes (malas prácticas) o transiciones limpias.
- **Mapa XY coloreado por velocidad**: permite ubicar visualmente en el circuito dónde se alcanzan velocidades máximas y cómo se distribuyen los picos de frenada/aceleración; útil para contrastar con el proxy de energía.
- **Compuesto y desgaste**: boxplots por `Compound` para LapTime/Energy/Jerk y scatter `TyreLife` vs LapTime/Energy permiten evaluar si compuesto y edad del neumático impactan la suavidad y el ritmo (p.ej., neumáticos más frescos con menor jerk y menor Energy_Index).
- **Correlación global**: el heatmap anotado resume qué features tienen mayor relación con el LapTime; típicamente `Avg_Speed_mps` y `MeanAbs_Jerk_Long` emergen como relevantes. Esto orienta la selección de variables para PCA y modelado.

## Módulo 3 – Ingeniería de features y PCA

Archivo: `feature_extraction/module3_pca.py`

### Qué hace
- Lee `lap_features_module2.csv` (artefactos de Módulo 2).
- Construye una matriz de features numéricas (`LapTimeSeconds`, `Energy_Index`, `MeanAbs_Jerk_*`, `Avg_Speed_mps`, `Brake_Aggression`, `Max_*_g`, `TyreLife`) y añade dummies de `Compound`.
- Estandariza con `StandardScaler` y aplica PCA (`n_components=3` por defecto).
- Persiste:
  - `pca_scores_module3.csv`: PC1/PC2/PC3 + metadatos (Driver, LapNumber, Compound).
  - `pca_model_module3.json`: varianza explicada, loadings (componentes), medias/escalas del scaler y nombres de features.

### Uso rápido
```bash
python feature_extraction/module3_pca.py
```
o con `uv`:
```bash
uv run python feature_extraction/module3_pca.py
```
Por defecto consume `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/lap_features_module2.csv` y genera los artefactos en el mismo directorio.

### Teoría y lectura del PCA
- PCA busca vectores ortogonales (componentes principales) que maximizan la varianza de los datos estandarizados. PC1 captura la mayor varianza, PC2 la siguiente, etc. Las “loadings” son las proyecciones de cada feature en estos ejes (componentes de los eigenvectores de la matriz de covarianza).
- Estandarización: imprescindible porque las features tienen unidades distintas (s, m/s, m/s³, g). Se resta la media y se divide por la desviación estándar antes del PCA.
- Interpretación: PC1/PC2 pueden separar “estilo de conducción” (jerk, brake aggression) de “condición del coche” (energía, velocidad media, compuesto). Un loading alto (en valor absoluto) indica que la feature contribuye fuertemente a esa componente. La varianza explicada indica cuánta información retienen PC1/PC2; si suman >60% ya dan buena visualización 2D.
- Gráficos sugeridos (añadidos en `notebooks/tests.ipynb`):
  - Dispersión PC1 vs PC2 coloreada por piloto (clusters de estilo).
  - Dispersión PC1 vs PC2 coloreada por compuesto (impacto de neumático).
  - Tabla de loadings ordenada para PC1/PC2 (qué físicas dominan cada eje).
  - Visuales avanzadas: ejes cruzados y elipse de confianza 95% en PC1/PC2; biplot con vectores de variables superpuestos a los individuos; scatter 3D PC1-PC2-PC3 coloreado por piloto; biplot 3D (vectores de variables en PC1/PC2/PC3).

### Consideraciones sobre variancia y mejoras posibles
- Con los artefactos actuales: PC1 ≈ 39%, PC2 ≈ 22%, PC3 ≈ 12% (≈73% acumulado). PC1–PC2 (~61%) es aceptable para visualización 2D; para capturar más estructura usa PC3 o las features originales. Para más separación en 2D, puedes: (a) añadir features específicas (p.ej., medias/picos de throttle/brake, % tiempo en throttle/brake, número de eventos de jerk alto, min/max speed, sectorización por distancia), (b) quitar colinealidad si hay signals redundantes, o (c) usar proyecciones no lineales (UMAP/t-SNE) solo para visual.
- Los vectores de variables (biplot) muestran la inclinación de cada feature respecto a PC1/PC2 (y en 3D, respecto a PC3): flechas orientadas hacia cuadrantes con mayor contribución de jerk/agresividad o de velocidad/energía/compound. Esto ayuda a inferir si un eje es más “estilo” o más “coche/neumático”.
- Lectura de gráficos: en 2D, PC1 separa principalmente ritmo (Avg_Speed_mps a la derecha, LapTime a la izquierda); jerk y energía apuntan en el mismo cuadrante (más demanda/agresividad). Compounds: SOFT se alinea con velocidad, HARD con TyreLife/LapTime. En 3D, PC3 aporta separación residual; vueltas que se solapan en el plano PC1-PC2 pueden distanciarse en PC3 (útil para clustering/modelo).

## Módulo 4 – Modelado predictivo (LapTime)

Archivo: `feature_extraction/module4_modeling.py`

### Qué hace
- Carga el dataset consolidado `feature_extraction/data/module1_ingestion/all_lap_features.csv` y, si existen, añade `PC1-3` globales normalizados (`pca_scores_global_norm.csv`) como features opcionales.
- Preprocesa: escalado para numéricas y one-hot para categóricas.
- Entrena y compara varios modelos:
- RandomForestRegressor (baseline no lineal, robusto).
- GradientBoostingRegressor (boosting escalar).
- Lasso (baseline lineal regularizado).
- XGBoostRegressor y LightGBMRegressor (si están instalados) con hiperparámetros moderados.
- Se eliminan filas con valores NaN antes de entrenar (al usar el dataset global y PCs) para evitar fallos de sklearn; si hay sesiones con datos incompletos, esas filas se descartan del entrenamiento.
- Validación cruzada K-Fold (5 folds, shuffle=True, random_state=42); métricas promedio: MAE_mean, RMSE_mean, R2_mean. Guarda:
  - `model_metrics_module4.json`
  - `best_model_module4.pkl` (pipeline completo)
  - `val_predictions_module4.csv` (y_true/y_pred en validación) para análisis visual.
- Interpretabilidad (opcional, si `shap` está instalado):
  - Se calcula un subconjunto de valores SHAP (muestra de 500 filas máx) y se guardan en `data/module1_ingestion/shap/`:
    - `shap_values.parquet`: matriz SHAP (muestra × features, en el espacio transformado/one-hot).
    - `shap_input_sample_raw.csv`: muestra de entrada cruda (sin preprocesar) para contexto.
    - `shap_feature_names.csv`: nombres de columnas ya transformadas (coinciden con `shap_values`).
    - `shap_mean_abs.csv`: importancia media absoluta por feature (post-encoding).
  - Esto permite explicar cómo cada variable física y de neumáticos empuja el tiempo de vuelta.

### Uso rápido
```bash
python feature_extraction/module4_modeling.py
```
o con `uv`:
```bash
uv run python feature_extraction/module4_modeling.py
```
Usa por defecto los artefactos en `data/module1_ingestion/2024_Bahrain_Grand_Prix_R/`.

### Visualizaciones de resultados (en `notebooks/tests.ipynb`)
- Barras comparando MAE y RMSE por modelo.
- Dispersión y_true vs y_pred del mejor modelo (línea y=x para ver sesgo).
- Histograma de residuos (ideal centrado en 0, simétrico).
- Violin/box de residuos por `Year` o `Compound` para ver estabilidad entre temporadas y compuestos.
- SHAP (si existe `data/module1_ingestion/shap/`):
  - Barra de importancias medias absolutas (top 15) desde `shap_mean_abs.csv`.
  - Beeswarm usando `shap_values.parquet` + `shap_feature_names.csv` (puedes cargar `shap_input_sample_raw.csv` para contexto de las features originales).
- Estas gráficas permiten evaluar precisión, sesgo, dispersión y qué drivers físicos dominan las predicciones.

### Justificación y siguientes pasos
- Modelos no lineales (RF, GB, XGB) capturan interacciones entre dinámica física (jerk, g’s, energía) y contexto (compound, TyreLife). Lasso da una referencia lineal e interpretable.
- Si la varianza del PCA no es suficiente para separar estilos, se pueden usar features originales + PC1-3 (pipeline lo soporta). XGBoost puede manejar ambas sin necesidad de PCA.
- Refinamientos: tuning de hiperparámetros, añadir más features de estilo (eventos de jerk alto, % tiempo en throttle/brake), o validación cruzada estratificada por piloto/compound para robustez.

## Batch y consolidación de datos (para escalar el dataset)

- `feature_extraction/batch_ingest.py`: recorre múltiples años/rounds/sesiones y ejecuta `ingest_session` (Módulo 1) en batch. Configura los años, ronda y sesiones (`YEARS`, `ROUND_RANGE`, `SESSION_TYPES`) y ejecuta:
  ```bash
  uv run python feature_extraction/batch_ingest.py
  ```
- `feature_extraction/merge_lap_features.py`: concatena todos los `lap_features_module2.csv` bajo `data/module1_ingestion/*` y añade columnas `Year`, `Event`, `SessionType`, generando `data/module1_ingestion/all_lap_features.csv`:
  ```bash
  uv run python feature_extraction/merge_lap_features.py
  ```
- Sesiones sin datos o incompletas: si un GP no tiene vueltas válidas (ej. Spa 2021 con 2 vueltas tras SC) o la API entrega telemetría/posición incompleta para un piloto, `ingest_session` ignora ese piloto y `merge_lap_features.py` omite sesiones vacías. Es normal ver warnings de “Position/Car data is incomplete”; los módulos posteriores sólo usan vueltas que se pudieron alinear.
- `feature_extraction/batch_module2.py`: ejecuta el Módulo 2 para todas las sesiones del Módulo 1 que aún no tengan `lap_features_module2.csv` (requiere que exista `telemetry_time_10hz.csv` en cada carpeta):
  ```bash
  uv run python feature_extraction/batch_module2.py
  ```
- Dataset consolidado actual: `feature_extraction/data/module1_ingestion/all_lap_features.csv` con 1,719 filas (vueltas rápidas por piloto) de 2021–2024, solo sesiones de carrera (`SessionType=R`). Conteo por año: 2021 (403), 2022 (426), 2023 (427), 2024 (463), con 21, 22, 22 y 24 eventos respectivamente. Esto es la base para reejecutar PCA y modelado multiaño.

### EDA multiaño (notebook `notebooks/tests.ipynb`)
- Se añadieron celdas para explorar `all_lap_features.csv`:
  - Boxplots y KDE de `LapTimeSeconds` por año.
  - Scatter `LapTime` vs `Energy_Index` y vs `Avg_Speed_mps` coloreado por año.
  - Boxplots por `Compound` (LapTime, Energy, Jerk_long).
  - Scatter `TyreLife` vs LapTime y vs Energy.
  - Heatmap de correlación de variables numéricas.
  - Violines de `Energy_Index` y `MeanAbs_Jerk_Long` por año.
- Estas visuales permiten verificar consistencia entre temporadas, impacto de neumático y relación ritmo/energía/jerk antes de reentrenar PCA y modelos con el dataset completo.

### PCA global multiaño
- Script: `feature_extraction/module3_pca_global.py` toma `feature_extraction/data/module1_ingestion/all_lap_features.csv`, arma la matriz con features físicas + dummies de Compound y Year, estandariza y calcula PCA (PC1–PC3). Persiste:
  - `pca_scores_global.csv` (PC1/PC2/PC3 + Driver, LapNumber, Compound, Year, Event, SessionType)
  - `pca_model_global.json` (varianza explicada, loadings, media/escala del scaler, nombres de features)
- Uso:
  ```bash
  uv run python feature_extraction/module3_pca_global.py
  ```
- Nota: el PCA global descarta filas con NaN en cualquiera de las features de entrada antes de escalar (para evitar fallos de scikit-learn). Asegura que `all_lap_features.csv` esté completo; en caso de datos incompletos de alguna sesión/piloto, esas filas se eliminan de la matriz PCA pero permanecen en el archivo original.

### PCA global normalizado por evento (nuevo)
- Script: `feature_extraction/module3_pca_global_normalized.py` normaliza por evento para quitar el efecto pista:
  - Z-score por (Year, Event) de LapTime, Speed, Jerk, g y Brake_Aggression.
  - Energy_Index se transforma a `Energy_per_m` (divide por distancia estimada de vuelta) y se z-score por evento.
  - TyreLife también se z-score por evento.
  - Se añaden dummies de Compound (Year ya absorbido en la normalización).
- Genera:
  - `pca_scores_global_norm.csv` (PCs + Driver, LapNumber, Compound, Year, Event, SessionType)
  - `pca_model_global_norm.json` (varianza, loadings, scaler)
- Uso:
  ```bash
  uv run python feature_extraction/module3_pca_global_normalized.py
  ```
- Visuales en el notebook (sección PCA global normalizado): varianza explicada, PC1/PC2 por año/compound, biplot 2D, PC1 vs LapTime y PC1 vs Energy_per_m. Este PCA suele concentrar más varianza en PC1/PC2 al eliminar el efecto pista y dejar más señal de estilo/estrategia.
- Último run (normalizado): PC1 ≈ 23.2%, PC2 ≈ 13.9%, PC3 ≈ 12.6% (≈ 49.7% acumulado). Aunque PC1 baja ligeramente respecto al PCA bruto, PC2/PC3 suben, señal de que la normalización repartió de forma más uniforme la varianza al eliminar el efecto pista; más útil para comparar estilos/estrategias entre eventos.
- Visuales en el notebook (sección PCA global): varianza explicada (barras+acumulada), scatter PC1/PC2 coloreado por año o compound, biplot 2D (vectores de variables), scatter 3D PC1/PC2/PC3 por año, y PC1 vs LapTime/Energy para comprobar que PC1 captura ritmo/demanda.

## Nuevas features (para robustecer el modelado)

A partir de `telemetry_time_10hz_enriched.csv` se derivan:
- `Throttle_mean`, `Throttle_p90`, `Throttle_time_pct`: estadísticos y % de tiempo con throttle > 10%.
- `Brake_mean`, `Brake_p90`, `Brake_time_pct`: estadísticos y % de tiempo con brake > 5%.
- `Speed_min`, `Speed_max`: velocidad mínima y máxima de la vuelta.
- `JerkLong_events`, `JerkLat_events`: conteo de muestras donde |jerk| > 5 m/s³ (eventos de brusquedad).

Estas se fusionan con `lap_features_module2.csv` antes del modelado y pueden ayudar a separar estilos/estrategias y rendimiento del coche.

## K-Fold en el modelado

- El Módulo 4 ahora usa K-Fold (5-fold, shuffle, random_state=42) para calcular métricas promedio (MAE_mean, RMSE_mean, R2_mean), reduciendo la variabilidad de un solo split en datasets pequeños. Para mayor robustez, puedes aumentar `n_splits` o usar CV estratificado por piloto/compound.
## Glosario (términos y variables)
- Throttle: porcentaje de apertura del acelerador (0% sin gas, 100% pedal a fondo).
- Brake: porcentaje de presión de freno (0% sin freno, 100% máxima presión).
- Speed: velocidad del coche (km/h); `Avg_Speed_mps` es la velocidad media en m/s.
- LapTimeSeconds: tiempo de vuelta en segundos.
- LapNumber: número de vuelta en la sesión.
- Compound: tipo de neumático (SOFT, MEDIUM, HARD); influye en grip y desgaste.
- TyreLife: edad del neumático en número de vueltas desde que se montó.
- nGear: marcha engranada (entero).
- RPM: revoluciones del motor por minuto.
- DRS: estado del ala trasera móvil (0 cerrado, 1 abierto).
- Distance / Distance_m: distancia recorrida en la vuelta (m); `Distance_pct` es el porcentaje de vuelta.
- RelativeTime_s: tiempo relativo desde el inicio de la vuelta (segundos) tras remuestreo a 10 Hz.
- Savitzky–Golay: filtro que ajusta polinomios locales en ventanas fijas para suavizar sin retraso y derivar con ruido reducido.
- Aceleración longitudinal (AX_long): componente tangencial de la aceleración (cambios de velocidad en la dirección de avance).
- Aceleración lateral (AY_lat): componente normal de la aceleración (cambios de dirección; carga en curvas).
- Jerk_long / Jerk_lat: derivada de la aceleración (tasa de cambio de AX/AY). Mide brusquedad de inputs (pedal/volante); menor jerk = conducción más suave.
- Brake_Aggression: desviación estándar de la señal de freno; variabilidad alta indica uso más agresivo/errático del pedal.
- Max_Longitudinal_g / Max_Lateral_g: picos de aceleración normalizados por la gravedad (9.81 m/s²); reflejan frenadas/aceleraciones fuertes y carga en curvas.
- Energy_Index (TireEnergyProxy): integral aproximada de |a|·v a lo largo de la vuelta (masa unitaria), proxy de energía/carga disipada en el neumático; mayor valor implica mayor demanda térmica/mecánica.
- PCA (PC1/PC2/PC3): componentes principales que combinan linealmente las features para maximizar varianza explicada (tras estandarizar).

### Resultados y visualización (artefactos ejecutados)
- Los scores y el modelo quedan en `pca_scores_module3.csv` y `pca_model_module3.json`. El notebook carga estos artefactos, grafica la varianza explicada (barras + acumulada) y muestra loadings ordenados (magnitud de contribución por feature).
- Las dispersión PC1 vs PC2 se colorea por piloto y por compuesto para ver si hay clusters de estilo (jerk/agresividad) vs efectos de neumático. PC1 vs LapTimeSeconds permite comprobar si PC1 captura principalmente ritmo.
- Los loadings muestran qué variables dominan PC1/PC2; una magnitud alta en jerk o brake aggression sugiere eje de “suavidad/estilo”, mientras que alta en Avg_Speed/Energy/Compound indica eje más ligado a rendimiento del coche/neumático.
- En los artefactos actuales la varianza explicada es aproximada: PC1 ≈ 39%, PC2 ≈ 22%, PC3 ≈ 12% (≈ 73% acumulado con 3 PCs). Con solo dos componentes ~61% de varianza: es aceptable para visualización 2D (PC1 vs PC2) pero para capturar más estructura en modelado conviene usar PC3 (o mantener features originales). Si requieres >70% en 2D, puedes: (a) ampliar features relevantes (p.ej. medias/picos de throttle/brake, sectores), (b) probar PCA tras eliminar colinealidad fuerte, o (c) explorar métodos no lineales (UMAP/t-SNE) para visual sólo.

### Uso rápido
```bash
python feature_extraction/module2_signals.py
```
o con `uv`:
```bash
uv run python feature_extraction/module2_signals.py
```
Por defecto consume los outputs de Bahrein 2024 (`data/module1_ingestion/2024_Bahrain_Grand_Prix_R/`) y genera:
- `telemetry_time_10hz_enriched.csv` (telemetría temporal con dinámicas, jerks y energía)
- `lap_features_module2.csv` (features agregadas por vuelta)

### Guía de modificación (rápida y explícita)
- Cambiar GP o sesión: ajusta `DEFAULT_BASE_DIR` o pasa `base_dir` a `run_module2()`.
- Suavizado SG: controla ventana y orden con `DEFAULT_SAVGOL_WINDOW` (impar) y `DEFAULT_SAVGOL_POLY`. Ventanas mayores = más suavidad, menos captura de picos rápidos.
- Frecuencia de muestreo: `TIME_SAMPLE_RATE_HZ` debe coincidir con Módulo 1; si cambias el remuestreo en M1, actualiza aquí para derivadas consistentes.
- Señales: se asume que `Speed`, `X`, `Y`, `Throttle`, `Brake` existen en `telemetry_time_10hz.csv`. Si añades más canales en M1 (p.ej. ERSDeploy), se pueden agregar a los agregados de vuelta fácilmente.

### Interpretación para defensa
- **Jerk**: mide la tasa de cambio de la aceleración. Alto jerk longitudinal → transiciones bruscas de acelerador/freno; alto jerk lateral → inputs de dirección rápidos. Pilotos suaves tendrán jerk medio/bajo y menores picos.
- **Energía de neumático**: integra carga dinámica (aceleración) multiplicada por velocidad; vueltas con más `Energy_Index` tienden a ser más exigentes con la goma (mayor probabilidad de degradación térmica/mecánica).
- **Aceleraciones g**: `Max_Lateral_g` refleja la carga en curvas; `Max_Longitudinal_g` refleja frenadas/aceleraciones fuertes. Combinado con jerk ayuda a distinguir coche con buen grip vs. piloto agresivo.

## Notas sobre calidad de datos actuales

- Los CSV ya recolectados con `feature_extraction/main.py` son útiles para análisis agregados, pero **no traen sincronización 10 Hz ni interpolación por distancia**, y la telemetría está submuestreada (1 de cada 10 puntos) y limitada a las 3 mejores vueltas por piloto. Para análisis físico fino, recomendación: regenerar las vueltas clave con `module1_ingestion.py`.
- El nuevo módulo preserva más fidelidad y produce artefactos específicos para EDA, feature engineering y modelado (PCA, XGBoost/SHAP en módulos siguientes).

## Próximos módulos (plan)

- Módulo 2: procesamiento de señales (Savitzky-Golay para suavizar velocidad y derivadas, jerk longitudinal/lateral, proxy de energía de neumático).
- Módulo 3: ingeniería de características y PCA (matriz por vuelta, reducción de dimensionalidad).
- Módulo 4: modelado predictivo (XGBoost + SHAP para explicar tiempos de vuelta).
- Módulo 5: app Streamlit (ghost car, monitor de degradación, driving style map).
