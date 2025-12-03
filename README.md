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
