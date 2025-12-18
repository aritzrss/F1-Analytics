import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from pathlib import Path

import requests
from feature_extraction import module4_modeling

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
st.set_page_config(
    page_title="F1 Analytics - Industry 4.0",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ruta base donde los scripts guardan los datos (ajustar según tu repo)
BASE_DATA_DIR = Path("feature_extraction/data/module1_ingestion")

# Colores F1 (Aproximados por equipo 2024 para visualización)
TEAM_COLORS = {
    'VER': '#3671C6', 'PER': '#3671C6',  # Red Bull
    'LEC': '#E8002D', 'SAI': '#E8002D',  # Ferrari
    'HAM': '#27F4D2', 'RUS': '#27F4D2',  # Mercedes
    'NOR': '#FF8000', 'PIA': '#FF8000',  # McLaren
    'ALO': '#229971', 'STR': '#229971',  # Aston Martin
}

# ==========================================
# FUNCIONES DE CARGA DE DATOS (CACHÉ)
# ==========================================

@st.cache_data
def get_available_events():
    """Escanea el directorio de datos para encontrar eventos procesados."""
    if not BASE_DATA_DIR.exists():
        return []
    # Filtrar solo directorios que parecen eventos (tienen año al principio)
    dirs = [d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()]
    return sorted(dirs, reverse=True)

@st.cache_data
def load_session_data(event_folder):
    """Carga los CSVs clave de una sesión específica con manejo de errores y cálculo físico."""
    path = BASE_DATA_DIR / event_folder
    
    data = {}
    
    # 1. Telemetría Enriquecida (Módulo 2)
    if (path / "telemetry_time_10hz_enriched.csv").exists():
        data['telemetry_time'] = pd.read_csv(path / "telemetry_time_10hz_enriched.csv")
    elif (path / "telemetry_time_10hz.csv").exists():
        data['telemetry_time'] = pd.read_csv(path / "telemetry_time_10hz.csv")
        
    # 2. Telemetría Alineada por Distancia (Módulo 1)
    if (path / "telemetry_distance_aligned.csv").exists():
        df_dist = pd.read_csv(path / "telemetry_distance_aligned.csv")
        
        # --- FIX CRÍTICO: Reconstruir el tiempo usando Física ---
        # El archivo solo tiene Distancia y Velocidad. T = Distancia / Velocidad.
        # Integramos paso a paso para obtener RelativeTime_s.
        if 'RelativeTime_s' not in df_dist.columns:
            if 'Speed' in df_dist.columns and 'Distance_m' in df_dist.columns:
                try:
                    # Función interna para calcular tiempo acumulado por piloto
                    def calculate_time_from_speed(group):
                        # Ordenar por distancia asegurada
                        group = group.sort_values('Distance_m')
                        
                        # Velocidad en m/s (Speed viene en km/h)
                        v_ms = group['Speed'] / 3.6
                        
                        # Evitar división por cero (coche parado)
                        v_ms = v_ms.replace(0, 0.1)
                        
                        # Diferencia de distancia (debería ser 1m, pero calculamos por si acaso)
                        d_dist = group['Distance_m'].diff().fillna(0)
                        
                        # Tiempo por paso: dt = dx / v
                        dt = d_dist / v_ms
                        
                        # Tiempo acumulado
                        group['RelativeTime_s'] = dt.cumsum()
                        return group

                    # Aplicar a cada piloto por separado
                    df_dist = df_dist.groupby('Driver', group_keys=False).apply(calculate_time_from_speed)
                    
                except Exception as e:
                    st.warning(f"No se pudo reconstruir el tiempo desde la velocidad: {e}")

        data['telemetry_dist'] = df_dist
        
    # 3. Lap Features (Módulo 2)
    if (path / "lap_features_module2.csv").exists():
        data['laps'] = pd.read_csv(path / "lap_features_module2.csv")
        
    return data

@st.cache_data
def load_global_data():
    """Carga datos globales (PCA, Modelado) del root de ingestion."""
    data = {}
    
    # PCA Global
    pca_path = BASE_DATA_DIR / "pca_scores_global_norm.csv"
    if pca_path.exists():
        data['pca'] = pd.read_csv(pca_path)
    else:
        pca_path_raw = BASE_DATA_DIR / "pca_scores_global.csv"
        if pca_path_raw.exists():
            data['pca'] = pd.read_csv(pca_path_raw)
            
    # SHAP / Modelado
    shap_path = BASE_DATA_DIR / "shap" / "shap_mean_abs.csv"
    if shap_path.exists():
        data['shap_importance'] = pd.read_csv(shap_path)
        
    # Predicciones validación
    preds_path = BASE_DATA_DIR / "val_predictions_module4.csv"
    if preds_path.exists():
        data['predictions'] = pd.read_csv(preds_path)

    return data

# ==========================================
# INTERFAZ DE USUARIO
# ==========================================

st.title("🏎️ F1-Analytics: Dashboard de Ingeniería")
st.markdown("""
Esta aplicación visualiza el pipeline de procesamiento de **Industria 4.0** definido en el repositorio.
Desde la ingesta y alineación de señales hasta el modelado predictivo de tiempos de vuelta.
""")

# --- SIDEBAR: SELECCIÓN DE DATOS ---
st.sidebar.header("📁 Configuración de Sesión")

available_events = get_available_events()

if not available_events:
    st.error(f"No se encontraron datos en `{BASE_DATA_DIR}`. Por favor ejecuta `module1_ingestion.py` primero.")
    st.stop()

selected_event = st.sidebar.selectbox("Seleccionar Evento (Cache)", available_events)
session_data = load_session_data(selected_event)
global_data = load_global_data()

if 'laps' not in session_data:
    st.warning("Faltan archivos de 'features' (Módulo 2). Ejecuta `module2_signals.py`.")
else:
    drivers_list = session_data['laps']['Driver'].unique()
    
    st.sidebar.subheader("🆚 Comparativa")
    driver_1 = st.sidebar.selectbox("Piloto Referencia", drivers_list, index=0)
    idx_2 = 1 if len(drivers_list) > 1 else 0
    driver_2 = st.sidebar.selectbox("Piloto Comparación", drivers_list, index=idx_2)

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Telemetría & Ghost Car", 
    "🧪 Física & Neumáticos (M2)", 
    "🧬 Estilo de Conducción (PCA)", 
    "🤖 Modelado AI (M4)",
    "🛠️ Lab de IA & Despliegue"
])

# ==========================================
# TAB 1: TELEMETRÍA Y GHOST CAR
# ==========================================
with tab1:
    st.header("Análisis de Vuelta Rápida (Alineación Espacial)")
    
    if 'telemetry_dist' in session_data:
        df_dist = session_data['telemetry_dist']
        
        d1_data = df_dist[df_dist['Driver'] == driver_1].copy()
        d2_data = df_dist[df_dist['Driver'] == driver_2].copy()
        
        # Verificar si tenemos tiempo (reconstruido o real)
        can_calculate_delta = 'RelativeTime_s' in d1_data.columns and 'RelativeTime_s' in d2_data.columns
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if can_calculate_delta:
                try:
                    merged_ghost = pd.merge(
                        d1_data[['Distance_m', 'RelativeTime_s', 'Speed']], 
                        d2_data[['Distance_m', 'RelativeTime_s', 'Speed']], 
                        on='Distance_m', 
                        suffixes=(f'_{driver_1}', f'_{driver_2}')
                    )
                    
                    merged_ghost['Delta_s'] = merged_ghost[f'RelativeTime_s_{driver_2}'] - merged_ghost[f'RelativeTime_s_{driver_1}']
                    
                    fig_delta = px.line(merged_ghost, x='Distance_m', y='Delta_s', 
                                        title=f"⏱️ Ghost Car Delta (Tiempo reconstruido): {driver_2} vs {driver_1} (Ref)",
                                        labels={'Delta_s': f'Dif. Tiempo (s) - Negativo: {driver_2} más rápido'})
                    fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_delta.update_traces(line_color='white')
                    st.plotly_chart(fig_delta, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo calcular el Delta: {e}")
            else:
                st.info("⚠️ No se pudo reconstruir el tiempo. Mostrando solo velocidad.")
            
            if 'Speed' in d1_data.columns and 'Distance_m' in d1_data.columns:
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scatter(x=d1_data['Distance_m'], y=d1_data['Speed'], 
                                            mode='lines', name=driver_1, line=dict(color=TEAM_COLORS.get(driver_1, 'red'))))
                fig_speed.add_trace(go.Scatter(x=d2_data['Distance_m'], y=d2_data['Speed'], 
                                            mode='lines', name=driver_2, line=dict(color=TEAM_COLORS.get(driver_2, 'white'))))
                fig_speed.update_layout(title="Velocidad vs Distancia", xaxis_title="Distancia (m)", yaxis_title="Speed (km/h)")
                st.plotly_chart(fig_speed, use_container_width=True)
            else:
                st.error("Faltan columnas Speed/Distance.")

        with col2:
            st.markdown("### Datos Vuelta")
            if 'laps' in session_data:
                laps_df = session_data['laps']
                try:
                    l1 = laps_df[laps_df['Driver'] == driver_1]
                    l2 = laps_df[laps_df['Driver'] == driver_2]
                    
                    if not l1.empty and not l2.empty:
                        lap1 = l1.iloc[0]
                        lap2 = l2.iloc[0]
                        
                        st.metric(label=f"Tiempo {driver_1}", value=f"{lap1['LapTimeSeconds']:.3f} s")
                        st.metric(label=f"Tiempo {driver_2}", value=f"{lap2['LapTimeSeconds']:.3f} s", 
                                delta=f"{lap1['LapTimeSeconds'] - lap2['LapTimeSeconds']:.3f}")
                        
                        st.markdown("---")
                        st.markdown(f"**Neumático {driver_1}:** {lap1['Compound']} ({lap1['TyreLife']} vueltas)")
                        st.markdown(f"**Neumático {driver_2}:** {lap2['Compound']} ({lap2['TyreLife']} vueltas)")
                    else:
                        st.warning("Datos no disponibles para estos pilotos.")
                except IndexError:
                    st.warning("Error leyendo metadatos.")
            else:
                st.warning("Faltan metadatos de vueltas.")
    else:
        st.warning("No se encontró `telemetry_distance_aligned.csv`.")

# ==========================================
# TAB 2: FÍSICA Y NEUMÁTICOS
# ==========================================
with tab2:
    st.header("Dinámica Vehicular y Energía (Módulo 2)")
    
    if 'telemetry_time' in session_data:
        df_time = session_data['telemetry_time']
        has_physics = 'AX_long' in df_time.columns and 'TireEnergyProxy' in df_time.columns
        
        if has_physics:
            col_phy_1, col_phy_2 = st.columns(2)
            d1_phys = df_time[df_time['Driver'] == driver_1]
            d2_phys = df_time[df_time['Driver'] == driver_2]
            
            with col_phy_1:
                st.subheader("🔋 Energía de Neumático Acumulada")
                st.markdown("Integral de `(|Lat_G| + |Long_G|) * Speed`. Proxy de degradación.")
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(x=d1_phys['RelativeTime_s'], y=d1_phys['TireEnergyProxy'], 
                                                name=driver_1, line=dict(color=TEAM_COLORS.get(driver_1, 'red'))))
                fig_energy.add_trace(go.Scatter(x=d2_phys['RelativeTime_s'], y=d2_phys['TireEnergyProxy'], 
                                                name=driver_2, line=dict(color=TEAM_COLORS.get(driver_2, 'white'))))
                fig_energy.update_layout(xaxis_title="Tiempo (s)", yaxis_title="Energy Index (J/kg aprox)")
                st.plotly_chart(fig_energy, use_container_width=True)

            with col_phy_2:
                st.subheader("🎯 Círculo de Fricción (G-G Diagram)")
                fig_gg = go.Figure()
                fig_gg.add_trace(go.Scatter(x=d1_phys['AY_lat']/9.81, y=d1_phys['AX_long']/9.81, 
                                            mode='markers', name=driver_1, 
                                            marker=dict(size=4, color=TEAM_COLORS.get(driver_1, 'red'), opacity=0.5)))
                fig_gg.add_trace(go.Scatter(x=d2_phys['AY_lat']/9.81, y=d2_phys['AX_long']/9.81, 
                                            mode='markers', name=driver_2, 
                                            marker=dict(size=4, color=TEAM_COLORS.get(driver_2, 'white'), opacity=0.5)))
                fig_gg.update_layout(xaxis_title="Lateral G", yaxis_title="Longitudinal G",
                                     width=500, height=500, xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]))
                st.plotly_chart(fig_gg, use_container_width=True)

            st.subheader("🌊 Suavidad de Conducción (Jerk)")
            col_j1, col_j2 = st.columns(2)
            with col_j1:
                fig_jerk_long = px.box(df_time[df_time['Driver'].isin([driver_1, driver_2])], 
                                       x='Driver', y='Jerk_long', title="Distribución Jerk Longitudinal")
                st.plotly_chart(fig_jerk_long, use_container_width=True)
            with col_j2:
                 st.markdown("**Puntos de Brusquedad (Jerk > 5 m/s³)**")
                 fig_map = px.scatter(d1_phys, x='X', y='Y', color='Jerk_long', 
                                      color_continuous_scale='RdBu_r', title=f"Mapa de Jerk: {driver_1}")
                 fig_map.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False)
                 st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Falta física (AX_long, TireEnergyProxy). Revisa el Módulo 2.")
    else:
        st.warning("Falta `telemetry_time_10hz.csv`.")

# ==========================================
# TAB 3: PCA
# ==========================================
with tab3:
    st.header("Análisis de Componentes Principales (Módulo 3)")
    
    if 'pca' in global_data:
        df_pca = global_data['pca']
        st.markdown("El **PC1** suele capturar ritmo/energía, y el **PC2** estilo/agresividad.")
        
        all_drivers = df_pca['Driver'].unique()
        pca_drivers = st.multiselect("Filtrar Pilotos", all_drivers, default=all_drivers[:5] if len(all_drivers) > 0 else [])
        df_pca_filt = df_pca[df_pca['Driver'].isin(pca_drivers)]
        
        col_pca1, col_pca2 = st.columns([3, 1])
        with col_pca1:
            hover_cols = ['LapTimeSeconds', 'Event']
            if 'TyreLife' in df_pca_filt.columns: hover_cols.append('TyreLife')
            if 'Energy_per_m' in df_pca_filt.columns: hover_cols.append('Energy_per_m')

            if not df_pca_filt.empty:
                fig_pca = px.scatter(df_pca_filt, x='PC1', y='PC2', color='Driver', symbol='Compound',
                                     hover_data=hover_cols, title="Espacio Latente de Conducción", width=800, height=600)
                st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.info("Selecciona al menos un piloto.")
        with col_pca2:
            st.markdown("### Insights")
            st.info("- **PC1:** Ritmo/Gestión.\n- **PC2:** Agresividad.")
    else:
        st.warning("Faltan datos de PCA.")

# ==========================================
# TAB 4: MODELADO
# ==========================================
with tab4:
    st.header("Modelo Predictivo de LapTime (Módulo 4)")
    col_mod1, col_mod2 = st.columns(2)
    
    with col_mod1:
        st.subheader("Importancia de Features (SHAP)")
        if 'shap_importance' in global_data:
            df_shap = global_data['shap_importance']
            # Detección automática de columnas
            numeric_cols = df_shap.select_dtypes(include=[np.number]).columns
            text_cols = df_shap.select_dtypes(include=['object', 'string']).columns
            
            if len(numeric_cols) > 0:
                val_col = numeric_cols[0]
                feature_col = text_cols[0] if len(text_cols) > 0 else df_shap.columns[0]
                try:
                    df_sorted = df_shap.sort_values(val_col, ascending=True).tail(15)
                    fig_shap = px.bar(df_sorted, x=val_col, y=feature_col, orientation='h',
                                    title="Top 15 Variables (SHAP)", labels={val_col: 'Impacto medio'})
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error SHAP: {e}")
            else:
                st.warning("Error formato SHAP.")
        else:
            st.info("No hay datos SHAP.")
            
    with col_mod2:
        st.subheader("Precisión del Modelo")
        if 'predictions' in global_data:
            df_pred = global_data['predictions']
            fig_pred = px.scatter(df_pred, x='y_true', y='y_pred', color='SessionType',
                                  title="LapTime Real vs Predicho", labels={'y_true': 'Real', 'y_pred': 'Predicción'})
            min_val = min(df_pred['y_true'].min(), df_pred['y_pred'].min())
            max_val = max(df_pred['y_true'].max(), df_pred['y_pred'].max())
            fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="White", dash="dash"))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            mae = np.mean(np.abs(df_pred['y_true'] - df_pred['y_pred']))
            r2 = 1 - (np.sum((df_pred['y_true'] - df_pred['y_pred'])**2) / np.sum((df_pred['y_true'] - df_pred['y_true'].mean())**2))
            st.metric("MAE", f"{mae:.3f} s")
            st.metric("R2 Score", f"{r2:.4f}")
        else:
            st.info("No hay predicciones.")

# ==========================================
# TAB 5: LAB DE IA & DESPLIEGUE
# ==========================================
with tab5:
    st.header("🛠️ Laboratorio de Modelos & BentoML")
    st.markdown("Experimenta con hiperparámetros en tiempo real y prueba la API de predicción.")
    
    tab_train, tab_infer = st.tabs(["Entrenamiento Interactivo", "Inferencia Real-time (API)"])
    
    # --- SUB-TAB: TRAINING ---
    with tab_train:
        st.subheader("Ajuste de Hiperparámetros (Random Forest)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            n_estimators = st.slider("Número de Árboles (n_estimators)", 10, 500, 100, step=10)
        with c2:
            max_depth = st.slider("Profundidad Máxima (max_depth)", 3, 50, 10, step=1)
        with c3:
            st.markdown("### ") 
            train_btn = st.button("🚀 Entrenar Nuevo Modelo", type="primary")
            
        if train_btn:
            with st.spinner("Entrenando modelo..."):
                try:
                    # Llamada al módulo 4
                    params = {"n_estimators": n_estimators, "max_depth": max_depth}
                    res = module4_modeling.train_single_model("RandomForest", params, use_pcs=True)
                    
                    st.success(f"Modelo Entrenado! MAE: {res.mae:.4f} s | R2: {res.r2:.4f}")
                    
                    # Mostrar gráfico simple de pred vs real
                    fig_res = px.scatter(x=res.y_true, y=res.y_pred, 
                                       labels={'x': 'Real', 'y': 'Predicho'}, 
                                       title="Resultados Validación Cross-Validation (3-Fold)")
                    fig_res.add_shape(type="line", x0=res.y_true.min(), y0=res.y_true.min(),
                                    x1=res.y_true.max(), y1=res.y_true.max(),
                                    line=dict(color="White", dash="dash"))
                    st.plotly_chart(fig_res, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error en entrenamiento: {e}")

    # --- SUB-TAB: INFERENCE ---
    with tab_infer:
        st.subheader("Invocación a API BentoML")
        st.caption("Asegúrate de ejecutar `bentoml serve service:svc` en tu terminal.")
        
        # Formulario de entrada manual
        with st.form("inference_form"):
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                driver_id = st.selectbox("Piloto (ID)", ["1", "11", "16", "55", "44", "63", "14", "4"])
                comp = st.selectbox("Neumático", ["SOFT", "MEDIUM", "HARD"])
                tyre_life = st.number_input("Vida Neumático", 1, 50, 5)
                
            with col_i2:
                avg_speed = st.number_input("Velocidad Media (m/s)", 30.0, 70.0, 55.0)
                brake_agg = st.number_input("Agresividad Freno", 0.0, 20.0, 5.0)
                energy = st.number_input("Energy Index", 100.0, 2000.0, 500.0)
                
            with col_i3:
                pc1 = st.slider("PC1 (Ritmo)", -5.0, 5.0, 0.0)
                pc2 = st.slider("PC2 (Estilo)", -5.0, 5.0, 0.0)
                
            submit_api = st.form_submit_button("📡 Enviar a API")
            
        if submit_api:
            payload = {
                "Driver": driver_id,
                "Compound": comp,
                "TyreLife": tyre_life,
                "Avg_Speed_mps": avg_speed,
                "Brake_Aggression": brake_agg,
                "Energy_Index": energy,
                "PC1": pc1,
                "PC2": pc2,
                # Valores default para el resto
                "SessionType": "R",
                "Avg_Throttle": 40.0,
                "MeanAbs_Jerk_Long": 2.0,
                "MeanAbs_Jerk_Lat": 2.0,
                "Max_Lateral_g": 3.0,
                "Max_Longitudinal_g": 2.0,
                "PC3": 0.0
            }
            
            try:
                # URL local de BentoML por defecto
                # BentoML espera que el key del JSON coincida con el nombre del argumento de la funcion ('input_data')
                wrapped_payload = {"input_data": payload}
                response = requests.post("http://localhost:3000/predict_laptime", json=wrapped_payload, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"⏱️ Predicción API: {data.get('predicted_laptime_s', 'N/A'):.3f} segundos")
                else:
                    st.error(f"Error API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"No se pudo conectar con BentoML. ¿Está corriendo el servidor? Error: {e}")
                st.code("bentoml serve service:svc", language="bash")


# Footer
st.markdown("---")
st.caption("F1-Analytics Dashboard | Datos generados por FastF1 y procesados localmente.")