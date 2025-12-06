"""
Streamlit app para explorar el pipeline F1 Analytics.

Se apoya en los artefactos generados en feature_extraction/data/module1_ingestion:
- all_lap_features.csv (dataset consolidado multiaño)
- pca_scores_global_norm.csv (PCA global normalizado)
- val_predictions_module4.csv + model_metrics_module4.json (modelado)
- shap_mean_abs.csv (opcional, interpretabilidad)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# Rutas base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "feature_extraction" / "data" / "module1_ingestion"


@st.cache_data(show_spinner=False)
def load_laps() -> pd.DataFrame:
    path = DATA_DIR / "all_lap_features.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_pca_scores() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "pca_scores_global_norm.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_model_metrics() -> Optional[dict]:
    path = DATA_DIR / "model_metrics_module4.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data(show_spinner=False)
def load_val_preds() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "val_predictions_module4.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_shap_mean_abs() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "shap" / "shap_mean_abs.csv"
    if path.exists():
        return pd.read_csv(path, header=None, names=["feature", "mean_abs"])
    return None


def style_app() -> None:
    st.set_page_config(page_title="F1 Analytics", layout="wide")
    st.markdown(
        """
        <style>
        body {background: #0b1021;}
        .block-container {padding: 2rem 2.5rem;}
        h1,h2,h3,h4 {color: #e4e7ef;}
        p, li, span, label {color: #cfd3e2;}
        .metric {background: linear-gradient(135deg,#192140,#0f162d); padding: 1rem; border-radius: 10px; border:1px solid #1f2a48;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    years = sorted(df["Year"].unique())
    events = sorted(df["Event"].unique())
    compounds = sorted(df["Compound"].unique())
    drivers = sorted(df["Driver"].unique())

    year_sel = st.sidebar.multiselect("Años", years, default=years)
    event_sel = st.sidebar.multiselect("Eventos", events, default=events[:5])
    comp_sel = st.sidebar.multiselect("Compound", compounds, default=compounds)
    driver_sel = st.sidebar.multiselect("Pilotos (ID FIA)", drivers, default=drivers)

    filt = df[
        df["Year"].isin(year_sel)
        & df["Event"].isin(event_sel)
        & df["Compound"].isin(comp_sel)
        & df["Driver"].isin(driver_sel)
    ]
    return filt


def kpi_cards(df: pd.DataFrame, metrics: Optional[dict]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric'>Vueltas filtradas</div>", unsafe_allow_html=True)
        st.metric("Vueltas", f"{len(df):,}")
    with col2:
        st.markdown("<div class='metric'>LapTime medio</div>", unsafe_allow_html=True)
        st.metric("LapTime (s)", f"{df['LapTimeSeconds'].mean():.2f}")
    with col3:
        st.markdown("<div class='metric'>Energy_per_m medio</div>", unsafe_allow_html=True)
        if "Energy_per_m" in df.columns:
            st.metric("Energy/m", f"{df['Energy_per_m'].mean():.0f}")
        else:
            st.metric("Energy/m", "N/A")
    with col4:
        best = metrics.get("best_model") if metrics else None
        st.markdown("<div class='metric'>Mejor modelo (CV)</div>", unsafe_allow_html=True)
        if best:
            st.metric(best["name"], f"MAE {best['MAE_mean']:.2f}s")
        else:
            st.metric("Modelo", "N/D")


def plot_laptime_distribution(df: pd.DataFrame) -> None:
    fig = px.violin(
        df,
        x="Year",
        y="LapTimeSeconds",
        color="Compound",
        box=True,
        points=False,
        title="Distribución de LapTime por año y compound",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_energy_vs_laptime(df: pd.DataFrame) -> None:
    fig = px.scatter(
        df,
        x="Energy_Index",
        y="LapTimeSeconds",
        color="Compound",
        hover_data=["Event", "LapNumber", "TyreLife"],
        trendline="ols",
        title="LapTime vs Energy_Index (proxy de demanda)",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_pca(pca_df: pd.DataFrame) -> None:
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Compound",
        hover_data=["Driver", "Event", "Year"],
        title="PCA global normalizado (PC1 vs PC2)",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_residuals(preds: pd.DataFrame) -> None:
    preds = preds.copy()
    preds["resid"] = preds["y_true"] - preds["y_pred"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            preds,
            x="resid",
            nbins=50,
            marginal="box",
            title="Histograma de residuos",
            color_discrete_sequence=["#6c8ef7"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Compound" in preds.columns:
            fig2 = px.box(
                preds,
                x="Compound",
                y="resid",
                points="all",
                title="Residuos por Compound",
                color_discrete_sequence=["#8892bf"],
            )
            st.plotly_chart(fig2, use_container_width=True)


def plot_r2_bar(metrics: Optional[dict]) -> None:
    if not metrics:
        st.info("No se encontró model_metrics_module4.json")
        return
    metrics_no_best = {k: v for k, v in metrics.items() if k != "best_model"}
    mdf = pd.DataFrame(metrics_no_best).T.astype(float).reset_index().rename(columns={"index": "Model"})
    fig = px.bar(
        mdf,
        x="Model",
        y="R2_mean",
        title="R2 (promedio CV) por modelo",
        color="R2_mean",
        color_continuous_scale="Blues",
        range_y=[0, 1],
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def plot_shap(shap_df: pd.DataFrame) -> None:
    shap_df = shap_df.sort_values("mean_abs", ascending=False).head(15)
    fig = px.bar(
        shap_df,
        x="mean_abs",
        y="feature",
        orientation="h",
        title="SHAP | Importancia media absoluta (top 15)",
        color="mean_abs",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    style_app()
    st.title("🏎️ F1 Analytics – Telemetría, PCA y Modelado")
    st.caption("Explora vueltas, proxies de energía, PCA global y desempeño del modelo (MAE/RMSE/R2).")

    laps = load_laps()
    pca_df = load_pca_scores()
    metrics = load_model_metrics()
    preds = load_val_preds()
    shap_mean_abs = load_shap_mean_abs()

    filt_laps = sidebar_filters(laps)
    kpi_cards(filt_laps, metrics)

    st.subheader("Distribuciones y relaciones clave")
    plot_laptime_distribution(filt_laps)
    plot_energy_vs_laptime(filt_laps)

    if pca_df is not None:
        st.subheader("PCA global normalizado")
        plot_pca(pca_df)
    else:
        st.info("No se encontró pca_scores_global_norm.csv; ejecuta module3_pca_global_normalized.py.")

    if preds is not None:
        st.subheader("Desempeño del modelo (val. cruzada)")
        plot_residuals(preds)
        plot_r2_bar(metrics)
    else:
        st.info("No se encontró val_predictions_module4.csv; ejecuta module4_modeling.py.")

    if shap_mean_abs is not None:
        st.subheader("Interpretabilidad (SHAP)")
        plot_shap(shap_mean_abs)
    else:
        st.info("No se encontraron artefactos SHAP; instala shap y vuelve a correr module4_modeling.py.")

    st.markdown("---")
    st.markdown(
        "Tip: ajusta filtros en el sidebar para comparar eventos, pilotos y compuestos. "
        "La barra R2 muestra estabilidad entre modelos; usa SHAP para ver qué variables empujan el tiempo de vuelta."
    )


if __name__ == "__main__":
    main()
