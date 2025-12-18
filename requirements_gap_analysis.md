# Análisis de Cumplimiento de Requerimientos (Gap Analysis)

Este documento compara el estado actual del proyecto `F1-Analytics` con las instrucciones oficiales de la asignatura "Analítica de Datos para la Industria".

## 🔴 Puntos Críticos Faltantes (Prioridad Alta)

### 1. Despliegue con BentoML (0% Implementado)
*   **Requisito:** "Uso de BentoML para empaquetar el modelo... y publicar una API (en local)... probarse dentro del flujo... a través de un formulario".
*   **Estado Actual:** No existe ninguna referencia a `bentoml` en el código.
*   **Acción Necesaria:**
    *   Crear archivo `service.py` con BentoML.
    *   Modificar `module4_modeling.py` para guardar el modelo en formato BentoML (`bentoml.sklearn.save_model`).
    *   Añadir en `app.py` un formulario que envíe datos (POST) a la API de BentoML y muestre la respuesta.

### 2. Entrenamiento Interactivo en Streamlit (Parcialmente Implementado)
*   **Requisito:** "Ofrezca la posibilidad de **entrenar** y evaluar modelos... de forma interactiva... integrando sliders/selectores".
*   **Estado Actual:** `app.py` visualiza resultados *pre-calculados* (lee CSVs). No permite al usuario cambiar hiperparámetros (ej: número de árboles, learning rate) y re-entrenar el modelo en vivo.
*   **Acción Necesaria:**
    *   Añadir en `app.py` (Tab 4) controles (`st.slider`, `st.selectbox`) para hiperparámetros.
    *   Integrar la lógica de entrenamiento de `module4` directamente en `app.py` (o llamarla) para generar nuevos gráficos al pulsar un botón "Entrenar".

---

## 🟡 Puntos a Revisar (Prioridad Media)

### 3. Notebook de Documentación
*   **Requisito:** "Justificar proceso... descripción del dataset... decisiones adoptadas".
*   **Estado Actual:** Existe `notebooks/notebook.ipynb`.
*   **Acción:** Verificar que este notebook contenga **texto explicativo** (Markdown) y no solo código. Debe contar la "historia" de la limpieza y decisiones.

### 4. Ingeniería de Características en UI
*   **Requisito:** "Visualizarse el impacto de las transformaciones... ingeniería de características... desde los componentes interactivos".
*   **Estado Actual:** Muestras PCA y Física, lo cual es muy positivo.
*   **Acción:** Asegurar que quede claro en `app.py` cómo las variables `Jerk` o `Energy` mejoran el modelo (el gráfico SHAP actual cumple esto parcialmente, pero podría ser más explícito).

---

## 🟢 Puntos Cumplidos (Fortalezas)

*   **Visualización de Datos:** Excelente. El uso de trazas, mapas de calor y gráficos dinámicos cumple sobradamente con "métodos de visualización... claridad, precisión".
*   **Preparación de Datos:** El pipeline `module1` -> `module2` está muy bien estructurado (Clean Code).
*   **Conexión Teórica:** El enfoque de "Industria 4.0" y "Gemelo Digital" le da un valor añadido fuerte de originalidad.

---

## Plan de Acción Recomendado (Hoja de Ruta)

1.  **Integrar BentoML:** Crear un servicio básico que reciba features de una vuelta y prediga el tiempo.
2.  **Actualizar Streamlit:** Añadir una sección "Simulación en Tiempo Real" que consuma esa API.
3.  **Refinar Notebook:** Asegurar que el notebook cuenta la narrativa completa.
