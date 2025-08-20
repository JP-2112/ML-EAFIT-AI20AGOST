import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

# ---------------------
# 1. Generar datos simulados
# ---------------------
@st.cache_data
def generar_datos(n=300, random_state=42):
    np.random.seed(random_state)
    data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n),
        "feature2": np.random.normal(5, 2, n),
        "feature3": np.random.randint(0, 100, n),
        "feature4": np.random.binomial(1, 0.5, n),
        "feature5": np.random.uniform(0, 10, n),
        "feature6": np.random.poisson(3, n),
    })
    data["target"] = (data["feature1"] + data["feature2"] + np.random.randn(n)) > 5
    data["target"] = data["target"].astype(int)
    return data

# ---------------------
# 2. Función para colores aleatorios (evitando negro)
# ---------------------
def random_color():
    colores = ["red", "blue", "green", "orange", "purple", "brown", "pink", "teal"]
    return random.choice(colores)

# Estado inicial
if "color" not in st.session_state:
    st.session_state.color = "blue"

# Botón para cambiar color
if st.button("🎨 Cambiar color de estilo"):
    st.session_state.color = random_color()

# ---------------------
# 3. Estilos CSS dinámicos
# ---------------------
st.markdown(
    f"""
    <style>
    /* Cambiar color de todos los textos */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown {{
        color: {st.session_state.color} !important;
    }}
    /* Líneas horizontales de adorno */
    hr {{
        border: 2px solid {st.session_state.color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# 4. Interfaz en Streamlit
# ---------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.title("📊 Demo de ML Supervisado con Datos Simulados o CSV")
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------
# 5. Cargar archivo CSV opcional
# ---------------------
archivo = st.file_uploader("📂 Cargar archivo CSV", type=["csv"])

if archivo is not None:
    datos = pd.read_csv(archivo)
    st.success("✅ Archivo cargado correctamente")
else:
    datos = generar_datos()
    st.info("ℹ️ No se cargó ningún archivo, usando datos simulados")

# Mostrar dataset organizado
st.subheader("Vista previa de los datos")
st.dataframe(datos.head(10))  # organizado con tabla interactiva

# ---------------------
# 6. Preparar datos y entrenamiento
# ---------------------
if "target" not in datos.columns:
    st.error("❌ El dataset cargado no tiene la columna 'target'. Agrega una columna de objetivo para entrenar el modelo.")
else:
    test_size = st.slider("Proporción de datos de validación", 0.1, 0.5, 0.3, 0.05)

    X = datos.drop("target", axis=1)
    y = datos["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write(f"**Tamaño entrenamiento:** {X_train.shape[0]} | **Tamaño validación:** {X_test.shape[0]}")

    # ---------------------
    # 7. Entrenar modelo simple
    # ---------------------
    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    st.subheader("Resultados del modelo (Logistic Regression)")
    st.text(classification_report(y_test, y_pred))

    # ---------------------
    # 8. Visualización
    # ---------------------
    st.subheader("Matriz de confusión")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
