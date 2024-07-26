import streamlit as st

# Configuración de la página
st.set_page_config(layout="wide")

# Títulos de la aplicación
st.title("Mi Aplicación en Streamlit")
st.sidebar.title("Opciones de Navegación")

# Opciones de navegación en el sidebar
option = st.sidebar.selectbox(
    "Seleccione una página",
    ("Página Principal", "Visualizador de Datos", "Preferencias del Usuario")
)

# Contenido de la página principal
if option == "Página Principal":
    st.header("Bienvenido a la Página Principal")
    st.write("Aquí va el contenido de la página principal.")

# Contenido del visualizador de datos
elif option == "Visualizador de Datos":
    st.header("Visualizador de Datos")
    st.write("Aquí va el contenido del visualizador de datos.")
    # Añade aquí los elementos de la interfaz necesarios para el visualizador de datos

# Contenido de las preferencias del usuario
elif option == "Preferencias del Usuario":
    st.header("Preferencias del Usuario")
    st.write("Aquí van las preferencias del usuario.")
    # Añade aquí los elementos de la interfaz necesarios para las preferencias del usuario

# Pie de página
st.sidebar.write("© 2024 Mi Empresa")

