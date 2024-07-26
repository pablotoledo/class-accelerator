import streamlit as st
import os
import tempfile
import subprocess

st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("Environment variables")
variable1 = st.sidebar.text_input("Variable 1", "Valor predeterminado")
st.sidebar.write("Class - Accelerator")

# Captura dinámica del número de threads
num_threads = st.sidebar.slider("Número de threads", min_value=1, max_value=os.cpu_count(), value=4)

# Contenido principal
st.title("Class Accelerator")

# Función para procesar el archivo subido
def process_uploaded_file(uploaded_file, threads):
    if uploaded_file is not None:
        file_details = {
            "Nombre del archivo": uploaded_file.name,
            "Tipo de archivo": uploaded_file.type,
            "Tamaño": f"{uploaded_file.size} bytes"
        }
        st.write("Detalles del archivo:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        # Extraer audio
        with st.spinner('Extrayendo audio del video...'):
            # Crear archivos temporales para el video y el audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                
                tmp_video_file.write(uploaded_file.getvalue())
                tmp_video_path = tmp_video_file.name
                tmp_audio_path = tmp_audio_file.name

            # Usar FFmpeg para extraer el audio
            command = [
                'ffmpeg',
                '-i', tmp_video_path,
                '-acodec', 'pcm_s16le',
                '-ac', '2',
                '-ar', '44100',
                '-threads', str(threads),  # Utilizar el número de threads seleccionado
                tmp_audio_path
            ]
            
            try:
                subprocess.run(command, check=True, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                st.error(f"Error al extraer el audio: {e.stderr.decode()}")
                return
            
            # Leer el archivo de audio
            with open(tmp_audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Limpiar archivos temporales
            os.unlink(tmp_video_path)
            os.unlink(tmp_audio_path)
        
        st.success('Audio extraído con éxito!')
        
        # Reproducir audio
        st.audio(audio_bytes, format='audio/wav')
        
        # Botón de descarga
        st.download_button(
            label="Descargar audio WAV",
            data=audio_bytes,
            file_name="extracted_audio.wav",
            mime="audio/wav"
        )
    else:
        st.write("Por favor, sube un archivo mp4.")

# Área de carga de archivos
uploaded_file = st.file_uploader("Arrastra y suelta un archivo aquí o haz clic para seleccionar",
                                 type=["mp4"],  # Solo permite archivos mp4
                                 accept_multiple_files=False)  # Solo permite un archivo

# Procesar el archivo
if uploaded_file is not None:
    if st.button("Convertir archivo a audio"):
        process_uploaded_file(uploaded_file, num_threads)
