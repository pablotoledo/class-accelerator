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

def process_uploaded_file(uploaded_file, threads):
    if uploaded_file is not None:
        file_details = {
            "Nombre del archivo": uploaded_file.name,
            "Tipo de archivo": uploaded_file.type,
            "Tamaño": f"{uploaded_file.size} bytes"
        }
        st.write("Detalles del archivo:")
        with st.expander("Mostrar detalles del archivo"):
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")

        with st.spinner('Extrayendo audio del video...'):
            # Crear archivos temporales para el video y el audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file:
                tmp_video_file.write(uploaded_file.getvalue())
                tmp_video_path = tmp_video_file.name

            tmp_audio_path = tempfile.mktemp(suffix='.wav')

            # Usar FFmpeg para extraer el audio
            command = [
                'ffmpeg',
                '-i', tmp_video_path,
                '-vn',  # Ignorar el video
                '-acodec', 'pcm_s16le',  # Codec para WAV
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Número de canales
                '-threads', str(threads),
                tmp_audio_path
            ]
            
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                with st.expander("Mostrar la salida de FFmpeg para diagnóstico"):
                    st.text(result.stderr)
            except subprocess.CalledProcessError as e:
                st.error(f"Error al extraer el audio: {e.stderr}")
                return
            
            # Verificar que el archivo de audio se ha creado y tiene un tamaño > 0
            if not os.path.exists(tmp_audio_path) or os.path.getsize(tmp_audio_path) == 0:
                st.error("Error: El archivo de audio no se creó correctamente.")
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