import streamlit as st
import os
import tempfile
import subprocess
import whisper
import psutil
import threading
import time

st.set_page_config(layout="wide")

# Inicializar el estado de la sesión si no existe
if 'audio_extracted' not in st.session_state:
    st.session_state.audio_extracted = False
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# Sidebar
st.sidebar.title("Environment variables")
variable1 = st.sidebar.text_input("Variable 1", "Valor predeterminado")
st.sidebar.write("Class - Accelerator")

# Captura dinámica del número de threads
num_threads = st.sidebar.slider("Número de threads", min_value=1, max_value=os.cpu_count(), value=4)

# Selección del modelo de Whisper
whisper_model = st.sidebar.selectbox("Modelo de Whisper", ["small", "medium", "large"])

# Contenido principal
st.title("Class Accelerator")

def monitor_resources(stop_event, progress_bar):
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        progress_bar.progress(min(cpu_percent / 100, 1.0))
        time.sleep(1)

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
                st.session_state.audio_bytes = audio_file.read()
           
            # Limpiar archivos temporales
            os.unlink(tmp_video_path)
            os.unlink(tmp_audio_path)

        st.session_state.audio_extracted = True
        st.experimental_rerun()

# Área de carga de archivos
uploaded_file = st.file_uploader("Arrastra y suelta un archivo aquí o haz clic para seleccionar",
                                 type=["mp4"],  # Solo permite archivos mp4
                                 accept_multiple_files=False)  # Solo permite un archivo

# Procesar el archivo
if uploaded_file is not None and not st.session_state.audio_extracted:
    if st.button("Convertir archivo a audio"):
        process_uploaded_file(uploaded_file, num_threads)

# Mostrar audio y botón de transcripción si el audio ha sido extraído
if st.session_state.audio_extracted:
    st.success('Audio extraído con éxito!')
    
    # Reproducir audio
    st.audio(st.session_state.audio_bytes, format='audio/wav')
    
    # Botón de descarga
    st.download_button(
        label="Descargar audio WAV",
        data=st.session_state.audio_bytes,
        file_name="extracted_audio.wav",
        mime="audio/wav"
    )

    # Realizar speech-to-text
    if st.button("Transcribir audio"):
        with st.spinner(f'Transcribiendo audio con el modelo {whisper_model}...'):
            # Guardar el audio en un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio_file:
                tmp_audio_file.write(st.session_state.audio_bytes)
                tmp_audio_path = tmp_audio_file.name

            # Configurar la barra de progreso y el monitor de recursos
            progress_bar = st.progress(0)
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, progress_bar))
            monitor_thread.start()

            try:
                # Cargar el modelo de Whisper
                model = whisper.load_model(whisper_model)

                # Realizar la transcripción
                result = model.transcribe(tmp_audio_path)

                # Detener el monitor de recursos
                stop_event.set()
                monitor_thread.join()

                # Mostrar la transcripción
                st.subheader("Transcripción:")
                st.write(result["text"])
            except Exception as e:
                st.error(f"Error durante la transcripción: {str(e)}")
            finally:
                # Limpiar el archivo temporal
                os.unlink(tmp_audio_path)