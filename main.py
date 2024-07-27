import streamlit as st
import os
import tempfile
import subprocess
import whisper
import psutil
import threading
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(layout="wide")

# Inicializar el estado de la sesión si no existe
if 'audio_extracted' not in st.session_state:
    st.session_state.audio_extracted = False
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Sidebar
st.sidebar.title("Configuración")

st.sidebar.write("**Extracción de Audio**")

# Captura dinámica del número de threads
num_threads = st.sidebar.slider("Número de threads", min_value=1, max_value=os.cpu_count(), value=4)

st.sidebar.divider()

# Whisper Speech-to-Text
st.sidebar.write("**Speech-to-Text**")

# Selección del modelo de Whisper
whisper_model = st.sidebar.selectbox("Modelo de Whisper", ["tiny", "base", "small", "medium", "large"], index=2)

# Selección de idioma de Whisper
language = st.sidebar.selectbox("Idioma de Whisper", ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"], index=1)

# Sumarización de texto
st.sidebar.divider()
st.sidebar.write("**Sumarización de Texto**")
summary_model = st.sidebar.selectbox("Modelo de Resumen", ["dolphin-2.8-mistral-7b-v02 32k", "llama3"], index=0)

# Contenido principal
st.title("Class Accelerator")

def summarize_text(text, model_name):
    if model_name == "dolphin-2.8-mistral-7b-v02 32k":
        model_path = "cognitivecomputations/dolphin-2.8-mistral-7b-v02"
    elif model_name == "llama3":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error al cargar el modelo o generar el resumen: {str(e)}")
        return None

def monitor_resources(stop_event, progress_bar):
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        progress_bar.progress(min(cpu_percent / 100, 1.0))
        time.sleep(1)

def process_uploaded_file_mp4(uploaded_file_mp4, threads):
    if uploaded_file_mp4 is not None:
        file_details = {
            "Nombre del archivo": uploaded_file_mp4.name,
            "Tipo de archivo": uploaded_file_mp4.type,
            "Tamaño": f"{uploaded_file_mp4.size} bytes"
        }
        st.write("Detalles del archivo:")
        with st.expander("Mostrar detalles del archivo"):
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")

        with st.spinner('Extrayendo audio del video...'):
            # Crear archivos temporales para el video y el audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file:
                tmp_video_file.write(uploaded_file_mp4.getvalue())
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
uploaded_file_mp4 = st.file_uploader("Arrastra y suelta un archivo aquí o haz clic para seleccionar",
                                 type=["mp4"],  # Solo permite archivos mp4
                                 accept_multiple_files=False)  # Solo permite un archivo

# Procesar el archivo
if uploaded_file_mp4 is not None and not st.session_state.audio_extracted:
    if st.button("Convertir archivo a audio"):
        process_uploaded_file_mp4(uploaded_file_mp4, num_threads)

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
                result = model.transcribe(tmp_audio_path, language=language)

                # Guardar la transcripción en el estado de la sesión
                st.session_state.transcription = result["text"]

                # Detener el monitor de recursos
                stop_event.set()
                monitor_thread.join()

                # Mostrar la transcripción
                with st.expander("Mostrar transcripción"):
                    st.subheader("Transcripción:")
                    st.write(st.session_state.transcription)
                st.download_button(
                    label="Descargar transcripción como .txt",
                    data=st.session_state.transcription,
                    file_name="transcripcion.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error durante la transcripción: {str(e)}")
            finally:
                # Limpiar el archivo temporal
                os.unlink(tmp_audio_path)

# Botón para generar resumen (fuera del bloque de transcripción)
if st.session_state.transcription:
    if st.button("Generar Resumen"):
        with st.spinner(f'Generando resumen con el modelo {summary_model}...'):
            try:
                summary = summarize_text(st.session_state.transcription, summary_model)
                st.session_state.summary = summary

                with st.expander("Mostrar resumen"):
                    st.subheader("Resumen:")
                    st.write(st.session_state.summary)
                
                st.download_button(
                    label="Descargar resumen como .txt",
                    data=st.session_state.summary,
                    file_name="resumen.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error durante la generación del resumen: {str(e)}")

# Mostrar el resumen si ya está generado
if st.session_state.summary:
    with st.expander("Mostrar resumen"):
        st.subheader("Resumen:")
        st.write(st.session_state.summary)
    
    st.download_button(
        label="Descargar resumen como .txt",
        data=st.session_state.summary,
        file_name="resumen.txt",
        mime="text/plain"
    )