#python transcribe_videos.py /ruta/al/directorio es small
import os
import argparse
import subprocess
import tempfile
import whisper
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        audio_path
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

def transcribe_audio(audio_path, language, model_size):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language)
    return result["text"]

def process_directory(directory, language, model_size):
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc=f"Processing files in {root}", unit="file"):
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"\nProcessing: {video_path}")
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                
                try:
                    print("Extracting audio...")
                    extract_audio(video_path, temp_audio_path)
                    
                    print(f"Transcribing with Whisper (Language: {language}, Model: {model_size})...")
                    transcription = transcribe_audio(temp_audio_path, language, model_size)
                    
                    output_file = os.path.join(root, f"whisper-{language}-{model_size}-{os.path.splitext(file)[0]}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    
                    print(f"Transcription saved: {output_file}")
                
                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                
                finally:
                    os.unlink(temp_audio_path)

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP4 files in a directory using Whisper")
    parser.add_argument("directory", help="Directory to search for MP4 files")
    parser.add_argument("language", help="Language for Whisper transcription")
    parser.add_argument("model_size", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    
    args = parser.parse_args()
    
    process_directory(args.directory, args.language, args.model_size)

if __name__ == "__main__":
    main()