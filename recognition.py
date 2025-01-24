import sys
import whisper
import os

# Завантажуємо модель Whisper
model = whisper.load_model("medium")

def transcribe_audio(file_path, language="uk"):
    # Перевіряємо, чи файл існує
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не знайдено")
    
    # Виконуємо транскрипцію
    result = model.transcribe(file_path, language=language, fp16=False)
    return result['text']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Помилка: Не передано шлях до аудіофайлу.")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    try:
        transcription = transcribe_audio(audio_file_path)
        print(transcription)  # Виводимо результат для Electron
    except Exception as e:
        print(f"Помилка: {e}")
        sys.exit(1)
