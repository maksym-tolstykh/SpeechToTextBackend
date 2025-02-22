import sys, io
import whisper
import os
import json
import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def safe_print_json(data):
    try:
        print(json.dumps(data), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Помилка JSON: {str(e)}"}), flush=True)

def format_time(seconds):
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

def detect_language(model, file_path):
    result = model.transcribe(file_path, task="detect-language")
    return result['language']


def transcribe_audio(file_path, model_name, language="uk", device="cpu"):
    try:
        # Встановлюємо шлях до моделей Whisper
        os.environ["WHISPER_MODELS_DIR"] = os.path.abspath("models")  # Примусово вказуємо шлях
        safe_print_json({"progress":"Завантаження моделі розпізнавання аудіо.."})
        # Завантажуємо модель Whisper
        model = whisper.load_model(model_name, device=device, download_root=os.environ["WHISPER_MODELS_DIR"])
        
        
        
    # Автоматичне розпізнавання мови, якщо вибрано "auto"
        if language == "auto":
            safe_print_json({"progress":"Автоматичне розпізнавання мови.."})
            language = detect_language(model, file_path)
            safe_print_json({"language":f"{language}" })
        
        
     
        # Виконуємо транскрипцію
        safe_print_json({"progress":"Транскрибування аудіо.."})
        result = model.transcribe(file_path, language=language, fp16=True)
        
        transcription = []
        for segment in result['segments']:
            transcription.append({
                "start": segment['start'],
                "end": segment['end'],
                "time": f"{format_time(segment['start'])} - {format_time(segment['end'])}",
                "text": segment['text']
            })

        
        safe_print_json({"status": "завершено"})
        # os.remove(file_path)  # Видаляємо тимчасовий WAV файл
        return transcription
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 5:
        safe_print_json({"error": "Помилка: Не передано всі необхідні параметри."})
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3]
    device = sys.argv[4]
    
    try:
        transcription = transcribe_audio(audio_file_path, model_name, language, device)
        safe_print_json(transcription) # Виводимо результат для Electron
    except Exception as e:
        safe_print_json({"error": f"Помилка: {e}"})
        sys.exit(1)