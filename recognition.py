import sys
import whisper
import os
import wave
from pydub import AudioSegment
import json

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def format_time(seconds):
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

def transcribe_audio(file_path, model_name, language="uk", char_limit=50, device="cpu"):
    # Встановлюємо шлях до моделей Whisper
    os.environ["WHISPER_MODELS_DIR"] = "models"
    # Завантажуємо модель Whisper
    model = whisper.load_model(model_name, device=device)
    # Конвертуємо аудіофайл у формат WAV
    wav_path = convert_to_wav(file_path)
    
    print("Статус обробки: початок", flush=True)
    
    # Розділяємо аудіофайл на менші частини
    with wave.open(wav_path, 'rb') as wf:
        total_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        chunk_size = frame_rate * 10  # 10 секундні частини
        num_chunks = total_frames // chunk_size + 1

        transcription = []
        for i in range(num_chunks):
            start_frame = i * chunk_size
            wf.setpos(start_frame)
            frames = wf.readframes(chunk_size)
            
            # Зберігаємо тимчасовий файл
            temp_file_path = f"temp_chunk_{i}.wav"
            with wave.open(temp_file_path, 'wb') as temp_wf:
                temp_wf.setnchannels(wf.getnchannels())
                temp_wf.setsampwidth(wf.getsampwidth())
                temp_wf.setframerate(frame_rate)
                temp_wf.writeframes(frames)
            
            # Виконуємо транскрипцію
            result = model.transcribe(temp_file_path, language=language, fp16=False)
            for segment in result['segments']:
                transcription.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "time": f"{format_time(segment['start'])} - {format_time(segment['end'])}",
                    "text": segment['text']
                })
            
            # Видаляємо тимчасовий файл
            os.remove(temp_file_path)
            
            # Відправляємо прогрес
            progress = (i + 1) / num_chunks * 100
            print(f"Прогрес: {progress:.2f}%", flush=True)
    
    print("Статус обробки: завершено", flush=True)
    os.remove(wav_path)  # Видаляємо тимчасовий WAV файл
    return transcription

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(json.dumps({"error": "Помилка: Не передано всі необхідні параметри."}, ensure_ascii=False))
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3]
    char_limit = int(sys.argv[4])
    device = sys.argv[5]
    
    try:
        transcription = transcribe_audio(audio_file_path, model_name, language, char_limit, device)
        print(json.dumps(transcription, ensure_ascii=False))  # Виводимо результат для Electron
    except Exception as e:
        print(json.dumps({"error": f"Помилка: {e}"}, ensure_ascii=False))
        sys.exit(1)