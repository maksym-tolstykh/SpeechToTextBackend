import sys, io
import whisper
import os
import wave
from pydub import AudioSegment
import json
from moviepy import VideoFileClip, TextClip, CompositeVideoClip

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, fps=16000)
    return audio_path

def format_time(seconds):
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

def detect_language(model, file_path):
    result = model.transcribe(file_path, task="detect-language")
    return result['language']


def transcribe_audio(file_path, model_name, language="uk", device="cuda"):
    try:
        # Встановлюємо шлях до моделей Whisper
        os.environ["WHISPER_MODELS_DIR"] = os.path.abspath("models")  # Примусово вказуємо шлях
        print(json.dumps({"progress":"Завантаження моделі розпізнавання аудіо.."}), flush=True)
        # Завантажуємо модель Whisper
        model = whisper.load_model(model_name, device=device, download_root=os.environ["WHISPER_MODELS_DIR"])
        
        

   
        wav_path = convert_to_wav(file_path)
        
        
    # Автоматичне розпізнавання мови, якщо вибрано "auto"
        if language == "auto":
            print(json.dumps({"progress":"Автоматичне розпізнавання мови.."}), flush=True)
            language = detect_language(model, wav_path)
            print(json.dumps({"language":f"{language}" }), flush=True)
        
        
        # Розділяємо аудіофайл на менші частини
        with wave.open(wav_path, 'rb') as wf:
            print(json.dumps({"progress":"Розбивання аудіо на фрагменти.."}), flush=True)
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
                    segment_start = segment['start'] + (i * 10)
                    segment_end = segment['end'] + (i * 10)
                    transcription.append({
                        "start": segment_start,
                        "end": segment_end,
                        "time": f"{format_time(segment_start)} - {format_time(segment_end)}",
                        "text": segment['text']
                    })
                
                # Видаляємо тимчасовий файл
                os.remove(temp_file_path)
                
                # Відправляємо прогрес
                print(json.dumps({"progress":f"Обробка частини {i + 1}/{num_chunks}"}), flush=True)
                # progress = (i + 1) / num_chunks * 100
                # print(json.dumps({"progress": progress}), flush=True)
        
        print(json.dumps({"progress": 100}), flush=True)  # Відправляємо 100% прогрес після завершення
        print(json.dumps({"status": "Статус обробки: завершено"}), flush=True)
        os.remove(wav_path)  # Видаляємо тимчасовий WAV файл
        return transcription
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Помилка: Не передано всі необхідні параметри."}, ensure_ascii=False))
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    model_name = sys.argv[2]
    language = sys.argv[3]
    device = sys.argv[4]
    
    try:
        transcription = transcribe_audio(audio_file_path, model_name, language, device)
        print(json.dumps(transcription))  # Виводимо результат для Electron
    except Exception as e:
        print(json.dumps({"error": f"Помилка: {e}"}, ensure_ascii=False))
        sys.exit(1)