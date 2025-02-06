import sys
import whisper
import os
import wave
from pydub import AudioSegment

# Завантажуємо модель Whisper
model = whisper.load_model("medium")

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(file_path, language="uk"):
    # Конвертуємо аудіофайл у формат WAV
    wav_path = convert_to_wav(file_path)
    
    print("Статус обробки: початок", flush=True)
    
    # Розділяємо аудіофайл на менші частини
    with wave.open(wav_path, 'rb') as wf:
        total_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        chunk_size = frame_rate * 10  # 10 секундні частини
        num_chunks = total_frames // chunk_size + 1

        transcription = ""
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
            transcription += result['text'] + " "
            
            # Видаляємо тимчасовий файл
            os.remove(temp_file_path)
            
            # Відправляємо прогрес
            progress = (i + 1) / num_chunks * 100
            print(f"Прогрес: {progress:.2f}%", flush=True)
    
    print("Статус обробки: завершено", flush=True)
    os.remove(wav_path)  # Видаляємо тимчасовий WAV файл
    return transcription.strip()

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