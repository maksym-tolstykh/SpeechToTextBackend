import whisper
import os
import moviepy.editor as mp
import srt
from datetime import timedelta
import ffmpeg

# Завантаження моделі Whisper
model = whisper.load_model("medium")

def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    """
    Витягує аудіо з відео.
    :param video_path: шлях до відеофайлу
    :param audio_path: шлях до збереженого аудіофайлу
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, fps=16000)
    return audio_path

def generate_srt_segments(transcription):
    """
    Генерує сегменти субтитрів у форматі SRT.
    :param transcription: результат транскрипції Whisper з таймкодами
    :return: список об'єктів SRT
    """
    segments = transcription["segments"]
    srt_segments = []
    for segment in segments:
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        content = segment["text"]
        srt_segments.append(srt.Subtitle(index=len(srt_segments) + 1, start=start, end=end, content=content))
    return srt_segments

def save_srt_file(srt_segments, output_path="subtitles.srt"):
    """
    Зберігає субтитри у форматі SRT.
    :param srt_segments: список об'єктів SRT
    :param output_path: шлях до файлу субтитрів
    """
    with open(output_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt.compose(srt_segments))

def add_subtitles_to_video(video_path, srt_path, output_path="video_with_subtitles.mp4"):
    """
    Додає субтитри до відео.
    :param video_path: шлях до оригінального відеофайлу
    :param srt_path: шлях до файлу субтитрів
    :param output_path: шлях до вихідного відеофайлу
    """
    ffmpeg.input(video_path).output(output_path, vf=f"subtitles={srt_path}").run()

# Тестовий виклик
if __name__ == "__main__":
    # Шляхи до файлів
    video_file_path = os.path.abspath("test_video.mp4")  # Ваш відеофайл
    audio_file_path = "extracted_audio.wav"
    srt_file_path = "subtitles.srt"
    output_video_path = "video_with_subtitles.mp4"

    # 1. Витягування аудіо
    print("Витягую аудіо з відео...")
    audio_file_path = extract_audio_from_video(video_file_path)

    # 2. Транскрибування аудіо
    print("Транскрибую аудіо...")
    transcription = model.transcribe(audio_file_path, language="uk", fp16=False)

    # 3. Генерація SRT
    print("Генерую субтитри...")
    srt_segments = generate_srt_segments(transcription)
    save_srt_file(srt_segments, srt_file_path)

    # 4. Додавання субтитрів до відео
    print("Додаю субтитри до відео...")
    add_subtitles_to_video(video_file_path, srt_file_path, output_video_path)

    print(f"Результат збережено: {output_video_path}")
