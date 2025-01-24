import whisper
import os
import torch




# Завантаження моделі Whisper
model = whisper.load_model("medium")  # Можна обрати 'tiny', 'base', 'small', 'medium', 'large'

def transcribe_audio(file_path, language="uk"):
    """
    Функція для транскрибування аудіо за допомогою Whisper.
    :param file_path: шлях до аудіофайлу
    :param language: мова розпізнавання (за замовчуванням українська)
    :return: текстовий результат транскрибування
    """
    try:
        # Виконання транскрипції
        result = model.transcribe(file_path, language=language,fp16=False)
        return result["text"]
    except Exception as e:
        return f"Помилка: {e}"


# Тестовий виклик
if __name__ == "__main__":
    # Вкажіть шлях до вашого аудіофайлу
    file_path = os.path.abspath("test.wav")  # Змінити на ваш аудіофайл
    print(file_path)
    print("Транскрибую аудіо...")
    transcript = transcribe_audio(file_path)
    print("Результат транскрибування:")
    print(transcript)
