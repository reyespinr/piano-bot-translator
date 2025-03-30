import whisper
import requests


async def transcribe(audio_file_path):
    # Debugging statement
    print(f"Starting transcription for {audio_file_path}...")

    # Load Whisper model with GPU support
    model = whisper.load_model("base", device="cuda")  # Use "cuda" for GPU
    # Enable fp16 for faster processing
    result = model.transcribe(audio_file_path, fp16=True)
    text = result["text"]

    print(f"Transcription complete. Text: {text}")  # Debugging statement
    return text


async def translate(text):
    print(f"Starting translation for text: {text}")  # Debugging statement

    # Translate text from any language to English using DeepL API
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": "5ac935ea-9ed2-40a7-bd4d-8153c941c79f:fx",
            "text": text,
            "target_lang": "EN"
        },
        timeout=10
    )
    translation = response.json()["translations"][0]["text"]

    # Debugging statement
    print(f"Translation complete. Result: {translation}")
    return translation
