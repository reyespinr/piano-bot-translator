import whisper
import requests


# Preload the Whisper model globally
print("Loading Whisper model...")
MODEL = whisper.load_model(
    "large-v3-turbo", device="cuda")  # Use "cuda" for GPU
# MODEL = whisper.load_model(
#     "base", device="cuda")  # Use "cuda" for GPU
print("Whisper model loaded successfully!")


async def transcribe(audio_file_path):
    # Use the preloaded model for transcription
    result = MODEL.transcribe(audio_file_path, fp16=True)
    text = result["text"]
    return text


async def translate(text):
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
    return translation
