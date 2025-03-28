import asyncio
import pyaudio
import wave
import whisper
import requests
import logging


class AudioRecorder:
    def __init__(self):
        self.frames = []

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_frames(self):
        return b''.join(self.frames)


async def listen(vc, gui_instance):
    print("Starting to listen...")  # Debugging statement

    p = pyaudio.PyAudio()
    recorder = AudioRecorder()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True,
                    frames_per_buffer=1024, stream_callback=recorder.callback)
    stream.start_stream()

    # Wait until listening is stopped (controlled by GUI button)
    while gui_instance.is_listening:  # Use GUI's is_listening variable
        await asyncio.sleep(0.1)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = recorder.get_frames()
    # Debugging statement
    print(f"Captured {len(audio_data)} bytes of audio data")

    # Save audio to file
    output_file = "output.wav"
    wf = wave.open(output_file, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(48000)
    wf.writeframes(audio_data)
    wf.close()

    # Debugging statement
    print(f"Listening stopped. Audio saved to {output_file}")
    return output_file


async def transcribe(audio_file_path):
    # Debugging statement
    print(f"Starting transcription for {audio_file_path}...")

    # Transcribe speech to text using Whisper
    model = whisper.load_model("base", device="cpu")  # Removed fp16 argument
    result = model.transcribe(audio_file_path, fp16=False)
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


async def listen_and_process(vc, gui_instance):
    try:
        # Step 1: Listen
        audio_file = await listen(vc, gui_instance)

        # Step 2: Transcribe
        transcribed_text = await transcribe(audio_file)

        # Step 3: Translate
        translated_text = await translate(transcribed_text)

        print(f"Final Translation: {translated_text}")  # Debugging statement
    except Exception as e:
        logging.exception(f"Error during listen and process: {e}")
