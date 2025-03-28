import asyncio
import pyaudio
import wave
import whisper
import requests
import logging
import numpy as np


class AudioRecorder:
    def __init__(self):
        self.frames = []

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_frames(self):
        return b''.join(self.frames)


async def listen(vc, gui_instance):
    """Capture audio for processing."""
    print("Starting to listen...")  # Debugging statement

    p = pyaudio.PyAudio()
    recorder = AudioRecorder()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True,
                    frames_per_buffer=1024, stream_callback=recorder.callback)
    stream.start_stream()

    # Wait until listening is stopped (controlled by GUI button)
    while gui_instance.is_listening:
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

    # Load Whisper model with GPU support
    model = whisper.load_model("large-v2", device="cuda")  # Use "cuda" for GPU
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


async def process_audio(vc, gui_instance):
    try:
        # Step 1: Listen
        audio_file = await listen(vc, gui_instance)

        # Step 2: Transcribe
        transcribed_text = await transcribe(audio_file)

        # Step 3: Translate
        translated_text = await translate(transcribed_text)

        print(f"Final Translation: {translated_text}")  # Debugging statement
        return transcribed_text, translated_text  # Return both texts
    except Exception as e:
        logging.exception(f"Error during process_audio: {e}")
        return None, None


async def listen_and_transcribe(vc, gui_instance):
    """Continuously capture audio and transcribe it after detecting a pause."""
    print("Starting real-time transcription with pause detection...")  # Debugging statement

    p = pyaudio.PyAudio()
    recorder = AudioRecorder()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True,
                    frames_per_buffer=1024, stream_callback=recorder.callback)
    stream.start_stream()

    # Load the Whisper base model
    model = whisper.load_model("base", device="cuda")

    # Rolling buffer for audio chunks
    buffer = []
    silence_threshold = 100  # Energy threshold for silence detection
    # Minimum duration of silence (in seconds) to trigger transcription
    silence_duration = 1.0
    # Number of samples for the silence duration
    silence_samples = int(48000 * silence_duration)

    try:
        while gui_instance.is_listening:
            await asyncio.sleep(0.1)

            # Append new audio frames to the buffer
            buffer.extend(recorder.frames)
            recorder.frames = []  # Clear the recorder's frames

            # Check if the buffer contains enough data for silence detection
            if len(buffer) * 1024 >= silence_samples:
                # Combine frames into a single chunk for silence detection
                audio_chunk = b''.join(buffer[-silence_samples // 1024:])

                # Check for silence
                if is_silent(audio_chunk, threshold=silence_threshold):
                    print("Detected pause, processing transcription...")

                    # Combine the entire buffer into a single audio chunk for transcription
                    full_audio_chunk = b''.join(buffer)
                    buffer = []  # Clear the buffer after processing

                    # Save the chunk to a temporary file
                    temp_file = "temp_chunk.wav"
                    wf = wave.open(temp_file, "wb")
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(48000)
                    wf.writeframes(full_audio_chunk)
                    wf.close()

                    # Transcribe the chunk
                    result = model.transcribe(temp_file, fp16=True)
                    transcribed_text = result["text"]

                    # Update the GUI with the transcribed text
                    gui_instance.update_text_display(transcribed_text, "")
                    # Debugging statement
                    print(f"Transcribed: {transcribed_text}")

    except Exception as e:
        logging.exception(f"Error during listen_and_transcribe: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stopped real-time transcription.")  # Debugging statement


def is_silent(audio_chunk, threshold=200):
    """Check if the audio chunk is silent based on an energy threshold."""
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    mean_energy = np.abs(audio_data).mean()
    # Debugging statement
    print(f"Mean energy: {mean_energy}, Threshold: {threshold}")
    return mean_energy < threshold
