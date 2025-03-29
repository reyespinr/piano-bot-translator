import asyncio
import pyaudio
import wave
import whisper
import requests
import logging
import numpy as np
import os
from discord.sinks import WaveSink


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
    """Continuously capture audio, transcribe it after detecting a pause, and translate it."""
    print("Starting real-time transcription with pause detection and translation...")  # Debugging statement

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
                    print("Detected pause, processing transcription and translation...")

                    # Combine the entire buffer into a single audio chunk for transcription
                    full_audio_chunk = b''.join(buffer)
                    buffer = []  # Clear the buffer after processing

                    # Skip processing if the buffer is empty
                    if not full_audio_chunk:
                        print(
                            "Buffer is empty, skipping transcription and translation.")
                        continue

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
                    transcribed_text = result["text"].strip()

                    # Skip translation if the transcription is empty
                    if not transcribed_text:
                        print("Transcription is empty, skipping translation.")
                        continue

                    # Translate the transcribed text
                    translated_text = await translate(transcribed_text)

                    # Update the GUI with the transcribed and translated text
                    gui_instance.update_text_display(
                        transcribed_text, translated_text)
                    # Debugging statement
                    print(f"Transcribed: {transcribed_text}")
                    # Debugging statement
                    print(f"Translated: {translated_text}")

    except Exception as e:
        logging.exception(f"Error during listen_and_transcribe: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Debugging statement
        print("Stopped real-time transcription and translation.")


def is_silent(audio_chunk, threshold=200):
    """Check if the audio chunk is silent based on an energy threshold."""
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    mean_energy = np.abs(audio_data).mean()
    return mean_energy < threshold


async def user_listener(vc, user_id, gui_instance):
    """Capture and process audio for a specific user."""
    print(f"Listening for user {user_id}...")  # Debugging statement

    sink = WaveSink()  # Create a separate WaveSink for this user
    temp_audio_file = f"{user_id}_audio.wav"
    silence_threshold = 200  # Energy threshold for silence detection
    # Event to signal when processing is complete
    processing_complete = asyncio.Event()

    # Create a wave file for the user
    with wave.open(temp_audio_file, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(96000)  # 96 kHz sample rate

        async def process_audio_callback(sink, channel):
            """Process audio from the sink."""
            try:
                for audio_user_id, audio in sink.audio_data.items():
                    if audio_user_id != user_id:
                        continue  # Skip audio data for other users

                    # Debugging statement
                    print(f"Received audio data for user {audio_user_id}.")

                    # Check if audio.file is valid
                    if audio.file is None:
                        print(
                            f"Audio file for user {audio_user_id} is None. Skipping...")
                        continue

                    audio_data = audio.file.read()
                    if not audio_data:
                        print(
                            f"Audio data for user {audio_user_id} is empty. Skipping...")
                        continue

                    if not is_silent(audio_data, threshold=silence_threshold):
                        wf.writeframes(audio_data)
                        print(
                            f"Appended audio for user {audio_user_id} to {temp_audio_file}")
            finally:
                # Signal that processing is complete
                processing_complete.set()

        vc.start_recording(
            sink,
            process_audio_callback,  # Pass the coroutine directly
            None,
        )

        # Keep the listener running until it is canceled
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            vc.stop_recording()
            print(f"Stopped listening for user {user_id}.")
            # Wait for the callback to finish processing
            await processing_complete.wait()
            print(f"Audio file {temp_audio_file} closed.")


async def start_listening(vc, gui_instance):
    """Start recording and processing audio from the voice channel."""
    print("Starting to listen...")  # Debugging statement

    sink = WaveSink()

    async def process_audio_callback(sink, channel):
        """Process audio from the sink."""
        for user_id, audio in sink.audio_data.items():
            # Debugging statement
            print(f"Processing audio for user {user_id}...")

            # Save the audio to a temporary file
            temp_audio_file = f"{user_id}_audio.wav"
            with open(temp_audio_file, "wb") as f:
                f.write(audio.file.read())

            # Debugging statement
            print(f"Saved audio for user {user_id} to {temp_audio_file}")

            # Transcribe and translate the audio
            transcribed_text = await transcribe(temp_audio_file)
            translated_text = await translate(transcribed_text)

            # Update the GUI with the results
            gui_instance.update_text_display(
                f"{user_id}: {transcribed_text}",
                f"{user_id}: {translated_text}"
            )

            # Clean up the temporary file
            os.remove(temp_audio_file)

        print("Finished processing audio.")  # Debugging statement

    vc.start_recording(
        sink,
        process_audio_callback,  # Pass the coroutine directly
        None,
    )
    print("Started listening to the voice channel.")
