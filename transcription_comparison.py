import time
import asyncio
import whisper
from transformers import pipeline

# List of audio files to process (replace with your actual file paths)
audio_files = ["test_audio.wav", "test_audio.wav", "test_audio.wav"]

# Case 1: Base model with loading the model every time


async def base_model_load_every_time():
    print("\n--- Case 1: Base Model (Load Every Time) ---")
    total_setup_time = 0
    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure setup time
        setup_start_time = time.time()
        model = whisper.load_model("base", device="cuda")
        setup_elapsed_time = time.time() - setup_start_time

        # Measure transcription time
        transcription_start_time = time.time()
        result = model.transcribe(audio_file, fp16=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_setup_time += setup_elapsed_time
        total_transcription_time += transcription_elapsed_time

        print(f"File: {audio_file}\n  Setup Time: {setup_elapsed_time:.2f}s\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Total Setup Time: {total_setup_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {total_setup_time + total_transcription_time:.2f}s")


# Case 2: Base model with setup done only once
async def base_model_preloaded():
    print("\n--- Case 2: Base Model (Preloaded) ---")
    # Measure setup time
    setup_start_time = time.time()
    model = whisper.load_model("base", device="cuda")
    setup_elapsed_time = time.time() - setup_start_time

    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure transcription time
        transcription_start_time = time.time()
        result = model.transcribe(audio_file, fp16=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_transcription_time += transcription_elapsed_time

        print(
            f"File: {audio_file}\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Setup Time: {setup_elapsed_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {setup_elapsed_time + total_transcription_time:.2f}s")


# Case 3: Large model with loading the model every time
async def large_model_load_every_time():
    print("\n--- Case 3: Large Model (Load Every Time) ---")
    total_setup_time = 0
    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure setup time
        setup_start_time = time.time()
        model = whisper.load_model("large-v3-turbo", device="cuda")
        setup_elapsed_time = time.time() - setup_start_time

        # Measure transcription time
        transcription_start_time = time.time()
        result = model.transcribe(audio_file, fp16=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_setup_time += setup_elapsed_time
        total_transcription_time += transcription_elapsed_time

        print(f"File: {audio_file}\n  Setup Time: {setup_elapsed_time:.2f}s\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Total Setup Time: {total_setup_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {total_setup_time + total_transcription_time:.2f}s")


# Case 4: Large model with setup done only once
async def large_model_preloaded():
    print("\n--- Case 4: Large Model (Preloaded) ---")
    # Measure setup time
    setup_start_time = time.time()
    model = whisper.load_model("large-v3-turbo", device="cuda")
    setup_elapsed_time = time.time() - setup_start_time

    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure transcription time
        transcription_start_time = time.time()
        result = model.transcribe(audio_file, fp16=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_transcription_time += transcription_elapsed_time

        print(
            f"File: {audio_file}\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Setup Time: {setup_elapsed_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {setup_elapsed_time + total_transcription_time:.2f}s")


# Case 5: Pipeline with setup done every time
def pipeline_load_every_time():
    print("\n--- Case 5: Pipeline (Load Every Time) ---")
    total_setup_time = 0
    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure setup time
        setup_start_time = time.time()
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            torch_dtype="auto",
            device="cuda:0"
        )
        setup_elapsed_time = time.time() - setup_start_time

        # Measure transcription time
        transcription_start_time = time.time()
        result = pipe(audio_file, return_timestamps=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_setup_time += setup_elapsed_time
        total_transcription_time += transcription_elapsed_time

        print(f"File: {audio_file}\n  Setup Time: {setup_elapsed_time:.2f}s\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Total Setup Time: {total_setup_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {total_setup_time + total_transcription_time:.2f}s")


# Case 6: Pipeline with setup done only once
def pipeline_preloaded():
    print("\n--- Case 6: Pipeline (Preloaded) ---")
    # Measure setup time
    setup_start_time = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype="auto",
        device="cuda:0"
    )
    setup_elapsed_time = time.time() - setup_start_time

    total_transcription_time = 0

    for audio_file in audio_files:
        # Measure transcription time
        transcription_start_time = time.time()
        result = pipe(audio_file, return_timestamps=True)
        transcription_elapsed_time = time.time() - transcription_start_time

        total_transcription_time += transcription_elapsed_time

        print(
            f"File: {audio_file}\n  Transcription Time: {transcription_elapsed_time:.2f}s")

    print(f"Setup Time: {setup_elapsed_time:.2f}s\nTotal Transcription Time: {total_transcription_time:.2f}s\nTotal Time: {setup_elapsed_time + total_transcription_time:.2f}s")


# Main function to run all cases
async def main():
    await base_model_load_every_time()
    await base_model_preloaded()
    await large_model_load_every_time()
    await large_model_preloaded()
    pipeline_load_every_time()
    pipeline_preloaded()


if __name__ == "__main__":
    asyncio.run(main())
