import asyncio
import os
import time
import numpy as np
from discord.sinks import WaveSink
import io
import wave
import utils


class RealTimeWaveSink(WaveSink):
    def __init__(self, pause_threshold=1.0, event_loop=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_threshold = pause_threshold  # Time in seconds to detect a pause
        self.user_last_packet_time = {}  # Track the last packet time for each user
        self.event_loop = event_loop  # Store the main thread's event loop
        self.is_speaking = {}  # Track if user is currently speaking
        self.silence_frames = {}  # Count consecutive silence frames
        self.speech_detected = {}  # Track if speech was detected in the current session
        self.speech_buffers = {}  # Separate buffer to store only speech content
        self.pre_speech_buffers = {}  # Buffer to store audio before speech is detected
        self.energy_history = {}  # Track recent energy levels for better detection

    def is_audio_active(self, audio_data, user):
        """Check if audio data contains active speech with improved detection."""
        # Convert bytes to numpy array (assuming PCM signed 16-bit little-endian)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Calculate energy of the current frame
        energy = np.mean(np.abs(audio_array))

        # Initialize energy history if needed
        if user not in self.energy_history:
            self.energy_history[user] = [energy] * \
                5  # Start with current energy

        # Update energy history
        self.energy_history[user].append(energy)
        # Keep last 5 frames
        self.energy_history[user] = self.energy_history[user][-5:]

        # Calculate average energy over recent frames
        avg_energy = np.mean(self.energy_history[user])

        # Lower threshold for considering audio as speech
        return avg_energy > 300  # Lower threshold to capture more of the speech

    def write(self, data, user):
        try:
            current_time = time.time()

            # Initialize user-specific data structures if needed
            if user not in self.user_last_packet_time:
                self.user_last_packet_time[user] = current_time
                self.is_speaking[user] = False
                self.silence_frames[user] = 0
                self.speech_detected[user] = False
                self.speech_buffers[user] = io.BytesIO()
                self.pre_speech_buffers[user] = []

            # Calculate time difference since the last packet
            time_diff = current_time - self.user_last_packet_time[user]

            # If pause is very long, process any speech and reset everything
            if time_diff > self.pause_threshold:
                print(
                    f"Long pause detected for user {user}. Processing any speech and resetting.")

                # Process any accumulated speech before resetting
                if self.is_speaking[user] and self.speech_detected[user]:
                    print(
                        f"Processing speech before resetting for user {user}")
                    self.process_speech_buffer(user)

                # Reset state
                self.is_speaking[user] = False
                self.silence_frames[user] = 0
                self.speech_detected[user] = False
                # Reset buffers
                self.speech_buffers[user] = io.BytesIO()
                self.pre_speech_buffers[user] = []

            # Always store recent frames for smoother beginning of speech
            self.pre_speech_buffers[user].append(data)
            if len(self.pre_speech_buffers[user]) > 10:  # Keep ~200ms of audio
                self.pre_speech_buffers[user].pop(0)

            # Check if the current packet contains active speech
            is_active = self.is_audio_active(data, user)

            if is_active:
                # Reset silence counter when speech is detected
                self.silence_frames[user] = 0

                # Mark that speech was detected in this session
                self.speech_detected[user] = True

                # If this is the start of speech, add pre-speech buffer first
                if not self.is_speaking[user]:
                    print(f"Speech started for user {user}")
                    self.is_speaking[user] = True
                    # Add pre-speech frames for smoother beginning
                    for pre_data in self.pre_speech_buffers[user]:
                        self.speech_buffers[user].write(pre_data)

                # Add this audio data to the speech buffer
                self.speech_buffers[user].write(data)

            else:
                # If we're speaking, still add a few frames to the speech buffer
                # for smoother transitions
                if self.is_speaking[user] and self.silence_frames[user] < 5:
                    self.speech_buffers[user].write(data)

                # Increment silence counter
                self.silence_frames[user] += 1

                # Check if silence has persisted long enough to consider speech ended
                if self.is_speaking[user] and self.silence_frames[user] > 15:
                    print(f"Speech ended for user {user}. Processing audio.")
                    self.is_speaking[user] = False

                    # Only process if speech was detected in this session
                    if self.speech_detected[user]:
                        self.process_speech_buffer(user)
                        self.speech_detected[user] = False
                        # Reset speech buffer for next utterance
                        self.speech_buffers[user] = io.BytesIO()

            # Update the last packet time
            self.user_last_packet_time[user] = current_time

            # Write to the main buffer
            super().write(data, user)

        except Exception as e:
            # Don't let exceptions in write crash the audio thread
            print(f"Error in write method for user {user}: {e}")

    def process_speech_buffer(self, user):
        """Process the speech buffer for a user and trigger transcription."""
        if user not in self.speech_buffers:
            print(f"No speech buffer found for user {user}.")
            return

        # Get the speech buffer
        speech_buffer = self.speech_buffers[user]
        speech_buffer.seek(0, os.SEEK_END)
        buffer_size = speech_buffer.tell()

        # Minimum size for meaningful transcription
        if buffer_size < 8000:  # Lowered threshold (~1/6 second)
            print(
                f"Speech buffer for user {user} is too small ({buffer_size} bytes), skipping transcription.")
            return

        # Create a temporary WAV file from the speech buffer with timestamp to avoid conflicts
        timestamp = int(time.time())
        temp_audio_file = f"{user}_{timestamp}_speech.wav"

        # Reset buffer pointer
        speech_buffer.seek(0)

        try:
            # Write the speech data to a WAV file
            with wave.open(temp_audio_file, 'wb') as wf:
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(48000)  # 48kHz (Discord's standard)
                wf.writeframes(speech_buffer.read())

            print(
                f"Processing speech for user {user}, audio file size: {os.path.getsize(temp_audio_file)} bytes")

            # Trigger transcription in the main thread's event loop
            asyncio.run_coroutine_threadsafe(
                self.transcribe_audio(temp_audio_file, user), self.event_loop
            )
        except Exception as e:
            print(f"Error creating WAV file for user {user}: {e}")

    async def transcribe_audio(self, audio_file, user):
        """Transcribe the audio file and handle the result."""
        try:
            # Check if the file exists before transcribing
            if not os.path.exists(audio_file):
                print(
                    f"Audio file {audio_file} does not exist, skipping transcription.")
                return

            # Get transcription
            transcribed_text = await utils.transcribe(audio_file)

            # Get the translation
            translated_text = await utils.translate(transcribed_text) if transcribed_text else ""

            # Debug output
            print(f"Transcription for user {user}: {transcribed_text}")
            print(f"Translation: {translated_text}")

            # Update the GUI
            try:
                # We need to get the Discord user object to display the username
                if hasattr(self, 'parent') and self.parent and hasattr(self.parent, 'vc') and self.parent.vc:
                    # Get the guild and find the member
                    guild = self.parent.vc.guild
                    member = guild.get_member(int(user))
                    user_name = member.display_name if member else f"User {user}"

                    # Format the display text
                    display_transcription = f"{user_name}: {transcribed_text}" if transcribed_text else ""
                    display_translation = f"{user_name}: {translated_text}" if translated_text else ""

                    if display_transcription and display_translation:
                        # Update the GUI display
                        self.parent.update_text_display(
                            display_transcription, display_translation)
                else:
                    print("Could not find parent or voice client to get username")
            except Exception as e:
                print(f"Error updating GUI for user {user}: {e}")

            # Clean up the temporary file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                print(f"Error removing audio file {audio_file}: {e}")

        except Exception as e:
            print(
                f"Error during transcription/translation for user {user}: {e}")
