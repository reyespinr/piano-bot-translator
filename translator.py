import asyncio
import os
from typing import Callable
from custom_sink import RealTimeWaveSink
import utils
from logging_config import get_logger

logger = get_logger(__name__)


class VoiceTranslator:
    def __init__(self, translation_callback: Callable[[str, str], None]):
        """
        Initialize the voice translator        Args:
            translation_callback: Function to call when a translation is ready
                                 Arguments: (user_name, translated_text)
        """
        self.translation_callback = translation_callback
        self.active_voices = {}
        self.sink = None
        self.is_listening = False        # No need to preload the model at initialization
        self.model_loaded = False

    async def load_models(self):
        """Verify model loading capabilities without actually loading the model"""
        try:
            logger.info("Loading translation models...")
            # We don't actually load the model here - it will be loaded
            # on-demand when transcribe is first called
            if hasattr(utils, "_load_model_if_needed"):
                self.model_loaded = True
                return True
            logger.warning("Model loading mechanism not found")
            return False
        except Exception as e:
            logger.error(
                "Error verifying model loading capability: %s", str(e))
            return False

    def setup_voice_receiver(self, voice_client):
        """Set up the voice receiver for a Discord voice client"""
        if not self.model_loaded:
            logger.warning("Models not loaded yet!")

        # Store the voice client for later use
        self.active_voices[voice_client.guild.id] = voice_client

    async def start_listening(self, voice_client):
        """Start listening and processing audio"""
        if not voice_client or not voice_client.is_connected():
            return False, "Not connected to a voice channel"

        try:
            # Create sink for real-time audio processing
            logger.info("Creating voice processing sink...")
            self.sink = RealTimeWaveSink(
                pause_threshold=1.0,
                event_loop=asyncio.get_event_loop()
            )

            # Set the translation callback on the sink
            self.sink.translation_callback = self.process_audio_callback

            # Set parent reference to access user_processing_enabled dictionary
            # Create a simple object to hold the user_processing_enabled dictionary
            if hasattr(self, 'user_processing_enabled'):
                # CRITICAL FIX: Create a fresh copy of the dictionary to prevent reference issues
                self.sink.parent = type('obj', (object,), {})
                processing_settings = {}
                for k, v in self.user_processing_enabled.items():
                    processing_settings[str(k)] = bool(v)

                self.sink.parent.user_processing_enabled = processing_settings
                settings_str = ', '.join(
                    [f'{k}={v}' for k, v in processing_settings.items()])
                logger.debug("Active user settings: %s", settings_str)
            elif hasattr(voice_client, 'guild') and voice_client.guild:
                # Try to import from server module as a fallback
                try:
                    # Import here to avoid circular imports
                    import sys
                    import importlib
                    if 'server' in sys.modules:
                        server_module = sys.modules['server']
                    else:
                        server_module = importlib.import_module('server')

                    if hasattr(server_module, 'user_processing_enabled'):
                        # Create a simple object with user_processing_enabled
                        self.sink.parent = type('obj', (object,), {})
                        self.sink.parent.user_processing_enabled = server_module.user_processing_enabled
                        print(
                            f"DEBUG: Set user_processing_enabled from server module with {len(server_module.user_processing_enabled)} entries")
                except ImportError:
                    # Define the audio callback - the KEY fix is here
                    print("DEBUG: Could not import server module")
            # The audio callback must be a normal function that returns a coroutine
            # And it needs to be resistant to any argument patterns

            def audio_callback(_sink, *args):
                # Create a dummy coroutine that does nothing
                async def dummy_process():
                    # Only log if we have proper arguments
                    if len(args) > 0:
                        try:
                            user_id = args[0]
                            audio_data = args[1] if len(args) > 1 else None
                            audio_length = len(audio_data) if audio_data else 0
                            print(
                                f"Audio received from user {user_id}, length: {audio_length}")
                        except Exception:
                            # Just silently handle any exceptions in the logging
                            pass
                    return
                # Return the coroutine object
                return dummy_process()

            # Start recording with our bullet-proof callback
            voice_client.start_recording(self.sink, audio_callback)
            print("DEBUG: Recording started successfully")

            self.is_listening = True
            print(
                f"DEBUG: Started listening in channel: {voice_client.channel.name}")
            return True, "Started listening"
        except Exception as e:
            import traceback
            print(f"DEBUG: Error in start_listening: {str(e)}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return False, f"Error: {str(e)}"

    async def stop_listening(self, voice_client):
        """Stop listening and processing audio"""
        print(
            f"DEBUG: stop_listening called. Voice client valid: {voice_client is not None}")
        if not voice_client:
            print("DEBUG: Voice client is None, returning")
            return False, "Not connected to a voice channel"

        try:
            # Stop recording
            print("DEBUG: Stopping recording")
            voice_client.stop_recording()
            print("DEBUG: Recording stopped successfully")

            # Add a small delay to ensure workers can complete
            print("DEBUG: Waiting for workers to finish...")
            await asyncio.sleep(0.5)

            # Clean up sink resources
            if self.sink:
                # Add extra safeguards before cleanup
                try:
                    print("DEBUG: Cleaning up sink resources")
                    self.sink.cleanup()
                    print("DEBUG: Sink cleanup completed")
                    # Wait for workers to finish
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"DEBUG: Error during sink cleanup: {e}")
                    import traceback
                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
                self.sink = None
                print("DEBUG: Sink reference cleared")

            # Make sure the listening state is updated
            self.is_listening = False
            print("DEBUG: Stopped listening")

            # Return success
            return True, "Stopped listening"
        except Exception as e:
            import traceback
            print(f"DEBUG: Error in stop_listening: {str(e)}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return False, f"Error: {str(e)}"

    async def toggle_listening(self, voice_client):
        """Toggle listening on/off"""
        if self.is_listening:
            return await self.stop_listening(voice_client)
        return await self.start_listening(voice_client)

    async def process_audio_callback(self, user_id, audio_file, message_type=None):
        """Process audio data and generate translations"""
        try:
            # If we're called with a message_type, it's a direct text message, not an audio file
            if message_type is not None:
                # Here audio_file is actually the text content when message_type is provided
                text_content = audio_file

                # Forward to translation callback
                if self.translation_callback:
                    try:
                        await self.translation_callback(user_id, text_content, message_type=message_type)
                    except Exception as cb_error:
                        print(f"Error in translation callback: {cb_error}")
                return

            try:
                # Get transcription and detected language
                transcribed_text, detected_language = await utils.transcribe(audio_file)

                # Skip processing if transcription was empty
                if not transcribed_text:
                    print(f"Empty transcription for user {user_id}, skipping")
                    await self._force_delete_file(audio_file)
                    return

                # Create transcription message
                print(
                    f"Sending transcription for user {user_id}: {transcribed_text}")
                await self.translation_callback(user_id, transcribed_text, message_type="transcription")

                # Determine if translation is needed
                needs_translation = await utils.should_translate(transcribed_text, detected_language)

                if needs_translation:
                    translated_text = await utils.translate(transcribed_text)
                    print(f"Translated from {detected_language} to English")
                    # Send the translation
                    print(
                        f"Sending translation for user {user_id}: {translated_text}")
                    await self.translation_callback(user_id, translated_text, message_type="translation")
                else:
                    # Skip translation for English
                    print(
                        f"Skipped translation - detected language: {detected_language}")
                    # For English, use the same text for translation display
                    print(
                        f"Sending direct text for user {user_id}: {transcribed_text}")
                    await self.translation_callback(user_id, transcribed_text, message_type="translation")
            finally:
                # CRITICAL: Always delete the file when done with it
                await self._force_delete_file(audio_file)

        except Exception as e:
            print(f"Error in process_audio_callback: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Make sure to clean up even if exception occurs
            await self._force_delete_file(audio_file)

    async def _force_delete_file(self, file_path):
        """Forcefully delete a file with multiple retries."""
        if not file_path or not os.path.exists(file_path):
            return False

        # Try to delete the file with multiple retries
        for attempt in range(3):
            try:
                # Force garbage collection to release file handles
                import gc
                gc.collect()

                # Delete and verify
                os.remove(file_path)
                print(f"🗑️ Deleted file: {os.path.basename(file_path)}")

                if not os.path.exists(file_path):
                    return True

                print(
                    f"⚠️ File still exists after deletion attempt: {file_path}")
            except (PermissionError, OSError) as e:
                print(f"🔄 Deletion attempt {attempt+1} failed: {e}")
                await asyncio.sleep(0.5)  # Wait before retry

        # Last resort: try with Windows-specific commands
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.run(f'del /F "{file_path}"',
                               shell=True, check=False)
                print(
                    f"🗑️ Attempted deletion with Windows command: {file_path}")
                return not os.path.exists(file_path)
            except Exception as e:
                print(f"❌ Windows command deletion failed: {e}")

        print(f"⚠️ Failed to delete file after multiple attempts: {file_path}")
        return False

    def _cleanup_audio_file(self, audio_file):
        """Legacy method for compatibility - use _force_delete_file instead."""
        # Run the async delete in a non-blocking way
        if audio_file and os.path.exists(audio_file):
            asyncio.create_task(self._force_delete_file(audio_file))

    def cleanup(self):
        """Clean up resources"""
        # Stop all recordings
        for guild_id, voice_client in list(self.active_voices.items()):
            try:
                if voice_client.is_connected():
                    if self.is_listening:
                        voice_client.stop_recording()
                    asyncio.create_task(voice_client.disconnect())
            except Exception as e:
                print(f"Error during cleanup for guild {guild_id}: {str(e)}")

        # Clean up sink
        if self.sink:
            self.sink.cleanup()
            self.sink = None

        self.active_voices = {}
        self.is_listening = False
