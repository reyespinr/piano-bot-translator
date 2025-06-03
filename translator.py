"""
Voice translation module for Discord bot.

This module provides real-time voice transcription and translation capabilities
for Discord voice channels. It handles audio processing, language detection,
transcription, and translation using machine learning models.
"""
import asyncio
import os
import gc
import sys
import importlib
import subprocess
from typing import Callable
from custom_sink import RealTimeWaveSink
import utils
from logging_config import get_logger

logger = get_logger(__name__)


class VoiceTranslator:
    """
    Handles real-time voice transcription and translation for Discord.

    This class manages audio processing, language detection, transcription,
    and translation for Discord voice channels. It provides methods to start
    and stop listening, process audio callbacks, and manage voice client
    connections.
    """

    def __init__(self, translation_callback: Callable[[str, str], None]):
        """
        Initialize the voice translator

        Args:
            translation_callback: Function to call when a translation is ready.
                                    Arguments: (user_name, translated_text)
        """
        self.translation_callback = translation_callback
        self.active_voices = {}
        self.sink = None
        self.is_listening = False
        self.model_loaded = False

    async def load_models(self):
        """Verify model loading capabilities without loading the model"""
        try:
            logger.info("Loading translation models...")
            # We don't actually load the model here - it will be loaded
            # on-demand when transcribe is first called
            if hasattr(utils, "_load_model_if_needed"):
                self.model_loaded = True
                return True
            logger.warning("Model loading mechanism not found")
            return False
        except (AttributeError, ImportError) as e:
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
            # Create a simple object to hold the user_processing_enabled dict
            if hasattr(self, 'user_processing_enabled'):
                # CRITICAL FIX: Create fresh copy to prevent reference issues
                self.sink.parent = type('obj', (object,), {})
                processing_settings = {}
                for k, v in self.user_processing_enabled.items():
                    processing_settings[str(k)] = bool(v)

                self.sink.parent.user_processing_enabled = processing_settings
                settings_items = [f"{k}={v}" for k,
                                  v in processing_settings.items()]
                settings_str = ', '.join(settings_items)
                logger.debug("Active user settings: %s", settings_str)
            elif hasattr(voice_client, 'guild') and voice_client.guild:
                # Try to import from server module as a fallback
                try:
                    if 'server' in sys.modules:
                        server_module = sys.modules['server']
                    else:
                        server_module = importlib.import_module('server')

                    if hasattr(server_module, 'user_processing_enabled'):
                        # Create a simple object with user_processing_enabled
                        self.sink.parent = type('obj', (object,), {})
                        self.sink.parent.user_processing_enabled = (
                            server_module.user_processing_enabled)
                        logger.debug(
                            "Set user_processing_enabled from server module "
                            "with %d entries",
                            len(server_module.user_processing_enabled))
                except ImportError:
                    logger.debug("Could not import server module")

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
                            logger.debug(
                                "Audio received from user %s, length: %d",
                                user_id, audio_length)
                        except (IndexError, TypeError, AttributeError):
                            # Just silently handle any exceptions in the logging
                            pass
                    return
                # Return the coroutine object
                return dummy_process()

            # Start recording with our bullet-proof callback
            voice_client.start_recording(self.sink, audio_callback)
            logger.debug("Recording started successfully")

            self.is_listening = True
            logger.debug("Started listening in channel: %s",
                         voice_client.channel.name)
            return True, "Started listening"
        except (ConnectionError, AttributeError, ValueError) as e:
            logger.error("Error in start_listening: %s", str(e))
            return False, f"Error: {str(e)}"

    async def stop_listening(self, voice_client):
        """Stop listening and processing audio"""
        logger.debug(
            "stop_listening called. Voice client valid: %s", voice_client is not None)
        if not voice_client:
            logger.debug("Voice client is None, returning")
            return False, "Not connected to a voice channel"

        try:
            # Stop recording
            logger.debug("Stopping recording")
            voice_client.stop_recording()
            logger.debug("Recording stopped successfully")

            # Add a small delay to ensure workers can complete
            logger.debug("Waiting for workers to finish...")
            await asyncio.sleep(0.5)

            # Clean up sink resources
            if self.sink:
                # Add extra safeguards before cleanup
                try:
                    logger.debug("Cleaning up sink resources")
                    self.sink.cleanup()
                    logger.debug("Sink cleanup completed")
                    # Wait for workers to finish
                    await asyncio.sleep(0.5)
                except (AttributeError, RuntimeError) as e:
                    logger.warning("Error during sink cleanup: %s", str(e))
                self.sink = None
                logger.debug("Sink reference cleared")

            # Make sure the listening state is updated
            self.is_listening = False
            logger.debug("Stopped listening")

            # Return success
            return True, "Stopped listening"
        except (ConnectionError, AttributeError) as e:
            logger.error("Error in stop_listening: %s", str(e))
            return False, f"Error: {str(e)}"

    async def toggle_listening(self, voice_client):
        """Toggle listening on/off"""
        if self.is_listening:
            return await self.stop_listening(voice_client)
        return await self.start_listening(voice_client)

    async def process_audio_callback(self, user_id, audio_file, message_type=None):
        """Process audio data and generate translations"""
        try:
            # If we're called with a message_type, it's a direct text message,
            # not an audio file
            if message_type is not None:
                # Here audio_file is actually the text content when
                # message_type is provided
                text_content = audio_file

                # Forward to translation callback
                if self.translation_callback:
                    try:
                        await self.translation_callback(
                            user_id, text_content, message_type=message_type)
                    except (TypeError, AttributeError, ValueError) as cb_error:
                        logger.error(
                            "Error in translation callback: %s", str(cb_error))
                return

            try:
                # Get transcription and detected language
                transcribed_text, detected_language = await utils.transcribe(audio_file)

                # Skip processing if transcription was empty
                if not transcribed_text:
                    logger.debug(
                        "Empty transcription for user %s, skipping", user_id)
                    await self._force_delete_file(audio_file)
                    return

                # Create transcription message
                logger.info("Sending transcription for user %s: %s",
                            user_id, transcribed_text)
                await self.translation_callback(user_id,
                                                transcribed_text,
                                                message_type="transcription")

                # Determine if translation is needed
                needs_translation = await utils.should_translate(transcribed_text,
                                                                 detected_language)

                if needs_translation:
                    translated_text = await utils.translate(transcribed_text)
                    logger.info("Translated from %s to English",
                                detected_language)
                    # Send the translation
                    logger.info("Sending translation for user %s: %s",
                                user_id, translated_text)
                    await self.translation_callback(user_id,
                                                    translated_text,
                                                    message_type="translation")
                else:
                    # Skip translation for English
                    logger.debug(
                        "Skipped translation - detected language: %s", detected_language)
                    # For English, use the same text for translation display
                    logger.info("Sending direct text for user %s: %s",
                                user_id, transcribed_text)
                    await self.translation_callback(user_id,
                                                    transcribed_text,
                                                    message_type="translation")
            finally:
                # CRITICAL: Always delete the file when done with it
                await self._force_delete_file(audio_file)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error in process_audio_callback: %s", str(e))
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
                gc.collect()

                # Delete and verify
                os.remove(file_path)
                logger.debug("Deleted file: %s", os.path.basename(file_path))

                if not os.path.exists(file_path):
                    return True

                logger.warning(
                    "File still exists after deletion attempt: %s", file_path)
            except (PermissionError, OSError) as e:
                logger.warning("Deletion attempt %d failed: %s",
                               attempt+1, str(e))
                await asyncio.sleep(0.5)  # Wait before retry

        # Last resort: try with Windows-specific commands
        if os.name == 'nt':
            try:
                subprocess.run(f'del /F "{file_path}"',
                               shell=True, check=False)
                logger.debug(
                    "Attempted deletion with Windows command: %s", file_path)
                return not os.path.exists(file_path)
            except (OSError, subprocess.SubprocessError) as e:
                logger.error("Windows command deletion failed: %s", str(e))

        logger.warning(
            "Failed to delete file after multiple attempts: %s", file_path)
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
            except (AttributeError, ConnectionError) as e:
                logger.error(
                    "Error during cleanup for guild %s: %s", guild_id, str(e))

        # Clean up sink
        if self.sink:
            self.sink.cleanup()
            self.sink = None

        self.active_voices = {}
        self.is_listening = False
