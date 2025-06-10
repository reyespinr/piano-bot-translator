"""
Voice translation module for Discord bot.

This module provides real-time voice transcription and translation capabilities
for Discord voice channels. It handles audio processing, language detection,
transcription, and translation using machine learning models.
"""
import asyncio
import os
import traceback
import struct
import sys
import gc
import subprocess
import translation
from typing import Callable
from custom_sink import RealTimeWaveSink
from logging_config import get_logger

logger = get_logger(__name__)

# Constants for cleanup and deletion
CLEANUP_DELAY = 0.5
MAX_DELETE_RETRIES = 3


class VoiceTranslator:
    """Voice translation manager for Discord bot."""

    def __init__(self, translation_callback: Callable = None):
        self.translation_callback = translation_callback
        self.websocket_handler = None
        self.voice_client = None
        self.sink = None
        self.is_listening = False
        self.user_processing_enabled = {}  # Track which users have processing enabled
        self.active_voices = {}  # Track active voice states
        self.connected_users = {}  # Track connected users
        self.current_channel = None  # CRITICAL FIX: Add missing attribute
        self.model_loaded = False  # CRITICAL FIX: Add missing attribute
        logger.info("✅ VoiceTranslator initialized")

    def set_websocket_handler(self, handler):
        """Set the WebSocket handler for sending user updates."""
        self.websocket_handler = handler

    async def join_voice_channel(self, channel):
        """Join a voice channel and set up audio processing."""
        try:
            logger.info("Joining voice channel: %s", channel.name)

            # Join the voice channel
            self.voice_client = await channel.connect()
            self.current_channel = channel

            # Initialize user processing states for current members
            self._initialize_user_states()

            # Send user list to WebSocket clients
            if self.websocket_handler:
                await self._send_user_updates()

            logger.info(
                "✅ Successfully joined voice channel: %s", channel.name)
            return True

        except Exception as e:
            logger.error("Failed to join voice channel %s: %s",
                         channel.name, str(e))
            return False

    def _initialize_user_states(self):
        """Initialize processing states for users currently in the voice channel."""
        if not self.current_channel:
            return

        # Clear existing states
        self.user_processing_enabled.clear()
        self.connected_users.clear()

        # Set all current members to enabled by default
        for member in self.current_channel.members:
            if not member.bot:  # Exclude bots
                user_id_str = str(member.id)
                self.user_processing_enabled[user_id_str] = True
                self.connected_users[user_id_str] = {
                    'id': user_id_str,
                    'name': member.display_name
                }
                logger.debug("Initialized user %s (%s) as enabled",
                             member.display_name, member.id)

    async def _send_user_updates(self):
        """Send current user list and states to WebSocket clients."""
        if not self.websocket_handler or not self.current_channel:
            logger.warning(
                "No websocket handler or current channel available for user updates")
            return

        users = []
        for member in self.current_channel.members:
            if not member.bot:  # Exclude bots
                user_id_str = str(member.id)
                users.append({
                    'id': user_id_str,
                    'name': member.display_name,
                    'avatar': str(member.avatar.url) if member.avatar else None
                })
                # Ensure user is in processing states
                if user_id_str not in self.user_processing_enabled:
                    self.user_processing_enabled[user_id_str] = True

        # Send users update
        try:
            await self.websocket_handler.broadcast_message({
                'type': 'users_update',
                'users': users,
                'enabled_states': self.user_processing_enabled.copy()
            })
            logger.info(
                "Sent user updates: %d users with processing states", len(users))
        except Exception as e:
            logger.error("Failed to send user updates: %s", str(e))

    async def start_listening(self):
        """Start listening to voice channel audio with real-time processing."""
        if not self.voice_client:
            logger.error("No voice client available")
            return False

        if self.is_listening:
            logger.warning("Already listening")
            return True

        try:
            logger.info("Creating voice processing sink...")

            # Create the sink with proper callback
            self.sink = RealTimeWaveSink(
                pause_threshold=1.0,
                event_loop=asyncio.get_event_loop(),
                num_workers=6
            )

            # Set the parent reference for user toggle access
            self.sink.parent = self

            # Set the translation callback
            self.sink.translation_callback = self.process_audio_callback

            # Apply Discord audio corruption protection
            logger.debug("Applied Discord audio corruption protection")

            # Start recording
            self.voice_client.start_recording(
                self.sink,
                self._handle_audio_finished,
                *[self.voice_client.guild.get_member(user_id)
                  for user_id in [m.id for m.id in self.current_channel.members if not m.bot]]
            )

            logger.debug("Recording started successfully")

            self.is_listening = True
            logger.debug("Started listening in channel: %s",
                         self.current_channel.name)

            # CRITICAL FIX: Send user updates after starting to listen
            await self._send_user_updates()

            return True

        except Exception as e:
            logger.error("Failed to start listening: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return False

    def _handle_audio_finished(self, sink, *args):
        """Handle audio recording finished event."""
        logger.debug("Audio recording finished")

    async def process_audio_callback(self, user_id, text, message_type):
        """Process audio transcription/translation callback."""
        try:
            logger.debug(
                "🔄 process_audio_callback called for user %s with message_type: %s", user_id, message_type)

            if message_type == "transcription":
                logger.info(
                    "📝 Received transcription for user %s: %s", user_id, text)
                # Send to WebSocket clients
                if self.websocket_handler:
                    await self.websocket_handler.broadcast_translation(user_id, text, "transcription")

            elif message_type == "translation":
                logger.info(
                    "🌍 Received translation for user %s: %s", user_id, text)
                # Send translation to WebSocket clients
                if self.websocket_handler:
                    await self.websocket_handler.broadcast_translation(user_id, text, "translation")
                    logger.debug(
                        "✅ Translation message sent to WebSocket clients")
                else:
                    logger.warning(
                        "❌ No websocket_handler available for translation")

        except Exception as e:
            logger.error("Error in process_audio_callback: %s", str(e))

    async def _get_user_display_name(self, user_id):
        """Get user display name from Discord bot - simplified version."""
        try:
            if self.current_channel:
                # Try to find user in current voice channel first
                for member in self.current_channel.members:
                    if str(member.id) == str(user_id):
                        return member.display_name
        except Exception as e:
            logger.debug("Error getting user display name: %s", str(e))

        # Fallback to generic name if we can't resolve it
        return f"User {user_id}"

    async def toggle_user_processing(self, user_id, enabled):
        """Toggle processing for a specific user."""
        try:
            user_id_str = str(user_id)
            self.user_processing_enabled[user_id_str] = enabled

            # Get user info for logging
            if self.current_channel:
                member = self.current_channel.guild.get_member(int(user_id))
                user_name = member.display_name if member else f"User {user_id}"
            else:
                user_name = f"User {user_id}"

            logger.info("User processing %s for %s (%s)",
                        "enabled" if enabled else "disabled", user_name, user_id)

            # Send update to WebSocket clients
            if self.websocket_handler:
                await self.websocket_handler.broadcast_message({
                    'type': 'user_toggle',
                    'user_id': user_id_str,
                    'enabled': enabled
                })

            return True

        except Exception as e:
            logger.error("Error toggling user processing: %s", str(e))
            return False

    async def handle_voice_state_update(self, member, before, after):
        """Handle voice state updates (users joining/leaving)."""
        try:
            if not self.current_channel:
                return

            # Check if the update is for our current channel
            if (before.channel == self.current_channel or
                    after.channel == self.current_channel):

                if member.bot:  # Ignore bots
                    return

                user_id_str = str(member.id)

                # User joined our channel
                if after.channel == self.current_channel and before.channel != self.current_channel:
                    # Enable processing by default for new users
                    self.user_processing_enabled[user_id_str] = True
                    logger.info("👋 User %s joined voice channel",
                                member.display_name)

                    # Send user joined update
                    if self.websocket_handler:
                        user_data = {
                            "id": user_id_str,
                            "name": member.display_name,
                            "avatar": str(member.avatar.url) if member.avatar else None
                        }
                        await self.websocket_handler.broadcast_user_joined(user_data, True)

                # User left our channel
                elif before.channel == self.current_channel and after.channel != self.current_channel:
                    # Remove from processing states
                    self.user_processing_enabled.pop(user_id_str, None)
                    logger.info("👋 User %s left voice channel",
                                member.display_name)

                    # Send user left update
                    if self.websocket_handler:
                        await self.websocket_handler.broadcast_user_left(user_id_str)

        except Exception as e:
            logger.error("Error handling voice state update: %s", str(e))

    async def load_models(self):
        """Verify model loading capabilities and warm up models with timeout protection"""
        try:
            logger.info("Loading translation models...")
            # Import utils module to access transcription functionality
            import utils

            # Check if we have the transcribe function available
            if hasattr(utils, "transcribe"):
                try:
                    # Just load the models without warmup first
                    utils._load_models_if_needed()
                    self.model_loaded = True
                    logger.info("✅ Models loaded successfully")

                    # The server.py handles the warmup, so we don't need to do it here
                    logger.info(
                        "✅ Model loading completed (warmup handled by server)")
                    return True

                except Exception as warmup_error:
                    logger.warning(
                        "Model loading error but continuing: %s", str(warmup_error))
                    return True
            else:
                logger.warning("Model loading mechanism not found")
                return False
        except Exception as e:
            logger.error("Error during model loading: %s", str(e))
            # Check if models were actually loaded despite the error
            if hasattr(utils, '_load_models_if_needed'):
                try:
                    utils._load_models_if_needed()
                    self.model_loaded = True
                    logger.info("✅ Models loaded successfully despite error")
                    return True
                except Exception as load_error:
                    logger.error("Failed to load models: %s", str(load_error))
                    return False
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
            if hasattr(self, 'user_processing_enabled'):
                self.sink.parent = type('obj', (object,), {})()
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
                        import server as server_module

                    if hasattr(server_module, 'user_processing_enabled'):
                        self.sink.parent = server_module
                except ImportError:
                    logger.debug("Could not import server module")

            # Create a robust audio callback that handles Discord.py crashes
            def audio_callback(_sink, *args):
                async def robust_process():
                    try:
                        logger.debug("Audio callback executed successfully")
                    except Exception as e:
                        logger.error("Error in audio callback: %s", str(e))
                    return
                return robust_process()

            # Monkey patch Discord's voice client to handle decryption errors
            original_unpack_audio = voice_client.unpack_audio

            def safe_unpack_audio(data):
                try:
                    return original_unpack_audio(data)
                except (IndexError, struct.error, ValueError, TypeError, AttributeError) as e:
                    logger.warning(
                        "Discord audio packet corruption (ignoring): %s", str(e))
                    return
                except Exception as e:
                    logger.error(
                        "Unexpected Discord audio error (ignoring): %s", str(e))
                    return

            # Apply the patch
            voice_client.unpack_audio = safe_unpack_audio
            logger.debug("Applied Discord audio corruption protection")

            # Start recording with our bullet-proof callback
            voice_client.start_recording(self.sink, audio_callback)
            logger.debug("Recording started successfully")

            self.is_listening = True
            logger.debug("Started listening in channel: %s",
                         voice_client.channel.name)

            # Broadcast listening status to all clients
            if self.websocket_handler:
                await self.websocket_handler.broadcast_listen_status(True)

            return True, "Started listening"
        except (ConnectionError, AttributeError, ValueError) as e:
            logger.error("Error in start_listening: %s", str(e))
            return False, f"Error: {str(e)}"

    async def stop_listening(self, voice_client):
        """Stop listening and processing audio"""
        logger.debug("stop_listening called. Voice client valid: %s",
                     voice_client is not None)
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
            await asyncio.sleep(CLEANUP_DELAY)

            # Clean up sink resources
            if self.sink:
                try:
                    logger.debug("Cleaning up sink")
                    self.sink.cleanup()
                    logger.debug("Sink cleanup completed")
                except (AttributeError, RuntimeError) as e:
                    logger.error("Error during sink cleanup: %s", str(e))
                self.sink = None
                logger.debug("Sink reference cleared")

            self.is_listening = False
            logger.debug("Stopped listening")

            # Broadcast listening status to all clients
            if self.websocket_handler:
                await self.websocket_handler.broadcast_listen_status(False)

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
            logger.debug(
                "🔄 process_audio_callback called for user %s with message_type: %s", user_id, message_type)

            # If we're called with a message_type, it's a direct text message
            if message_type is not None:
                text_content = audio_file
                logger.info("📝 Received %s for user %s: %s",
                            message_type, user_id, text_content)

                # Forward to translation callback
                if self.translation_callback:
                    await self.translation_callback(user_id, text_content, message_type)
                else:
                    logger.warning("No translation callback available")
                return

            # If we get here, it's an audio file processing request (legacy path)
            logger.warning(
                "⚠️ Legacy audio file processing path called - this should not happen in current design")
            try:
                import utils
                # Get current queue size from sink if available for smart routing
                current_queue_size = 0
                if hasattr(self, 'sink') and hasattr(self.sink, 'workers'):
                    current_queue_size = self.sink.workers.queue.qsize()

                # Get transcription and detected language with smart routing
                transcribed_text, detected_language = await utils.transcribe(
                    audio_file, current_queue_size=current_queue_size
                )

                # Skip processing if transcription was empty
                if not transcribed_text:
                    return

                # Create transcription message
                logger.info("Sending transcription for user %s: %s",
                            user_id, transcribed_text)
                await self.translation_callback(user_id, transcribed_text, message_type="transcription")

                # Determine if translation is needed
                needs_translation = await translation.should_translate(transcribed_text, detected_language)

                if needs_translation:
                    translated_text = await utils.translate(transcribed_text)
                    if translated_text:
                        await self.translation_callback(user_id, translated_text, message_type="translation")
                else:
                    # Send original text as translation
                    await self.translation_callback(user_id, transcribed_text, message_type="translation")
            finally:
                # Always delete the file when done with it
                await self._force_delete_file(audio_file)

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error in process_audio_callback: %s", str(e))
            logger.debug("Process audio callback error traceback: %s",
                         traceback.format_exc())
            # Make sure to clean up even if exception occurs
            if message_type is None:  # Only try to delete if it was an audio file
                await self._force_delete_file(audio_file)

    async def _force_delete_file(self, file_path):
        """Forcefully delete a file with multiple retries."""
        if not file_path or not os.path.exists(file_path):
            return False

        # Try to delete the file with multiple retries
        for attempt in range(MAX_DELETE_RETRIES):
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
        if audio_file and os.path.exists(audio_file):
            asyncio.create_task(self._force_delete_file(audio_file))

    def cleanup(self):
        """Clean up resources"""
        # Stop all recordings
        for guild_id, voice_client in list(self.active_voices.items()):
            try:
                if voice_client.is_connected():
                    asyncio.create_task(self.stop_listening(voice_client))
            except (AttributeError, ConnectionError) as e:
                logger.error(
                    "Error during cleanup for guild %s: %s", guild_id, str(e))

        # Clean up sink
        if self.sink:
            self.sink.cleanup()
            self.sink = None

        self.active_voices = {}
        self.is_listening = False

        try:
            if hasattr(self, 'active_voices'):
                self.active_voices.clear()
            if hasattr(self, 'connected_users'):
                self.connected_users.clear()
            if hasattr(self, 'user_processing_enabled'):
                self.user_processing_enabled.clear()
            logger.info("✅ VoiceTranslator cleanup completed")
        except Exception as e:
            logger.error("❌ Error during VoiceTranslator cleanup: %s", str(e))
