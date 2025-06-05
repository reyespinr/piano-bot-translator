"""
server.py - Main FastAPI and Discord bot server module for the Piano Bot Translator.

This module sets up the FastAPI app, manages Discord bot events, handles WebSocket connections
for the frontend, and coordinates audio translation and user state for the Discord voice channel.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import discord
from discord import VoiceClient
from translator import VoiceTranslator
from cleanup import clean_temp_files
import utils
from logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Define lifespan context manager for FastAPI


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pylint: disable=unused-argument
    """FastAPI lifespan context manager for startup and shutdown tasks.

    Initializes the translator, loads models, cleans up temp files, starts the Discord bot,
    and ensures proper shutdown of resources.
    """
    # Startup
    try:
        clean_temp_files()
    except (OSError, ImportError) as e:
        logger.warning("Could not run cleanup utility: %s", e)

    # Initialize the translator
    state.translator = VoiceTranslator(broadcast_translation)
    await state.translator.load_models()
    await utils.warm_up_pipeline()

    try:
        with open("token.txt", "r", encoding="utf-8") as f:
            bot_token = f.read().strip()
        if not bot_token:
            logger.warning("token.txt file is empty")
            yield
            return
    except FileNotFoundError:
        logger.warning("token.txt file not found")
        yield
        return

    asyncio.create_task(bot.start(bot_token))
    yield

    # Shutdown
    if state.is_listening and state.connected_channel is not None:
        if hasattr(state.connected_channel, 'guild'):
            guild_id = state.connected_channel.guild.id
            if guild_id in state.voice_clients:
                try:
                    await state.translator.stop_listening(state.voice_clients[guild_id])
                except (RuntimeError, AttributeError) as e:
                    logger.error(
                        "Error stopping listening during shutdown: %s", e)
    if state.translator:
        state.translator.cleanup()

# Create the FastAPI app with the lifespan
app = FastAPI(lifespan=lifespan)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Discord bot configuration
intents = discord.Intents.default()
intents.voice_states = True
intents.message_content = True
intents.members = True  # This is already set, but we need to make a bigger change

# Create the bot with proper intents - change Bot to Client for better member access
bot = discord.Client(intents=intents)  # Change from Bot to Client

# Add the on_ready event handler for discord.Client


@bot.event
async def on_ready():
    """Discord bot ready event.

    Called when the bot has successfully connected to Discord and is ready to operate.
    """
    state.bot_ready = True
    logger.info("Bot is ready as %s", bot.user)
    for connection in active_connections:
        await connection.send_json({
            "type": "bot_status",
            "ready": True
        })


@dataclass
class BotServerState:  # pylint: disable=too-many-instance-attributes
    """Encapsulates all shared state for the Discord bot and FastAPI server."""
    bot_ready: bool = False
    connected_channel: Optional[discord.VoiceChannel] = None
    translations: list = field(default_factory=list)
    voice_clients: Dict[int, VoiceClient] = field(default_factory=dict)
    connected_users: list = field(default_factory=list)
    user_processing_enabled: dict = field(default_factory=dict)
    is_listening: bool = False
    # Audio processing state
    translator: Optional[VoiceTranslator] = None
    last_toggle_time: float = 0.0


# Constants for timing control
TOGGLE_COOLDOWN = 1.0  # seconds

state = BotServerState()

# WebSocket connection handling


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend clients.

    Accepts connections, sends initial state, and dispatches commands from the frontend.
    """
    await websocket.accept()
    active_connections.append(websocket)    # Send initial state to new client
    await websocket.send_json({
        "type": "status",
        "bot_ready": state.bot_ready,
        "connected_channel": str(state.connected_channel) if state.connected_channel else None,
        "translations": state.translations[-20:],  # Send last 20 translations
        # CRITICAL FIX: Include current listening state
        "is_listening": state.is_listening,
        "users": state.connected_users,  # Include current user list
        "enabled_states": state.user_processing_enabled  # Include user toggle states
    })

    try:
        while True:
            data = await websocket.receive_text()
            await handle_command(json.loads(data), websocket)
    except WebSocketDisconnect as exc:
        logger.info("WebSocket disconnected: %s", exc)
        if websocket in active_connections:
            active_connections.remove(websocket)
    # pylint: disable-next=broad-except
    except Exception as exc:
        logger.error("Unexpected error in websocket_endpoint: %s",
                     exc, exc_info=True)
        if websocket in active_connections:
            active_connections.remove(websocket)


# pylint: disable=too-many-locals, too-many-branches
async def handle_command(data, websocket):
    """Handle commands received from the frontend WebSocket client.

    Supports joining/leaving channels, toggling users, toggling listening,
    and fetching guild/channel info.
    """
    command = data.get("command")

    if command == "join_channel":
        channel_id = int(data.get("channel_id"))
        success, message = await join_voice_channel(channel_id)
        await websocket.send_json({
            "type": "response",
            "command": "join_channel",
            "success": success,
            "message": message
        })

    elif command == "leave_channel":
        success, message = await leave_voice_channel()
        await websocket.send_json({
            "type": "response",
            "command": "leave_channel",
            "success": success,
            "message": message
        })

    elif command == "get_guilds":
        # Send list of available Discord servers (guilds)
        guilds_data = []
        for guild in bot.guilds:
            guilds_data.append({
                "id": str(guild.id),
                "name": guild.name
            })
        await websocket.send_json({
            "type": "guilds",
            "guilds": guilds_data
        })

    elif command == "get_channels":
        # Send list of voice channels for the specified guild
        guild_id = data.get("guild_id")
        if guild_id:
            guild = bot.get_guild(int(guild_id))
            if guild:
                voice_channels = []
                for channel in guild.channels:
                    if isinstance(channel, discord.VoiceChannel):
                        voice_channels.append({
                            "id": str(channel.id),
                            "name": channel.name
                        })
                await websocket.send_json({
                    "type": "channels",
                    "channels": voice_channels
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Guild not found"
                })

    elif command == "toggle_user":
        user_id = str(data.get("user_id"))  # Ensure user_id is a string
        enabled = data.get("enabled")

        # Update the user processing state
        state.user_processing_enabled[user_id] = enabled
        # Completely restart listening if active - the only solution that works reliably
        logger.debug("User %s processing set to %s", user_id, enabled)
        if state.is_listening and state.connected_channel:
            guild_id = state.connected_channel.guild.id
            if guild_id in state.voice_clients:
                logger.debug("Restarting listener for user %s toggle", user_id)
                await state.translator.stop_listening(state.voice_clients[guild_id])
                await asyncio.sleep(0.3)
                fresh_settings = {str(uid): bool(state)
                                  for uid, state in state.user_processing_enabled.items()}
                state.translator.user_processing_enabled = fresh_settings
                success, message = await state.translator.start_listening(
                    state.voice_clients[guild_id])
                logger.debug("Restart result: %s, %s", success, message)
                # pylint: disable=protected-access
                if (hasattr(state.voice_clients[guild_id], '_player')
                        and state.voice_clients[guild_id]._player):
                    sink = state.voice_clients[guild_id]._player.source
                    if hasattr(sink, 'parent') and sink.parent:
                        enabled_status = "enabled" if sink.parent.user_processing_enabled.get(
                            user_id, False) else "disabled"
                        logger.debug("User %s is now %s",
                                     user_id, enabled_status)
        # Notify all clients of the change
        for connection in active_connections:
            await connection.send_json({
                "type": "user_toggle",
                "user_id": user_id,
                "enabled": enabled
            })

        await websocket.send_json({
            "type": "response",
            "command": "toggle_user",
            "success": True
        })

    elif command == "toggle_listen":
        success, message, current_state = await toggle_listening()
        await websocket.send_json({
            "type": "response",
            "command": "toggle_listen",
            "success": success,
            "message": message,
            "is_listening": current_state
        })


async def broadcast_translation(user_id, text, message_type="translation"):
    """Send translation or message to all connected frontend clients.

    Broadcasts translation results or other messages to all active WebSocket connections.
    """
    try:
        # Get the user's display name
        user_name = "Unknown User"
        if state.connected_channel is not None and hasattr(state.connected_channel, 'guild'):
            member = state.connected_channel.guild.get_member(int(user_id))
            if member:
                user_name = member.display_name

        # Create the message
        message = {
            "user": user_name,
            "user_id": user_id,
            "text": text,
            "timestamp": asyncio.get_event_loop().time()
        }

        # Store in history (only store translations to avoid duplicates)
        if message_type == "translation":
            # Keep only the last 100 translations
            state.translations.append(message)
            if len(state.translations) > 100:
                state.translations.pop(0)

        logger.info("Broadcasting %s from %s: %s",
                    message_type, user_name, text)

        # Broadcast to all clients
        for connection in active_connections:
            try:
                await connection.send_json({
                    "type": message_type,
                    "data": message
                })
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Error sending to client: %s", e)

    except (OSError, ValueError, RuntimeError) as e:
        logger.error("Error in broadcast_translation: %s", e)
        logger.debug("Traceback: %s", traceback.format_exc())


# Add a toggle tracking timestamp - moved to constants section
# Constants for timing control
TOGGLE_COOLDOWN = 1.0  # seconds


async def toggle_listening():
    """Toggle the listening state for the bot with cooldown protection.

    Starts or stops audio processing in the connected Discord voice channel.
    """
    current_time = time.time()
    time_since_last = current_time - state.last_toggle_time

    # Check for rapid toggling (without debug spam)
    if time_since_last < TOGGLE_COOLDOWN:
        return True, "Toggle cooldown in effect", state.is_listening

    # Update last toggle time
    state.last_toggle_time = current_time

    try:
        if state.connected_channel is None or not hasattr(state.connected_channel, 'guild'):
            return False, "Not connected to a voice channel", False

        guild_id = state.connected_channel.guild.id

        if guild_id not in state.voice_clients:
            return False, "Voice client not found", False

        voice_client = state.voice_clients[guild_id]        # Toggle listening
        try:
            if state.is_listening:
                success, message = await state.translator.stop_listening(voice_client)
                if success:
                    state.is_listening = False
            else:
                # CRITICAL FIX: Create a fresh copy of the processing settings
                string_user_processing_enabled = {}
                for user_id, enabled in state.user_processing_enabled.items():
                    string_user_processing_enabled[str(user_id)] = enabled

                # Set in the translator
                state.translator.user_processing_enabled = string_user_processing_enabled.copy()
                success, message = await state.translator.start_listening(voice_client)

                if success:
                    state.is_listening = True
                    logger.info("Active user processing settings: %s",
                                string_user_processing_enabled)

        except (OSError, RuntimeError, ValueError) as e:
            logger.info("Error in translator methods: %s", e)
            return False, f"Error in translator: {e}", state.is_listening

        # Notify all clients of the change
        for connection in active_connections:
            await connection.send_json({
                "type": "listen_status",
                "is_listening": state.is_listening
            })

        return success, message, state.is_listening
    except (OSError, RuntimeError, ValueError) as e:
        logger.info("Error in toggle_listening: %s", e)
        return False, f"Error toggling listening: {e}", state.is_listening


async def leave_voice_channel():
    """Disconnect the bot from the current Discord voice channel and clean up state.

    Stops listening, disconnects the voice client, and notifies all frontend clients.
    """

    try:
        if state.connected_channel is None or not hasattr(state.connected_channel, 'guild'):
            return False, "Not connected to any voice channel"

        guild_id = state.connected_channel.guild.id

        # Store channel info for logging before we clear it
        channel_name = getattr(state.connected_channel, 'name', 'Unknown')
        # CRITICAL FIX: Stop listening BEFORE disconnecting voice client
        guild_name = getattr(
            getattr(state.connected_channel, 'guild', None), 'name', 'Unknown')
        if state.is_listening and guild_id in state.voice_clients:
            try:
                await state.translator.stop_listening(state.voice_clients[guild_id])
                logger.info("Stopped listening before leaving channel")
            except (OSError, RuntimeError, ValueError) as e:
                logger.info(
                    "Error stopping listening during disconnect: %s", str(e))

        # Always reset listening state when leaving channel
        if state.is_listening:
            state.is_listening = False
            logger.info("Reset is_listening state to False")

        if guild_id in state.voice_clients:
            await state.voice_clients[guild_id].disconnect()
            del state.voice_clients[guild_id]

        # Log successful disconnection
        logger.info(
            "Disconnected from voice channel '%s' in server '%s'", channel_name, guild_name)

        state.connected_channel = None
        state.connected_users = []

        # Notify clients of updated (empty) user list
        for connection in active_connections:
            await connection.send_json({
                "type": "users_update",
                "users": [],
                "enabled_states": state.user_processing_enabled
            })

        # CRITICAL FIX: Notify all clients that listening has stopped
        for connection in active_connections:
            await connection.send_json({
                "type": "listen_status",
                "is_listening": False
            })

        return True, "Disconnected from voice channel"
    except (OSError, RuntimeError, ValueError) as e:
        return False, f"Error leaving channel: {str(e)}"


# Add voice state update event to track users joining/leaving
@bot.event
async def on_voice_state_update(member, before, after):
    """Track users joining or leaving the connected Discord voice channel.

    Updates the user list and notifies frontend clients of user join/leave events.
    """
    # Only process if we're connected to a channel
    if not state.connected_channel:
        return

    # Check if this event is relevant to our connected channel
    our_channel_id = state.connected_channel.id
    before_channel_id = before.channel.id if before.channel else None
    after_channel_id = after.channel.id if after.channel else None

    # User joined our channel
    if after_channel_id == our_channel_id and before_channel_id != our_channel_id:
        if member.id != bot.user.id:  # Ignore bot itself
            user_data = {
                "id": str(member.id),
                "name": member.display_name,
                "avatar": str(member.avatar.url) if member.avatar else None
            }

            # Add to our tracking
            state.connected_users.append(user_data)

            # Initialize processing state
            if user_data["id"] not in state.user_processing_enabled:
                state.user_processing_enabled[user_data["id"]] = True

            logger.info("User '%s' joined voice channel", member.display_name)

            # Notify clients
            for connection in active_connections:
                await connection.send_json({
                    "type": "user_joined",
                    "user": user_data,
                    "enabled": state.user_processing_enabled.get(user_data["id"], True)
                })

    # User left our channel
    elif before_channel_id == our_channel_id and after_channel_id != our_channel_id:
        if member.id != bot.user.id:  # Ignore bot itself
            # Remove from our tracking
            user_id = str(member.id)
            state.connected_users = [
                u for u in state.connected_users if u["id"] != user_id]

            # Remove user processing settings for users who left
            if user_id in state.user_processing_enabled:
                del state.user_processing_enabled[user_id]
                logger.info(
                    "User '%s' left voice channel, removed processing settings",
                    member.display_name)

            # Notify clients
            for connection in active_connections:
                await connection.send_json({
                    "type": "user_left",
                    "user_id": user_id
                })


# Mount static files for the frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


async def join_voice_channel(channel_id):
    """Connect the bot to a specified Discord voice channel and set up audio processing.

    Handles switching channels, user state, and notifies frontend clients of changes.
    """
    try:
        channel = bot.get_channel(channel_id)

        if not channel:
            return False, f"Channel with ID {channel_id} not found"

        if not isinstance(channel, discord.VoiceChannel):
            return False, "The specified channel is not a voice channel"

        # Check if already connected to a voice channel
        if state.connected_channel is not None and hasattr(state.connected_channel, 'guild'):
            current_guild_id = state.connected_channel.guild.id

            # CRITICAL FIX: Stop listening BEFORE disconnecting when switching channels
            if current_guild_id in state.voice_clients:
                if state.is_listening:
                    try:
                        await state.translator.stop_listening(state.voice_clients[current_guild_id])
                        logger.info(
                            "Stopped listening before switching channels")
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.info(
                            "Error stopping listening during channel switch: %s", str(e))
                    # Reset listening state
                    state.is_listening = False
                    logger.info(
                        "Reset is_listening state to False during channel switch")
                    for connection in active_connections:
                        await connection.send_json({
                            "type": "listen_status",
                            "is_listening": False
                        })
                # Disconnect from the current voice channel
                await state.voice_clients[current_guild_id].disconnect()
                del state.voice_clients[current_guild_id]
                logger.info("Disconnected from voice channel '%s' in server '%s'",
                            getattr(state.connected_channel,
                                    'name', 'Unknown'),
                            getattr(getattr(state.connected_channel, 'guild', None),
                                    'name', 'Unknown'))
                state.connected_channel = None

        # Connect to the new voice channel
        voice_client = await channel.connect()
        state.voice_clients[channel.guild.id] = voice_client
        state.connected_channel = channel

        # Log successful connection
        logger.info("Connected to voice channel '%s' in server '%s'",
                    channel.name, channel.guild.name)

        # Update connected users list - exclude bot and clear old user processing settings
        state.connected_users = []
        # Clear user processing settings for users not in the new channel
        current_user_ids = set()

        for member in channel.members:
            if member.id != bot.user.id:
                member_data = {
                    "id": str(member.id),
                    "name": member.display_name,
                    "avatar": str(member.avatar.url) if member.avatar else None
                }
                state.connected_users.append(member_data)
                current_user_ids.add(str(member.id))

                # Initialize processing state if not already set
                if member_data["id"] not in state.user_processing_enabled:
                    state.user_processing_enabled[member_data["id"]] = True

        # Remove processing settings for users who are no longer in the channel
        users_to_remove = []
        for user_id in state.user_processing_enabled:
            if user_id not in current_user_ids:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del state.user_processing_enabled[user_id]
            logger.debug(
                "Removed user processing settings for user %s (no longer in channel)", user_id)

        logger.info("Connected users: %s", [user['name']
                    for user in state.connected_users])

        # Notify clients of updated user list
        for connection in active_connections:
            await connection.send_json({
                "type": "users_update",
                "users": state.connected_users,
                "enabled_states": state.user_processing_enabled
            })

        # Set up the translator to process audio
        state.translator.setup_voice_receiver(voice_client)

        return True, f"Connected to {channel.name}"
    except (OSError, RuntimeError, ValueError) as e:
        return False, f"Error joining channel: {str(e)}"


if __name__ == "__main__":
    import uvicorn

    # Try multiple ports to avoid permission issues
    ports_to_try = [3000, 5000, 8080, 8888, 9000]

    for port in ports_to_try:
        try:
            logger.info("Trying to start server on port %d...", port)
            uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)
            break
        except OSError as e:
            if "10013" in str(e) or "Address already in use" in str(e):
                logger.info(
                    "Port %d is not available, trying next port...", port)
                continue
            raise e
    else:
        logger.info("Could not start server on any of the attempted ports.")
        logger.info(
            "Try running as administrator or check your firewall settings.")
