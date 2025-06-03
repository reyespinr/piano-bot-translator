from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import discord
from discord import VoiceClient
import os
import time
from typing import Dict, List
from contextlib import asynccontextmanager
# Import the toggle fix module to apply the patch
import toggle_fix  # This applies the monkey patch
from translator import VoiceTranslator
import utils  # Import utils for create_dummy_audio_file and transcribe
from logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Define lifespan context manager for FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global translator, bot_ready    # Clean up any stray WAV files from previous runs
    try:
        from cleanup import clean_temp_files
        clean_temp_files()
    except Exception as e:
        logger.warning(f"Could not run cleanup utility: {e}")

    # Initialize the translator
    translator = VoiceTranslator(broadcast_translation)
    await translator.load_models()

    # Force model loading by performing a small transcription
    logger.info("Initializing transcription pipeline...")
    # Create a dummy audio file and transcribe it to load the model
    temp_file = utils.create_dummy_audio_file()
    await utils.transcribe(temp_file)
    os.remove(temp_file)
    logger.info("Transcription pipeline ready")

    # Get token from token.txt file
    try:
        with open("token.txt", "r") as f:
            bot_token = f.read().strip()
        if not bot_token:
            logger.warning("token.txt file is empty")
            yield
            return
    except FileNotFoundError:
        logger.warning("token.txt file not found")
        yield
        return

    # Start the Discord bot
    asyncio.create_task(bot.start(bot_token))

    yield

    # Shutdown
    global is_listening    # Stop listening if active
    if is_listening and connected_channel:
        guild_id = connected_channel.guild.id
        if guild_id in voice_clients:
            try:
                await translator.stop_listening(voice_clients[guild_id])
            except Exception as e:
                logger.error(f"Error stopping listening during shutdown: {e}")

    # Clean up translator resources
    if translator:
        translator.cleanup()

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
    global bot_ready
    bot_ready = True
    logger.info(f"Bot is ready as {bot.user}")

    # Notify all connected clients that the bot is ready
    for connection in active_connections:
        await connection.send_json({
            "type": "bot_status",
            "ready": True
        })

# Bot state
bot_ready = False
connected_channel = None
translations = []
voice_clients: Dict[int, VoiceClient] = {}
# Add tracking for active users and their processing state
connected_users = []
user_processing_enabled = {}
is_listening = False

# Initialize the translator
translator = None

# WebSocket connection handling


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)    # Send initial state to new client
    await websocket.send_json({
        "type": "status",
        "bot_ready": bot_ready,
        "connected_channel": str(connected_channel) if connected_channel else None,
        "translations": translations[-20:],  # Send last 20 translations
        "is_listening": is_listening,  # CRITICAL FIX: Include current listening state
        "users": connected_users,  # Include current user list
        "enabled_states": user_processing_enabled  # Include user toggle states
    })

    try:
        while True:
            data = await websocket.receive_text()
            await handle_command(json.loads(data), websocket)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def handle_command(data, websocket):
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
        user_processing_enabled[user_id] = enabled
        logger.debug(f"User {user_id} processing set to {enabled}")

        # Completely restart listening if active - the only solution that works reliably
        if is_listening and connected_channel:
            guild_id = connected_channel.guild.id
            if guild_id in voice_clients:
                logger.debug(f"Restarting listener for user {user_id} toggle")
                # Stop listening
                await translator.stop_listening(voice_clients[guild_id])

                # Short pause to ensure cleanup
                await asyncio.sleep(0.3)

                # Create a fresh copy of settings
                fresh_settings = {}
                for uid, state in user_processing_enabled.items():
                    fresh_settings[str(uid)] = bool(state)

                # Update translator settings
                translator.user_processing_enabled = fresh_settings

                # Restart listening with fresh settings
                success, message = await translator.start_listening(voice_clients[guild_id])
                logger.debug(f"Restart result: {success}, {message}")

                # Verify the new settings were applied
                if hasattr(voice_clients[guild_id], '_player') and voice_clients[guild_id]._player:
                    sink = voice_clients[guild_id]._player.source
                    if hasattr(sink, 'parent') and sink.parent:
                        enabled_status = "enabled" if sink.parent.user_processing_enabled.get(
                            user_id, False) else "disabled"
                        logger.debug(f"User {user_id} is now {enabled_status}")

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
    """Send translation to all connected clients"""
    try:
        # Get the user's display name
        user_name = "Unknown User"
        if connected_channel:
            member = connected_channel.guild.get_member(int(user_id))
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
            translations.append(message)
            if len(translations) > 100:
                translations.pop(0)

        logger.info(f"Broadcasting {message_type} from {user_name}: {text}")

        # Broadcast to all clients
        for connection in active_connections:
            try:
                await connection.send_json({
                    "type": message_type,
                    "data": message
                })
            except Exception as e:
                logger.warning(f"Error sending to client: {e}")

    except Exception as e:
        logger.error(f"Error in broadcast_translation: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")


# Add a toggle tracking timestamp
last_toggle_time = 0
toggle_cooldown = 1.0  # 1 second cooldown


async def toggle_listening():
    """Toggle the listening state for the bot with cooldown protection"""
    global is_listening, last_toggle_time, user_processing_enabled

    current_time = time.time()
    time_since_last = current_time - last_toggle_time

    # Check for rapid toggling (without debug spam)
    if time_since_last < toggle_cooldown:
        return True, "Toggle cooldown in effect", is_listening

    # Update last toggle time
    last_toggle_time = current_time

    try:
        if not connected_channel:
            return False, "Not connected to a voice channel", False

        guild_id = connected_channel.guild.id

        if guild_id not in voice_clients:
            return False, "Voice client not found", False

        voice_client = voice_clients[guild_id]

        # Toggle listening
        try:
            if is_listening:
                success, message = await translator.stop_listening(voice_client)
                if success:
                    is_listening = False
            else:
                # CRITICAL FIX: Create a fresh copy of the processing settings
                string_user_processing_enabled = {}
                for user_id, enabled in user_processing_enabled.items():
                    string_user_processing_enabled[str(user_id)] = enabled

                # Set in the translator
                translator.user_processing_enabled = string_user_processing_enabled.copy()
                success, message = await translator.start_listening(voice_client)

                if success:
                    is_listening = True
                    print(
                        f"Active user processing settings: {string_user_processing_enabled}")

        except Exception as e:
            print(f"Error in translator methods: {e}")
            return False, f"Error in translator: {e}", is_listening

        # Notify all clients of the change
        for connection in active_connections:
            await connection.send_json({
                "type": "listen_status",
                "is_listening": is_listening
            })

        return success, message, is_listening
    except Exception as e:
        print(f"Error in toggle_listening: {e}")
        return False, f"Error toggling listening: {e}", is_listening


async def leave_voice_channel():
    global connected_channel, connected_users, is_listening

    try:
        if not connected_channel:
            return False, "Not connected to any voice channel"

        guild_id = connected_channel.guild.id

        # Store channel info for logging before we clear it
        channel_name = connected_channel.name
        guild_name = connected_channel.guild.name

        # CRITICAL FIX: Stop listening BEFORE disconnecting voice client
        if is_listening and guild_id in voice_clients:
            try:
                await translator.stop_listening(voice_clients[guild_id])
                print(f"Stopped listening before leaving channel")
            except Exception as e:
                print(f"Error stopping listening during disconnect: {str(e)}")

        # Always reset listening state when leaving channel
        if is_listening:
            is_listening = False
            print(f"Reset is_listening state to False")

        if guild_id in voice_clients:
            await voice_clients[guild_id].disconnect()
            del voice_clients[guild_id]

        # Log successful disconnection
        logger.info(
            f"Disconnected from voice channel '{channel_name}' in server '{guild_name}'")

        connected_channel = None
        connected_users = []

        # Notify clients of updated (empty) user list
        for connection in active_connections:
            await connection.send_json({
                "type": "users_update",
                "users": [],
                "enabled_states": user_processing_enabled
            })

        # CRITICAL FIX: Notify all clients that listening has stopped
        for connection in active_connections:
            await connection.send_json({
                "type": "listen_status",
                "is_listening": False
            })

        return True, "Disconnected from voice channel"
    except Exception as e:
        return False, f"Error leaving channel: {str(e)}"


# Add voice state update event to track users joining/leaving
@bot.event
async def on_voice_state_update(member, before, after):
    global connected_channel, connected_users, user_processing_enabled

    # Only process if we're connected to a channel
    if not connected_channel:
        return

    # Check if this event is relevant to our connected channel
    our_channel_id = connected_channel.id
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
            connected_users.append(user_data)

            # Initialize processing state
            if user_data["id"] not in user_processing_enabled:
                user_processing_enabled[user_data["id"]] = True

            # Notify clients
            for connection in active_connections:
                await connection.send_json({
                    "type": "user_joined",
                    "user": user_data,
                    "enabled": user_processing_enabled.get(user_data["id"], True)
                })

    # User left our channel
    elif before_channel_id == our_channel_id and after_channel_id != our_channel_id:
        if member.id != bot.user.id:  # Ignore bot itself
            # Remove from our tracking
            user_id = str(member.id)
            connected_users = [
                u for u in connected_users if u["id"] != user_id]

            # Notify clients
            for connection in active_connections:
                await connection.send_json({
                    "type": "user_left",
                    "user_id": user_id
                })


# Mount static files for the frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


async def join_voice_channel(channel_id):
    try:
        global connected_channel, connected_users, user_processing_enabled, is_listening
        channel = bot.get_channel(channel_id)

        if not channel:
            return False, f"Channel with ID {channel_id} not found"

        if not isinstance(channel, discord.VoiceChannel):
            # Check if already connected to a voice channel
            return False, "The specified channel is not a voice channel"
        if connected_channel:
            # Get the guild ID of the current connected channel
            current_guild_id = connected_channel.guild.id

            if current_guild_id in voice_clients:
                # CRITICAL FIX: Stop listening BEFORE disconnecting when switching channels
                if is_listening:
                    try:
                        await translator.stop_listening(voice_clients[current_guild_id])
                        print(f"Stopped listening before switching channels")
                    except Exception as e:
                        print(
                            f"Error stopping listening during channel switch: {str(e)}")

                    # Reset listening state
                    is_listening = False
                    print(f"Reset is_listening state to False during channel switch")

                    # Notify clients that listening has stopped
                    for connection in active_connections:
                        await connection.send_json({
                            "type": "listen_status",
                            "is_listening": False
                        })

                # Disconnect from the current voice channel
                await voice_clients[current_guild_id].disconnect()
                del voice_clients[current_guild_id]
                logger.info(
                    f"Disconnected from voice channel '{connected_channel.name}' in server '{connected_channel.guild.name}'")
                connected_channel = None

        # Connect to the new voice channel
        voice_client = await channel.connect()
        voice_clients[channel.guild.id] = voice_client
        connected_channel = channel

        # Log successful connection with channel and guild info
        logger.info(
            f"Connected to voice channel '{channel.name}' in server '{channel.guild.name}'")

        # Update connected users list - exclude bot
        # No need to call fetch_members, just use the cached members
        connected_users = []
        for member in channel.members:
            if member.id != bot.user.id:
                member_data = {
                    "id": str(member.id),
                    "name": member.display_name,
                    "avatar": str(member.avatar.url) if member.avatar else None
                }
                connected_users.append(member_data)

                # Initialize processing state if not already set
                if member_data["id"] not in user_processing_enabled:
                    user_processing_enabled[member_data["id"]] = True

        logger.info(
            f"Connected users: {[user['name'] for user in connected_users]}")

        # Notify clients of updated user list
        for connection in active_connections:
            await connection.send_json({
                "type": "users_update",
                "users": connected_users,
                "enabled_states": user_processing_enabled
            })

        # Set up the translator to process audio
        translator.setup_voice_receiver(voice_client)

        return True, f"Connected to {channel.name}"
    except Exception as e:
        return False, f"Error joining channel: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
