"""
Main server module for Piano Bot Translator.

Simplified server with modular architecture for better maintainability.
"""

import asyncio
import sys
import traceback
import uvicorn
import discord
import utils
from discord.ext import commands
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from bot_manager import DiscordBotManager
from websocket_handler import WebSocketManager
from translator import VoiceTranslator
from cleanup import clean_temp_files
from logging_config import get_logger

logger = get_logger(__name__)

# Discord bot setup

intents = discord.Intents.default()
intents.voice_states = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Global instances
bot_manager = None
websocket_manager = None
voice_translator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global bot_manager, websocket_manager, voice_translator

    # Startup
    logger.info("🚀 Starting Discord Voice Translator Server...")

    try:
        # Clean up any leftover audio files first
        logger.info("🧹 Cleaning up any leftover audio files...")
        cleaned_files = clean_temp_files()

        # CRITICAL FIX: Handle None return value from cleanup function
        if cleaned_files is not None and cleaned_files > 0:
            logger.info("✅ Cleaned up %d leftover audio files", cleaned_files)
        else:
            logger.info("✅ No leftover audio files found")

        # Initialize bot manager
        bot_manager = DiscordBotManager()

        # Initialize WebSocket manager
        websocket_manager = WebSocketManager(bot_manager)

        # Create translation callback
        async def translation_callback(user_id, text, message_type="transcription"):
            await websocket_manager.broadcast_translation(user_id, text, message_type)

        # Create voice translator
        voice_translator = VoiceTranslator(translation_callback)

        # Set voice translator in bot manager
        bot_manager.set_voice_translator(voice_translator)

        # Create Discord bot
        bot = bot_manager.create_bot(translation_callback)

        # Load models
        logger.info("Loading transcription models...")
        await voice_translator.load_models()
        logger.info("✅ Models loaded successfully")

        # Start model warm-up
        logger.info("Starting model warm-up...")
        asyncio.create_task(warm_up_models())

        # Start Discord bot
        logger.info("Starting Discord bot...")
        # CRITICAL FIX: Start bot in background task instead of blocking
        with open('token.txt', 'r', encoding='utf-8') as f:
            bot_token = f.read().strip()

        # Start bot as background task
        asyncio.create_task(bot_manager.start_bot(bot_token))
        logger.info("✅ Discord bot startup initiated")

        logger.info("🎉 Server startup completed successfully!")

    except Exception as e:
        logger.error("❌ Server startup failed: %s", str(e))
        logger.error("❌ Startup traceback: %s", traceback.format_exc())

    yield

    # Shutdown
    logger.info("🛑 Shutting down Discord Voice Translator Server...")

    try:
        if bot_manager:
            await bot_manager.stop_bot()
        if voice_translator:
            voice_translator.cleanup()
        logger.info("✅ Server shutdown completed")

    except Exception as e:
        logger.error("❌ Shutdown error: %s", str(e))


async def warm_up_models():
    """Warm up transcription models in the background."""
    try:
        logger.info("🔥 Starting background model warm-up...")
        await utils.warm_up_pipeline()
        logger.info("🎯 Model warm-up completed successfully!")

        # Broadcast ready status to all connected clients
        if websocket_manager:
            await websocket_manager.broadcast_bot_status(True)

    except Exception as e:
        logger.error("❌ Model warm-up failed: %s", str(e))
        logger.error("❌ Warm-up traceback: %s", traceback.format_exc())

        # Still broadcast that bot is ready (models may work anyway)
        if websocket_manager:
            await websocket_manager.broadcast_bot_status(True)


# Create FastAPI app
app = FastAPI(
    title="Piano Bot Translator",
    description="Discord Voice Translation Server",
    version="2.0.0",
    lifespan=lifespan
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend communication."""
    if websocket_manager:
        await websocket_manager.handle_connection(websocket)
    else:
        await websocket.close(code=1011, reason="Server not ready")


# Mount static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


# Serve frontend files
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html")


# Discord bot events
@bot.event
async def on_ready():
    """Bot ready event."""
    logger.info("🎯 Discord bot is ready! Logged in as %s", bot.user)

    # Set up the voice translator with WebSocket handler
    global voice_translator
    if voice_translator:
        voice_translator.set_websocket_handler(websocket_manager)


@bot.event
async def on_voice_state_update(member, before, after):
    """Handle voice state updates for user tracking."""
    global voice_translator
    if voice_translator:
        await voice_translator.handle_voice_state_update(member, before, after)


# Initialize voice translator
async def initialize_translator():
    """Initialize the voice translator."""
    global voice_translator
    voice_translator = VoiceTranslator(None)  # Callback will be set later
    voice_translator.set_websocket_handler(websocket_manager)
    return voice_translator


def read_token():
    """Read Discord bot token from token.txt file."""
    token_file = Path("token.txt")
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                logger.info("✅ Bot token found in token.txt")
                return token
            else:
                logger.error("❌ token.txt file is empty!")
        except Exception as e:
            logger.error("❌ Error reading token.txt: %s", str(e))
    else:
        logger.error("❌ token.txt file not found!")

    return None


async def main():
    """Main application entry point."""
    logger.info("🚀 Starting Discord Voice Translator Server...")

    # Read Discord bot token
    bot_token = read_token()
    if not bot_token:
        logger.error("❌ Cannot start without a valid Discord bot token!")
        logger.error(
            "❌ Please create a token.txt file with your Discord bot token")
        return

    try:
        # REMOVED: Duplicate model warmup - this is now handled in lifespan()
        # Initialize translator - this is also handled in lifespan()
        # await initialize_translator()
        # await utils.warm_up_pipeline()

        # Configure and start web server
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)

        logger.info("🌐 Server running at http://127.0.0.1:8000")
        logger.info("🎵 Discord Voice Translator is ready!")

        # Start server - this will run the FastAPI app with lifespan
        await server.serve()

    except Exception as e:
        logger.error("❌ Error in main: %s", str(e))
        logger.error("❌ Main traceback: %s", traceback.format_exc())
        raise


if __name__ == "__main__":
    # CRITICAL FIX: Logging is now automatically set up when logging_config is imported
    logger.info("🎹 Starting Piano Bot Translator Server...")

    # Check for Discord token - only use token.txt
    bot_token = read_token()
    if not bot_token:
        logger.error(
            "❌ Please create a token.txt file with your Discord bot token")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error("❌ Fatal error: %s", str(e))
        logger.error("❌ Fatal traceback: %s", traceback.format_exc())
        sys.exit(1)
