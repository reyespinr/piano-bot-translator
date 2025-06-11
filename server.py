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

        # CRITICAL FIX: Set the WebSocket handler in voice translator IMMEDIATELY
        voice_translator.set_websocket_handler(websocket_manager)
        logger.info("🔗 WebSocket handler connected to voice translator")

        # Set voice translator in bot manager
        bot_manager.set_voice_translator(voice_translator)
        logger.info("🔗 Voice translator connected to bot manager")

        # CRITICAL FIX: Verify the connection chain
        if (hasattr(bot_manager, 'voice_translator') and
            bot_manager.voice_translator and
            hasattr(bot_manager.voice_translator, 'websocket_handler') and
                bot_manager.voice_translator.websocket_handler):
            logger.info(
                "✅ Connection chain verified: bot_manager -> voice_translator -> websocket_handler")
        else:
            # Create Discord bot
            logger.error("❌ Connection chain BROKEN!")
        bot = bot_manager.create_bot(translation_callback)

        # Initialize models using the unified ModelManager
        logger.info("Initializing transcription models...")
        from model_manager import model_manager
        success = await model_manager.initialize_models(warm_up=False)
        if success:
            logger.info("✅ Models initialized successfully")
        else:
            logger.error("❌ Model initialization failed!")
            raise Exception("Model initialization failed")

        # Start model warm-up
        logger.info("Starting model warm-up...")
        asyncio.create_task(warm_up_models())

        # Start Discord bot
        logger.info("Starting Discord bot...")
        # Read token here instead of in main
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
    """Warm up transcription models in the background using the unified ModelManager."""
    try:
        logger.info("🔥 Starting background model warm-up...")
        from model_manager import model_manager

        # Use the model manager's warm-up functionality
        success = await model_manager.warm_up_models()

        if success:
            logger.info("🎯 Model warm-up completed successfully!")
        else:
            logger.warning(
                "⚠️ Model warm-up had some issues but models should still work")

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
    """WebSocket endpoint for admin frontend communication (full control)."""
    if websocket_manager:
        await websocket_manager.handle_admin_connection(websocket)
    else:
        await websocket.close(code=1011, reason="Server not ready")


@app.websocket("/ws/spectator")
async def websocket_spectator_endpoint(websocket: WebSocket):
    """WebSocket endpoint for spectator connections (read-only)."""
    if websocket_manager:
        await websocket_manager.handle_spectator_connection(websocket)
    else:
        await websocket.close(code=1011, reason="Server not ready")


# Serve frontend files BEFORE mounting static files
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html")


@app.get("/spectator")
async def serve_spectator():
    """Serve the spectator frontend page."""
    return FileResponse("frontend/spectator.html")


# Mount static files AFTER specific routes to avoid conflicts
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")


# Discord bot events
@bot.event
async def on_ready():
    """Bot ready event."""
    logger.info("🎯 Discord bot is ready! Logged in as %s", bot.user)

    # CRITICAL FIX: Re-verify and re-establish the connection chain when bot is ready
    global voice_translator, websocket_manager
    if voice_translator and websocket_manager:
        voice_translator.set_websocket_handler(websocket_manager)
        logger.info(
            "🔗 Re-established WebSocket handler connection after bot ready")

        # Verify the connection is working
        if (hasattr(voice_translator, 'websocket_handler') and voice_translator.websocket_handler):
            logger.info("✅ WebSocket handler verified in voice translator")
        else:
            logger.error("❌ WebSocket handler NOT SET in voice translator!")


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
    # REMOVED: Duplicate startup logging and token reading - this is now handled in lifespan()

    try:
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
