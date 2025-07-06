"""
Main server module for Piano Bot Translator.

Simplified server with modular architecture for better maintainability.
"""

import asyncio
import sys
import traceback
import signal
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
import discord
from discord.ext import commands
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model_manager import model_manager
from bot_manager import DiscordBotManager
from websocket_handler import WebSocketManager
from translation_service import VoiceTranslator
from cleanup import clean_temp_files
from logging_config import get_logger
import psutil
import threading
import time

logger = get_logger(__name__)

# Global exception handler for async tasks

# Global shutdown event
shutdown_event = asyncio.Event()


def handle_task_exception(task):
    """Handle exceptions from background tasks."""
    try:
        task.result()
    except asyncio.CancelledError:
        # This is normal during shutdown, don't log as error
        logger.debug("📝 Task cancelled during shutdown")
    except Exception as e:
        logger.error("❌ Background task failed: %s", str(e))
        logger.error("❌ Task exception traceback: %s", traceback.format_exc())
        # Don't re-raise - just log it


def create_task_with_exception_handling(coro, name=None):
    """Create a task with proper exception handling."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(handle_task_exception)
    return task

# Signal handlers for graceful shutdown


def signal_handler(signum, frame):
    """Handle termination signals."""
    global shutdown_event

    # Prevent multiple signal handling
    if shutdown_event and shutdown_event.is_set():
        logger.warning("🛑 Shutdown already in progress...")
        # If shutdown was already triggered but didn't complete, give it more time
        # Only force exit if we've been waiting too long
        if not hasattr(signal_handler, '_force_exit_started'):
            signal_handler._force_exit_started = True

            def force_exit():
                time.sleep(5)  # Give 5 more seconds
                logger.error("❌ Graceful shutdown timed out, forcing exit...")
                import os
                os._exit(1)

            force_exit_thread = threading.Thread(
                target=force_exit, daemon=True)
            force_exit_thread.start()
        return

    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")

    # Set the shutdown event
    if shutdown_event:
        shutdown_event.set()

    # Set a timeout for graceful shutdown
    def force_exit():
        time.sleep(8)  # Give 8 seconds for graceful shutdown
        logger.error("❌ Graceful shutdown timed out, forcing exit...")
        import os
        os._exit(1)

    force_exit_thread = threading.Thread(target=force_exit, daemon=True)
    force_exit_thread.start()


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

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
shutdown_event = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Handle application startup and shutdown events."""
    global bot_manager, websocket_manager, voice_translator
    # Startup
    logger.info("🚀 Starting Discord Voice Translator Server...")
    try:
        # Clean up any leftover audio files first
        logger.info("🧹 Cleaning up any leftover audio files...")
        clean_temp_files()  # This function doesn't return a value, just performs cleanup
        logger.info("✅ Audio file cleanup completed")

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
        # CRITICAL FIX: Verify the connection chain (updated for refactored structure)
        logger.info("🔗 Voice translator connected to bot manager")
        if (hasattr(bot_manager, 'voice_translator') and
            bot_manager.voice_translator and
            hasattr(bot_manager.voice_translator, 'state') and
            hasattr(bot_manager.voice_translator.state, 'websocket_handler') and
                bot_manager.voice_translator.state.websocket_handler):
            logger.info(
                "✅ Connection chain verified: bot_manager -> voice_translator -> state -> websocket_handler")
        else:
            logger.error("❌ Connection chain BROKEN!")
        global bot
        bot = bot_manager.create_bot(translation_callback)

        # Initialize models using the unified ModelManager
        logger.info("Initializing transcription models...")
        success = await model_manager.initialize_models(warm_up=False)
        if success:
            logger.info("✅ Models initialized successfully")
        else:
            logger.error("❌ Model initialization failed!")
            # Start model warm-up
            raise Exception("Model initialization failed")
        logger.info("Starting model warm-up...")

        async def warmup_task():
            try:
                logger.info("🔥 Starting background model warm-up...")
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
                # Still broadcast that bot is ready (models may work anyway)
                if websocket_manager:
                    await websocket_manager.broadcast_bot_status(True)

        create_task_with_exception_handling(warmup_task(), "model_warmup")

        # Start Discord bot
        logger.info("Starting Discord bot...")
        # Read token here instead of in main
        with open('token.txt', 'r', encoding='utf-8') as f:
            discord_bot_token = f.read().strip()

        # Start bot as background task
        create_task_with_exception_handling(
            bot_manager.start_bot(discord_bot_token), "discord_bot")
        logger.info("✅ Discord bot startup initiated")

        logger.info("🎉 Server startup completed successfully!")

    except Exception as e:
        logger.error("❌ Server startup failed: %s", str(e))
        logger.error("❌ Startup traceback: %s", traceback.format_exc())

    yield

    # Shutdown
    logger.info("🛑 Shutting down Discord Voice Translator Server...")

    try:
        # Cancel all running tasks first
        tasks = [t for t in asyncio.all_tasks(
        ) if t is not asyncio.current_task()]
        if tasks:
            logger.info(f"🔄 Cancelling {len(tasks)} running tasks...")
            for task in tasks:
                task.cancel()

            # Wait for tasks to complete cancellation
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ Some tasks did not complete cancellation within timeout")

        # Shutdown components in proper order
        if voice_translator:
            logger.info("🛑 Stopping voice translator...")
            voice_translator.cleanup()

        if bot_manager:
            logger.info("🛑 Stopping Discord bot...")
            await bot_manager.stop_bot()

        # Shutdown model manager
        if hasattr(model_manager, 'cleanup'):
            logger.info("🛑 Cleaning up models...")
            await model_manager.cleanup()

        logger.info("✅ Server shutdown completed")
    except Exception as e:
        logger.error("❌ Shutdown error: %s", str(e))


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
    try:
        logger.info("🎯 Discord bot is ready! Logged in as %s", bot.user)

        # CRITICAL FIX: Re-verify and re-establish the connection chain when bot is ready
        global voice_translator, websocket_manager
        if voice_translator and websocket_manager:
            voice_translator.set_websocket_handler(websocket_manager)
            logger.info(
                "🔗 Re-established WebSocket handler connection after bot ready")
            # Verify the connection is working
            if (hasattr(voice_translator.state, 'websocket_handler') and voice_translator.state.websocket_handler):
                logger.info("✅ WebSocket handler verified in voice translator")
            else:
                logger.error(
                    "❌ WebSocket handler NOT SET in voice translator!")
    except Exception as e:
        logger.error("❌ Error in on_ready event: %s", str(e))
        logger.error("❌ on_ready traceback: %s", traceback.format_exc())


@bot.event
async def on_error(event, *args, **kwargs):
    """Global Discord.py error handler."""
    logger.error("❌ Discord.py error in event %s: %s", event, str(args))
    logger.error("❌ Discord.py error traceback: %s", traceback.format_exc())


@bot.event
async def on_disconnect():
    """Handle bot disconnection."""
    logger.warning("⚠️ Discord bot disconnected!")


@bot.event
async def on_resumed():
    """Handle bot reconnection."""
    logger.info("🔄 Discord bot reconnected!")


# Initialize voice translator
async def initialize_translator():
    """Initialize the voice translation_service."""
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
            logger.error("❌ token.txt file is empty!")
        except Exception as e:
            logger.error("❌ Error reading token.txt: %s", str(e))
    else:
        logger.error("❌ token.txt file not found!")

    return None


# Add process monitoring
def monitor_process_health():
    """Monitor process health in background thread."""
    while True:
        try:            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"⚠️ High memory usage: {memory_percent}%")

            # Check if we have GPU
            try:
                import torch
                if torch.cuda.is_available():
                    # Get current device
                    current_device = torch.cuda.current_device()
                    device_count = torch.cuda.device_count()

                    # Get PyTorch allocated memory
                    gpu_memory = torch.cuda.memory_allocated(
                        current_device) / 1024**3
                    gpu_reserved = torch.cuda.memory_reserved(
                        current_device) / 1024**3

                    # Try to get total GPU memory
                    try:
                        device_props = torch.cuda.get_device_properties(
                            current_device)
                        total_memory = device_props.total_memory / 1024**3
                        device_name = device_props.name

                        logger.debug(
                            f"GPU Memory (Device {current_device}/{device_count} - {device_name}): "
                            f"{gpu_memory:.2f}GB allocated / {gpu_reserved:.2f}GB reserved / {total_memory:.2f}GB total"
                        )

                        # Also log memory info for all available devices (only once per startup)
                        if not hasattr(monitor_process_health, '_gpu_devices_logged'):
                            for i in range(device_count):
                                props = torch.cuda.get_device_properties(i)
                                logger.info(
                                    f"🎮 GPU {i}: {props.name} - {props.total_memory / 1024**3:.2f}GB total")
                            monitor_process_health._gpu_devices_logged = True

                    except Exception as gpu_error:
                        logger.debug(
                            f"GPU Memory (Device {current_device}): {gpu_memory:.2f}GB allocated / {gpu_reserved:.2f}GB reserved (PyTorch only)")
                        logger.debug(f"GPU details error: {gpu_error}")

                    if gpu_memory > 7.5:  # Alert if using more than 7.5GB
                        logger.warning(
                            f"⚠️ High GPU memory usage: {gpu_memory:.2f}GB")
            except ImportError:
                pass

            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Process monitor error: {e}")
            time.sleep(60)  # Wait longer if error


# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_process_health, daemon=True)
monitor_thread.start()


async def main():
    """Main application entry point."""
    global shutdown_event
    shutdown_event = asyncio.Event()

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

        # Create server task
        server_task = asyncio.create_task(server.serve())

        # Create shutdown monitoring task
        async def shutdown_monitor():
            await shutdown_event.wait()
            logger.info("🛑 Shutdown event triggered, stopping server...")

            # Step 1: Immediately force cleanup of voice connections and workers
            if voice_translator:
                logger.info("🛑 Forcing voice translator cleanup...")
                try:
                    # Stop all transcription workers first
                    if hasattr(voice_translator, 'state') and hasattr(voice_translator.state, 'worker_manager'):
                        logger.info("🛑 Stopping transcription workers...")
                        voice_translator.state.worker_manager.cleanup()

                    # Force stop all voice connections
                    if hasattr(voice_translator, 'state') and hasattr(voice_translator.state, 'active_voices'):
                        for guild_id, voice_client in list(voice_translator.state.active_voices.items()):
                            try:
                                if voice_client and voice_client.is_connected():
                                    logger.info(
                                        f"🛑 Forcing disconnect from guild {guild_id}")
                                    # Stop recording first
                                    try:
                                        voice_client.stop_recording()
                                    except Exception:
                                        pass  # Ignore errors, recording may not be active
                                    # Then disconnect
                                    await voice_client.disconnect()
                            except Exception as e:
                                logger.error(
                                    f"❌ Error forcing disconnect from guild {guild_id}: {e}")

                    # Then run cleanup
                    voice_translator.cleanup()
                except Exception as e:
                    logger.error(
                        f"❌ Error during voice translator cleanup: {e}")

            # Step 2: Force shutdown the Discord bot
            if bot_manager:
                logger.info("🛑 Forcing Discord bot shutdown...")
                try:
                    # First disconnect all voice clients
                    if bot_manager.bot and bot_manager.bot.voice_clients:
                        for vc in bot_manager.bot.voice_clients:
                            try:
                                if vc.is_connected():
                                    # Stop recording first
                                    try:
                                        vc.stop_recording()
                                    except Exception:
                                        pass  # Ignore errors, recording may not be active
                                    # Then disconnect
                                    await vc.disconnect()
                            except Exception as e:
                                logger.error(
                                    f"❌ Error disconnecting voice client: {e}")

                    # Then stop the bot
                    await bot_manager.stop_bot()
                except Exception as e:
                    logger.error(f"❌ Error during bot shutdown: {e}")

            # Step 3: Stop the uvicorn server
            server.should_exit = True

            logger.info(
                "🛑 Shutdown procedures completed, waiting for server to stop...")

        shutdown_task = asyncio.create_task(shutdown_monitor())

        # Wait for either server completion or shutdown signal
        try:
            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            logger.info(
                "🛑 Main task was cancelled, proceeding with shutdown...")
            pending = [server_task, shutdown_task]
            done = set()

        # Cancel any remaining tasks aggressively
        for task in pending:
            if not task.cancelled():
                task.cancel()

        # Wait for cancellation to complete, but with a timeout
        if pending:
            try:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "⚠️ Some tasks did not cancel within timeout, forcing shutdown...")
            except asyncio.CancelledError:
                logger.info("🛑 Task cancellation was cancelled, proceeding...")
            except Exception as e:
                logger.warning("⚠️ Error during task cancellation: %s", str(e))

        # Final cleanup - cancel all remaining tasks
        try:
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                logger.info(f"🛑 Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()

                # Wait briefly for tasks to cancel
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "⚠️ Some tasks did not cancel, but proceeding with shutdown...")
                except asyncio.CancelledError:
                    logger.info("🛑 Final cleanup was cancelled, proceeding...")
                except Exception as e:
                    logger.warning(
                        "⚠️ Error during final task cancellation: %s", str(e))
        except Exception as e:
            logger.warning("⚠️ Error during final cleanup: %s", str(e))

        logger.info("✅ Server shutdown completed gracefully")

    except asyncio.CancelledError:
        logger.info("🛑 Main function was cancelled, shutdown complete")
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
        logger.info("🛑 Received keyboard interrupt, shutdown completed")
    except asyncio.CancelledError:
        logger.info("🛑 Application was cancelled, shutdown completed")
    except Exception as e:
        logger.error("❌ Fatal error: %s", str(e))
        logger.error("❌ Fatal traceback: %s", traceback.format_exc())
        sys.exit(1)
