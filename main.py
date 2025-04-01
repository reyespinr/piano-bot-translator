"""
Discord Voice Channel Translation Bot.

This module serves as the main entry point for the Discord voice translator bot.
It provides functionality to:
- Connect to Discord voice channels
- Listen to audio from users in the channel
- Transcribe and translate the speech in real-time
- Display transcriptions and translations in a GUI

The application uses PyQt5 for the GUI interface, Discord.py for the Discord API
connection, and integrates with whisper and translation APIs for speech processing.

The bot automatically tracks users joining and leaving voice channels and manages
the recording and processing of audio data from multiple speakers.
"""
import asyncio
import sys
import logging
import os
import discord
# pylint: disable=no-name-in-module
from PyQt5.QtWidgets import QMessageBox, QApplication
# pylint: enable=no-name-in-module
import gui
import utils

# Error logging setup
error_formatter = logging.Formatter(
    fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
error_handler = logging.FileHandler("DAP_errors.log", delay=True)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(error_formatter)
base_logger = logging.getLogger()
base_logger.addHandler(error_handler)

# Verbose logs
debug_formatter = logging.Formatter(
    fmt="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
)
debug_handler = logging.FileHandler(
    filename="discord.log", encoding="utf-8", mode="w"
)
debug_handler.setFormatter(debug_formatter)
debug_logger = logging.getLogger("discord")
debug_logger.setLevel(logging.DEBUG)
debug_logger.addHandler(debug_handler)

# Load opus
opus_path = os.path.join(os.path.dirname(__file__), 'libopus.dll')
discord.opus.load_opus(opus_path)

# Main function


async def main(client):
    """Initialize and run the Discord bot with GUI interface.

    This is the primary entry point for the application. It handles:
    - Reading the Discord authentication token
    - Setting up the PyQt5 GUI
    - Registering Discord event handlers
    - Error handling for common startup issues

    The function sets up event handlers to track users entering and leaving
    voice channels, and to update the GUI with the current channel state.

    Args:
        client (discord.Client): The Discord client instance to use for the bot

    Raises:
        FileNotFoundError: If the token.txt file cannot be found
        discord.errors.LoginFailure: If the provided token is invalid
        Exception: For any other errors that occur during execution
    """
    try:
        token = open("token.txt", "r", encoding="utf-8").read()

        # Warm up the ML pipeline to reduce first-use latency
        await utils.warm_up_pipeline()

        # Initialize the GUI
        app = QApplication(sys.argv)
        bot_ui = gui.GUI(app, client)
        asyncio.ensure_future(bot_ui.ready())
        asyncio.ensure_future(bot_ui.run_Qt())

        @client.event
        async def on_ready():
            print(f'Logged in as {client.user}')
            print("Bot is ready and waiting for voice state updates...")
            # Check if the bot is already connected to a voice channel
            if client.voice_clients:
                vc = client.voice_clients[0]
                bot_ui.vc = vc  # Assign the voice client to the GUI instance
                print(f"Bot is connected to voice channel: {vc.channel.name}")
                # Populate and display the connected users list
                bot_ui.connected_users = [
                    member for member in vc.channel.members if member.id != client.user.id
                ]
                print(
                    f"Connected users: {[user.display_name for user in bot_ui.connected_users]}"
                )

        @client.event
        async def on_voice_state_update(member, before, after):
            print(
                f"Voice state update: {member.display_name}"
                f"moved {before.channel} -> {after.channel}")

            # Get current voice channel if connected
            current_channel = None
            if bot_ui.vc and bot_ui.vc.channel:
                current_channel = bot_ui.vc.channel
                print(f"Bot is currently in: {current_channel.name}")

            # Handle the bot moving to a new channel
            if member.id == client.user.id:
                print(
                    f"Bot voice state changed: {before.channel} -> {after.channel}")

                if after.channel:
                    print(f"Bot is now in channel: {after.channel.name}")
                    # Clear previous users list completely
                    bot_ui.connected_users = []
                    # Add all members in the new channel except the bot
                    bot_ui.connected_users = [
                        m for m in after.channel.members if m.id != client.user.id
                    ]
                    # Update the UI
                    bot_ui.update_connected_users(bot_ui.connected_users)
                else:
                    # Bot left all voice channels
                    bot_ui.connected_users = []
                    bot_ui.update_connected_users([])
                    print("Bot left all voice channels, cleared user list")
                return  # Important to return here to avoid confusion with other state changes

            # Skip if not in a voice channel
            if not current_channel:
                print("Bot is not in a voice channel, skipping member state update")
                return

            # IMPORTANT: Always rebuild the user list completely when anyone's voice state changes
            # This ensures we never have stale/outdated entries
            if current_channel:
                # Rebuild the entire list from current channel members
                bot_ui.connected_users = [
                    m for m in current_channel.members if m.id != client.user.id
                ]
                print(
                    f"Rebuilt user list: {[user.display_name for user in bot_ui.connected_users]}")
                # Update the UI with refreshed list
                bot_ui.update_connected_users(bot_ui.connected_users)

        await client.start(token)

    except FileNotFoundError:
        msg = QMessageBox()
        msg.setWindowTitle("Token Error")
        msg.setText("No Token Provided")
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    except discord.errors.LoginFailure:
        msg = QMessageBox()
        msg.setWindowTitle("Login Failed")
        msg.setText("Please check if the token is correct")
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    except (discord.errors.HTTPException, discord.errors.GatewayNotFound,
            discord.errors.ConnectionClosed) as e:
        logging.error("Discord connection error: %s", str(e))
        msg = QMessageBox()
        msg.setWindowTitle("Connection Error")
        msg.setText(f"Failed to connect to Discord: {str(e)}")
        msg.setIcon(QMessageBox.Critical)
        msg.exec()

    except asyncio.exceptions.CancelledError:
        logging.info("Asyncio task cancelled")

    # pylint: disable=broad-except
    # This is a last-resort error handler to log any unexpected exceptions
    # and prevent the application from crashing silently
    except Exception as e:
        logging.exception("Unexpected error in main: %s", str(e))

# Run program
intents = discord.Intents.default()
intents.voice_states = True  # Ensure the bot can track voice state updates
intents.members = True  # Add this to allow fetching member information
bot = discord.Client(intents=intents)
loop = asyncio.get_event_loop_policy().get_event_loop()

try:
    loop.run_until_complete(main(bot))

except KeyboardInterrupt:
    print("Exiting...")
    loop.run_until_complete(bot.close())

    # This sleep prevents a bugged exception on Windows
    loop.run_until_complete(asyncio.sleep(1))
    loop.close()
