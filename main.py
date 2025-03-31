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
            print("Voice state update detected...")  # Debugging statement

            # Handle the bot moving to a new channel
            if member.id == client.user.id and before.channel != after.channel:
                print(
                    f"Bot moved to a new channel: {after.channel.name if after.channel else 'None'}"
                )
                if after.channel:
                    bot_ui.connected_users = [
                        member for member in after.channel.members if member.id != client.user.id
                    ]
                    print(
                        f"Bot moved to a new channel: "
                        f"{after.channel.name if after.channel else 'None'}"
                    )
                else:
                    bot_ui.connected_users = []
                    print(
                        "Bot is no longer in a voice channel. Connected users cleared."
                    )

            # Ignore the bot itself for other updates
            if member.id == client.user.id:
                print("Ignoring voice state update for the bot itself.")
                return

            # Check if the bot is connected to a voice channel
            if bot_ui.vc and bot_ui.vc.channel:
                current_channel = bot_ui.vc.channel

                # Handle users joining the channel
                if after.channel == current_channel:
                    if member not in bot_ui.connected_users:
                        bot_ui.connected_users.append(member)
                        print(
                            f"User {member.display_name} added to connected_users list."
                        )
                        print(
                            f"Updated connected_users list: "
                            f"{[user.display_name for user in bot_ui.connected_users]}"
                        )

                # Handle users leaving the channel
                if (before.channel == current_channel and
                        after.channel != current_channel):
                    if member in bot_ui.connected_users:
                        bot_ui.connected_users.remove(member)
                        print(
                            f"User {member.display_name} removed from connected_users list."
                        )
                        print(
                            f"Updated connected_users list: "
                            f"{[user.display_name for user in bot_ui.connected_users]}"
                        )

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
