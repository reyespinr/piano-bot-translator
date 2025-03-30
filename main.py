import discord
import asyncio
import sys
import logging
import os
import gui
from PyQt5.QtWidgets import QApplication, QMessageBox

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


async def main(bot):
    try:
        token = open("token.txt", "r").read()

        app = QApplication(sys.argv)
        bot_ui = gui.GUI(app, bot)
        asyncio.ensure_future(bot_ui.ready())
        asyncio.ensure_future(bot_ui.run_Qt())

        @bot.event
        async def on_ready():
            print(f'Logged in as {bot.user}')
            print("Bot is ready and waiting for voice state updates...")

            # Check if the bot is already connected to a voice channel
            if bot.voice_clients:
                vc = bot.voice_clients[0]
                bot_ui.vc = vc  # Assign the voice client to the GUI instance
                print(f"Bot is connected to voice channel: {vc.channel.name}")

                # Populate and display the connected users list
                bot_ui.connected_users = [
                    member for member in vc.channel.members if member.id != bot.user.id
                ]
                print(
                    f"Connected users: {[user.display_name for user in bot_ui.connected_users]}"
                )

        @bot.event
        async def on_voice_state_update(member, before, after):
            print("Voice state update detected...")  # Debugging statement

            # Handle the bot moving to a new channel
            if member.id == bot.user.id and before.channel != after.channel:
                print(
                    f"Bot moved to a new channel: {after.channel.name if after.channel else 'None'}"
                )
                if after.channel:
                    bot_ui.connected_users = [
                        member for member in after.channel.members if member.id != bot.user.id
                    ]
                    print(
                        f"Updated connected_users list for new channel: {[user.display_name for user in bot_ui.connected_users]}"
                    )
                else:
                    bot_ui.connected_users = []
                    print(
                        "Bot is no longer in a voice channel. Connected users cleared."
                    )

            # Ignore the bot itself for other updates
            if member.id == bot.user.id:
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
                            f"Updated connected_users list: {[user.display_name for user in bot_ui.connected_users]}"
                        )

                # Handle users leaving the channel
                if before.channel == current_channel and after.channel != current_channel:
                    if member in bot_ui.connected_users:
                        bot_ui.connected_users.remove(member)
                        print(
                            f"User {member.display_name} removed from connected_users list."
                        )
                        print(
                            f"Updated connected_users list: {[user.display_name for user in bot_ui.connected_users]}"
                        )

        await bot.start(token)

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

    except Exception:
        logging.exception("Error on main")

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
