import discord
import asyncio
import sys
import logging
import pyaudio
import os
import gui
from PyQt5.QtWidgets import QApplication, QMessageBox

# error logging
error_formatter = logging.Formatter(
    fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

error_handler = logging.FileHandler("DAP_errors.log", delay=True)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(error_formatter)

base_logger = logging.getLogger()
base_logger.addHandler(error_handler)

# verbose logs
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

# load opus
opus_path = os.path.join(os.path.dirname(__file__), 'libopus.dll')
discord.opus.load_opus(opus_path)

# main


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

        @bot.event
        async def on_voice_state_update(member, before, after):
            print("Voice state update detected...")  # Debugging statement
            if after.channel is not None and after.channel != before.channel:
                print(f"User {member} joined voice channel {after.channel}")
                if bot.voice_clients:
                    vc = bot.voice_clients[0]
                    if vc.channel != after.channel:
                        await vc.move_to(after.channel)
                        print(f"Moved to voice channel: {after.channel}")
                else:
                    vc = await after.channel.connect()
                    print(f"Connected to voice channel: {after.channel}")
                bot_ui.vc = vc  # Store the voice client in the GUI object

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

# run program
intents = discord.Intents.default()
intents.voice_states = True  # Ensure the bot has the necessary intent
bot = discord.Client(intents=intents)
loop = asyncio.get_event_loop_policy().get_event_loop()

try:
    loop.run_until_complete(main(bot))

except KeyboardInterrupt:
    print("Exiting...")
    loop.run_until_complete(bot.close())

    # this sleep prevents a bugged exception on Windows
    loop.run_until_complete(asyncio.sleep(1))
    loop.close()
