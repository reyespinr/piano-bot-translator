import discord
import asyncio
import sys
import logging
import pyaudio
import os
import gui
from PyQt5.QtWidgets import QApplication, QMessageBox
from discord.sinks import WaveSink
import utils

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

            # Check if a channel is selected in the GUI
            selected_channel = bot_ui.connections[0].channels.currentData()
            if selected_channel is None:
                print("No channel selected in the GUI. Ignoring voice state update.")
                return

            # Proceed only if the selected channel matches the user's new channel
            if after.channel is not None and after.channel == selected_channel:
                print(
                    f"User {member} joined the selected voice channel: {after.channel}")
                if bot.voice_clients:
                    vc = bot.voice_clients[0]
                    if vc.channel != after.channel:
                        await vc.move_to(after.channel)
                        print(f"Moved to voice channel: {after.channel}")
                else:
                    vc = await after.channel.connect()
                    print(f"Connected to voice channel: {after.channel}")
                bot_ui.vc = vc  # Store the voice client in the GUI object

        async def start_listening(vc, gui_instance):
            """Start recording and processing audio from the voice channel."""
            sink = WaveSink()

            async def process_audio_callback(sink, channel):
                """Process audio from the sink."""
                await process_audio_from_sink(sink, channel, gui_instance)

            vc.start_recording(
                sink,
                process_audio_callback,  # Pass the coroutine directly
                None,
            )
            print("Started listening to the voice channel.")

        async def process_audio_from_sink(sink, _, gui_instance):
            """Process audio from the sink after recording."""
            for user_id, audio in sink.audio_data.items():
                temp_audio_file = f"{user_id}_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(audio.file.read())

                # Transcribe and translate the audio
                transcribed_text = await utils.transcribe(temp_audio_file)
                translated_text = await utils.translate(transcribed_text)

                # Update the GUI with the results
                gui_instance.update_text_display(
                    transcribed_text, translated_text)

                # Clean up the temporary file
                os.remove(temp_audio_file)

            print("Finished processing audio.")

        bot_ui.start_listening = start_listening  # Expose the function to the GUI

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
