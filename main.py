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

            # Check if the bot is already connected to a voice channel
            if bot.voice_clients:
                vc = bot.voice_clients[0]
                bot_ui.vc = vc  # Assign the voice client to the GUI instance
                print(f"Bot is connected to voice channel: {vc.channel.name}")

                # Populate the connected_users list
                bot_ui.connected_users = [
                    member for member in vc.channel.members if member.id != bot.user.id
                ]
                print(
                    f"Connected users: {[user.display_name for user in bot_ui.connected_users]}")

                # Spawn listeners for all users in the channel if listening is toggled on
                if bot_ui.is_listening:
                    for member in vc.channel.members:
                        # Ignore the bot itself
                        if member.id == bot.user.id:
                            print(
                                f"Ignoring bot itself: {member.display_name} (ID: {member.id})")
                            continue

                        # Start a listener for the user
                        if member.id not in bot_ui.active_listeners:
                            print(
                                f"Spawning listener for user: {member.display_name} (ID: {member.id})")
                            bot_ui.active_listeners[member.id] = asyncio.create_task(
                                utils.user_listener(vc, member.id, bot_ui)
                            )
                            print(
                                f"Started listener for user: {member.display_name} (ID: {member.id})")

        @bot.event
        async def on_voice_state_update(member, before, after):
            print("Voice state update detected...")  # Debugging statement

            # Handle the bot moving to a new channel
            if member.id == bot.user.id and before.channel != after.channel:
                print(
                    f"Bot moved to a new channel: {after.channel.name if after.channel else 'None'}")
                if after.channel:
                    bot_ui.connected_users = [
                        member for member in after.channel.members if member.id != bot.user.id
                    ]
                    print(
                        f"Updated connected_users list for new channel: {[user.display_name for user in bot_ui.connected_users]}")
                else:
                    bot_ui.connected_users = []
                    print(
                        "Bot is no longer in a voice channel. Connected users cleared.")

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
                            f"User {member.display_name} added to connected_users list.")
                        print(
                            f"Updated connected_users list: {[user.display_name for user in bot_ui.connected_users]}")

                        # If listening is active, spawn a listener for the new user
                        if bot_ui.is_listening:
                            print(
                                f"Listening is active. Spawning listener for user: {member.display_name} (ID: {member.id})")
                            bot_ui.active_listeners[member.id] = asyncio.create_task(
                                utils.user_listener(
                                    bot_ui.vc, member.id, bot_ui)
                            )
                            print(
                                f"Started listener for user: {member.display_name} (ID: {member.id})")

                # Handle users leaving the channel
                if before.channel == current_channel and after.channel != current_channel:
                    if member in bot_ui.connected_users:
                        bot_ui.connected_users.remove(member)
                        print(
                            f"User {member.display_name} removed from connected_users list.")
                        print(
                            f"Updated connected_users list: {[user.display_name for user in bot_ui.connected_users]}")

                        # Stop the listener for the user if it exists
                        if member.id in bot_ui.active_listeners:
                            bot_ui.active_listeners[member.id].cancel()
                            del bot_ui.active_listeners[member.id]
                            print(
                                f"Stopped listener for user: {member.display_name} (ID: {member.id})")

        async def start_listening(vc, gui_instance):
            """Start recording and processing audio from the voice channel."""
            sink = WaveSink()

            async def process_audio_callback(sink, channel):
                """Process audio from the sink."""
                for user_id, audio in sink.audio_data.items():
                    # Save the audio to a temporary file
                    temp_audio_file = f"{user_id}_audio.wav"
                    with open(temp_audio_file, "wb") as f:
                        f.write(audio.file.read())

                    # Transcribe and translate the audio
                    transcribed_text = await utils.transcribe(temp_audio_file)
                    translated_text = await utils.translate(transcribed_text)

                    # Get the user's name or mention
                    user = sink.vc.guild.get_member(user_id)
                    if user:
                        user_name = user.display_name  # Use the display name
                    else:
                        # Fallback: Try fetching the member if not in cache
                        try:
                            user = await sink.vc.guild.fetch_member(user_id)
                            user_name = user.display_name
                        except discord.NotFound:
                            user_name = f"Unknown User ({user_id})"

                    # Update the GUI with the results, including the user's name
                    gui_instance.update_text_display(
                        f"{user_name}: {transcribed_text}",
                        f"{user_name}: {translated_text}"
                    )

                    # Clean up the temporary file
                    os.remove(temp_audio_file)

                print("Finished processing audio.")

            vc.start_recording(
                sink,
                process_audio_callback,  # Pass the coroutine directly
                None,
            )
            print("Started listening to the voice channel.")

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
intents.voice_states = True  # Ensure the bot can track voice state updates
intents.members = True  # Add this to allow fetching member information
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
