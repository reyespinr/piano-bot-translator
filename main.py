import argparse
import discord
import asyncio
import sound
import cli
import sys
import logging
import whisper  # Updated import statement
import requests
import pyaudio
import wave
import os
import platform

# error logging
error_formatter = logging.Formatter(
    fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

error_handler = logging.FileHandler("DAP_errors.log", delay=True)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(error_formatter)

base_logger = logging.getLogger()
base_logger.addHandler(error_handler)


# commandline args
parser = argparse.ArgumentParser(description="Discord Audio Pipe")
connect = parser.add_argument_group("Command Line Mode")
query = parser.add_argument_group("Queries")

parser.add_argument(
    "-t",
    "--token",
    dest="token",
    action="store",
    default=None,
    help="The token for the bot",
)

parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Enable verbose logging",
)

connect.add_argument(
    "-c",
    "--channel",
    dest="channel",
    action="store",
    type=int,
    help="The channel to connect to as an id",
)

connect.add_argument(
    "-d",
    "--device",
    dest="device",
    action="store",
    type=int,
    help="The device to listen from as an index",
)

query.add_argument(
    "-D",
    "--devices",
    dest="query",
    action="store_true",
    help="Query compatible audio devices",
)

query.add_argument(
    "-C",
    "--channels",
    dest="online",
    action="store_true",
    help="Query servers and channels (requires token)",
)

args = parser.parse_args()
is_gui = not any([args.channel, args.device, args.query, args.online])

# verbose logs
if args.verbose:
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

# don't import qt stuff if not using gui
if is_gui:
    import gui
    from PyQt5.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)

# Ensure opus library is loaded
if not discord.opus.is_loaded():
    if platform.system() == 'Windows':
        opus_path = os.path.join(os.path.dirname(__file__), 'libopus.dll')
        discord.opus.load_opus(opus_path)
    elif platform.system() == 'Linux':
        discord.opus.load_opus('libopus.so')
    elif platform.system() == 'Darwin':  # macOS
        discord.opus.load_opus('libopus.dylib')
    else:
        raise RuntimeError('Unsupported operating system')

# main


async def transcribe_and_translate(audio_file_path):
    print(f"Transcribing audio file: {audio_file_path}")  # Debugging statement
    # Transcribe speech to text using Whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    german_text = result["text"]

    # Output the transcribed text for debugging
    print(f"Transcribed Text: {german_text}")

    # Translate text from German to English using DeepL API
    response = requests.post(
        "https://api-free.deepl.com/v2/translate",
        data={
            "auth_key": "5ac935ea-9ed2-40a7-bd4d-8153c941c79f:fx",
            "text": german_text,
            "target_lang": "EN"
        },
        timeout=10
    )
    translation = response.json()["translations"][0]["text"]
    return translation


class AudioRecorder:
    def __init__(self):
        self.frames = []

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_frames(self):
        return b''.join(self.frames)


async def capture_audio(vc):
    print("Starting audio capture...")  # Debugging statement

    p = pyaudio.PyAudio()
    recorder = AudioRecorder()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, input=True,
                    frames_per_buffer=1024, stream_callback=recorder.callback)
    stream.start_stream()

    await asyncio.sleep(5)  # Capture 5 seconds of audio

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = recorder.get_frames()
    # Debugging statement
    print(f"Captured {len(audio_data)} bytes of audio data")

    wf = wave.open("output.wav", "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(48000)
    wf.writeframes(audio_data)
    wf.close()

    print("Audio capture complete, saved to output.wav")  # Debugging statement
    return "output.wav"


async def listen_and_translate(vc):
    while True:
        audio_file = await capture_audio(vc)
        translation = await transcribe_and_translate(audio_file)
        print(f"Translation: {translation}")


async def main(bot):
    try:
        # query devices
        if args.query:
            for device, index in sound.query_devices().items():
                print(index, device)

            return

        # check for token
        token = args.token
        if token is None:
            token = open("token.txt", "r").read()

        # query servers and channels
        if args.online:
            await cli.query(bot, token)

            return

        # GUI
        if is_gui:
            bot_ui = gui.GUI(app, bot)
            asyncio.ensure_future(bot_ui.ready())
            asyncio.ensure_future(bot_ui.run_Qt())

        # CLI
        else:
            asyncio.ensure_future(cli.connect(bot, args.device, args.channel))

        @bot.event
        async def on_ready():
            print(f'Logged in as {bot.user}')
            # Debugging statement
            print("Bot is ready and waiting for voice state updates...")

        @bot.event
        async def on_voice_state_update(member, before, after):
            print("Voice state update detected...")  # Debugging statement
            if after.channel is not None and after.channel != before.channel:
                # Debugging statement
                print(f"User {member} joined voice channel {after.channel}")
                if bot.voice_clients:
                    vc = bot.voice_clients[0]
                    if vc.channel != after.channel:
                        await vc.move_to(after.channel)
                        # Debugging statement
                        print(f"Moved to voice channel: {after.channel}")
                else:
                    vc = await after.channel.connect()
                    # Debugging statement
                    print(f"Connected to voice channel: {after.channel}")
                asyncio.ensure_future(listen_and_translate(vc))

        await bot.start(token)

    except FileNotFoundError:
        if is_gui:
            msg.setWindowTitle("Token Error")
            msg.setText("No Token Provided")
            msg.exec()

        else:
            print("No Token Provided")

    except discord.errors.LoginFailure:
        if is_gui:
            msg.setWindowTitle("Login Failed")
            msg.setText("Please check if the token is correct")
            msg.exec()

        else:
            print("Login Failed: Please check if the token is correct")

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
