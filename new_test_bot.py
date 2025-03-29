import discord
from os import environ as env
import os
from utils import transcribe  # Import the transcribe function from utils.py

bot = discord.Bot()
connections = {}

opus_path = os.path.join(os.path.dirname(__file__), 'libopus.dll')
discord.opus.load_opus(opus_path)


@bot.command()
async def record(ctx):
    voice = ctx.author.voice

    if not voice:
        await ctx.respond("⚠️ You aren't in a voice channel!")
        return

    vc = await voice.channel.connect()
    connections.update({ctx.guild.id: vc})

    vc.start_recording(
        discord.sinks.WaveSink(),
        once_done,
        ctx.channel,
    )
    await ctx.respond("🔴 Recording audio...")


async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):
    await sink.vc.disconnect()

    # Process the recorded audio for each user
    for user_id, audio in sink.audio_data.items():
        # Save the audio to a temporary file
        temp_audio_file = f"{user_id}_audio.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio.file.read())

        # Transcribe the audio using the transcribe function from utils.py
        transcription = await transcribe(temp_audio_file)

        # Send the transcription to the text channel with the user's mention
        await channel.send(
            f"🎤 <@{user_id}> said: \n\n{transcription}"
        )

        # Clean up the temporary audio file
        # os.remove(temp_audio_file)


@bot.command()
async def stop_recording(ctx):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.respond("🛑 Stopped recording.")
    else:
        await ctx.respond("🚫 Not recording in this server.")


bot.run("OTkwNDAyNjY2NDA2NDkwMTIy.GRj9Y7.j6f8oaSizpix2zcMLGzKE-o81ufL7ol5sI9a2c")
