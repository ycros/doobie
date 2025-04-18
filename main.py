import discord
from discord import (
    app_commands,
    FFmpegOpusAudio,
    VoiceChannel,
    Member,
    VoiceClient,
)
from discord.errors import ClientException
import os
import io
import asyncio
import threading
import numpy as np  # Re-added as it's needed by TTSStreamReader
from dotenv import load_dotenv
from typing import Optional
import time
import torch
import torchaudio
import torch.serialization  # Added for safe loading

# TTS Imports
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs


load_dotenv()

# --- Constants ---
XTTS_SAMPLE_RATE = 24000

# --- TTS Setup ---
print("Loading XTTS model...")
# Determine the best device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")  # Added log to confirm device
xtts_config = XttsConfig()
# Assuming XTTS-v2 model is downloaded in the project root
MODEL_DIR = "XTTS-v2/"
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
CHECKPOINT_DIR = MODEL_DIR
REFERENCE_AUDIO_PATH = ["XTTS-v2/samples/en_sample.wav"]

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"XTTS config not found at {CONFIG_PATH}. Make sure the XTTS-v2 model is downloaded correctly."
    )
if not os.path.exists(CHECKPOINT_DIR):
    raise FileNotFoundError(f"XTTS checkpoint directory not found at {CHECKPOINT_DIR}.")
# Check if reference audio exists
if not all(os.path.exists(p) for p in REFERENCE_AUDIO_PATH):
    raise FileNotFoundError(
        f"Reference audio not found at {REFERENCE_AUDIO_PATH}. Please provide a valid path."
    )


xtts_config.load_json(CONFIG_PATH)
xtts_model = Xtts.init_from_config(xtts_config)
# Allow loading the config class within the checkpoint
torch.serialization.add_safe_globals(
    [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
)
xtts_model.load_checkpoint(
    xtts_config, checkpoint_dir=CHECKPOINT_DIR, use_deepspeed=False
)
xtts_model.to(device)
print("XTTS model loaded.")

print("Computing speaker latents...")
# This assumes REFERENCE_AUDIO_PATH points to valid audio file(s)
try:
    gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
        audio_path=REFERENCE_AUDIO_PATH
    )
    print("Speaker latents computed.")
except Exception as e:
    print(f"Error computing speaker latents: {repr(e)}")
    # Decide how to handle this - maybe exit or disable TTS commands
    gpt_cond_latent, speaker_embedding = None, None


# --- Bot Setup ---
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


async def _get_voice_client(interaction: discord.Interaction) -> Optional[VoiceClient]:
    if not interaction.guild:
        await interaction.followup.send(
            "This command requires a server.", ephemeral=True
        )
        return None

    voice_client = interaction.guild.voice_client
    if isinstance(voice_client, VoiceClient) and voice_client.is_connected():
        return voice_client
    else:
        await interaction.followup.send(
            "I'm not connected to a voice channel.", ephemeral=True
        )
        return None


# --- Custom IO Stream for TTS ---


class TTSStreamReader(io.RawIOBase):
    def __init__(self, text: str, language: str):
        super().__init__()
        self.text = text
        self.language = language
        self._generator = None
        self._buffer = b""
        self._eof = False
        self._lock = threading.Lock()
        print(f"TTSStreamReader initialized for '{text[:20]}...'")

    def _initialize_generator(self):
        if self._generator is None and not self._eof:
            try:
                self._generator = xtts_model.inference_stream(
                    self.text,
                    self.language,
                    gpt_cond_latent,
                    speaker_embedding,
                )
            except Exception as e:
                print(f"TTSStreamReader: Error initializing XTTS generator: {repr(e)}")
                self._eof = True  # Mark as EOF if init fails
        else:
            # This path should ideally not be taken if called correctly
            pass

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        with self._lock:
            if self._eof:
                return 0  # Signal EOF

            # Initialize generator on first read
            if self._generator is None:
                self._initialize_generator()
                if self._eof:
                    return 0

            # Fill the buffer 'b'
            bytes_read = 0
            while bytes_read < len(b):
                # If we have data in our internal buffer, use it first
                if self._buffer:
                    chunk_len = min(len(self._buffer), len(b) - bytes_read)
                    b[bytes_read : bytes_read + chunk_len] = self._buffer[:chunk_len]
                    self._buffer = self._buffer[chunk_len:]
                    bytes_read += chunk_len
                    continue

                # Get next chunk from TTS generator
                try:
                    if self._generator:
                        chunk = next(self._generator)
                        pcm_data = (
                            (chunk.squeeze().cpu().detach() * 32767)
                            .numpy()
                            .astype(np.int16)
                            .tobytes()
                        )
                        self._buffer += pcm_data
                    else:
                        print("TTSStreamReader: Generator is None during read.")
                        self._eof = True
                        break

                except StopIteration:
                    print("TTSStreamReader: Generator finished.")
                    self._eof = True
                    break
                except Exception as e:
                    print(f"TTSStreamReader: Error reading from generator: {repr(e)}")
                    self._eof = True
                    break

            return bytes_read

    def close(self) -> None:
        print("TTSStreamReader: Closing stream.")
        with self._lock:
            self._eof = True
            self._generator = None  # Allow garbage collection
            self._buffer = b""
        super().close()


# --- Events ---


@bot.event
async def on_ready():
    user_info = f"{bot.user} (ID: {bot.user.id})" if bot.user else "Unknown User"
    print(f"Logged in as {user_info}")
    print("Syncing command tree...")
    try:
        await tree.sync()
        print("Command tree synced.")
    except Exception as e:
        print(f"Failed to sync command tree: {repr(e)}")
    print("------")


# --- Slash Commands ---


@tree.command(name="ping", description="Replies with Pong!")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! (slash command)", ephemeral=True)


@tree.command(name="connect", description="Connects the bot to your voice channel.")
async def connect(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    if not interaction.guild:
        await interaction.followup.send(
            "This command requires a server.", ephemeral=True
        )
        return

    user = interaction.user
    if not isinstance(user, Member) or not user.voice or not user.voice.channel:
        await interaction.followup.send(
            "You must be in a voice channel.", ephemeral=True
        )
        return

    channel = user.voice.channel
    if not isinstance(channel, VoiceChannel):
        await interaction.followup.send(
            "Can only connect to voice channels.", ephemeral=True
        )
        return

    voice_client = interaction.guild.voice_client

    try:
        if isinstance(voice_client, VoiceClient):
            if voice_client.channel == channel:
                await interaction.followup.send(
                    f"Already in {channel.name}.", ephemeral=True
                )
            else:
                await voice_client.move_to(channel)
                await interaction.followup.send(
                    f"Moved to {channel.name}.", ephemeral=True
                )
        else:
            await channel.connect(timeout=60.0, reconnect=True)
            await interaction.followup.send(
                f"Connected to {channel.name}.", ephemeral=True
            )
    except asyncio.TimeoutError:
        await interaction.followup.send("Connection timed out.", ephemeral=True)
    except ClientException as e:
        await interaction.followup.send(f"Connection error: {e}", ephemeral=True)
    except Exception as e:
        print(f"Connect command error: {repr(e)}")
        await interaction.followup.send(
            f"Failed to connect/move: {repr(e)}", ephemeral=True
        )


@tree.command(
    name="disconnect", description="Disconnects the bot from the voice channel."
)
async def disconnect(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    voice_client = await _get_voice_client(interaction)
    if not voice_client:
        return  # Error message handled by helper

    try:
        await voice_client.disconnect(force=False)
        await interaction.followup.send("Disconnected.", ephemeral=True)
    except Exception as e:
        print(f"Disconnect error: {repr(e)}")
        await interaction.followup.send(
            f"Failed to disconnect: {repr(e)}", ephemeral=True
        )


@tree.command(name="say", description="Synthesizes speech using XTTS streaming.")
@app_commands.describe(
    text="The text to speak.",
    language="The language of the text (e.g., 'en', 'es', 'fr').",
)
@app_commands.choices(  # Keep choices for usability
    language=[
        app_commands.Choice(name="English", value="en"),
        app_commands.Choice(name="Spanish", value="es"),
        app_commands.Choice(name="French", value="fr"),
        app_commands.Choice(name="German", value="de"),
        app_commands.Choice(name="Italian", value="it"),
        app_commands.Choice(name="Portuguese", value="pt"),
        app_commands.Choice(name="Polish", value="pl"),
        app_commands.Choice(name="Turkish", value="tr"),
        app_commands.Choice(name="Russian", value="ru"),
        app_commands.Choice(name="Dutch", value="nl"),
        app_commands.Choice(name="Czech", value="cs"),
        app_commands.Choice(name="Arabic", value="ar"),
        app_commands.Choice(name="Chinese (Simplified)", value="zh-cn"),
        app_commands.Choice(name="Japanese", value="ja"),
        app_commands.Choice(name="Hungarian", value="hu"),
        app_commands.Choice(name="Korean", value="ko"),
    ]
)
async def say(interaction: discord.Interaction, text: str, language: str):
    await interaction.response.defer(ephemeral=True)

    voice_client = await _get_voice_client(interaction)
    if not voice_client:
        return  # Error message handled by helper

    if gpt_cond_latent is None or speaker_embedding is None:
        await interaction.followup.send(
            "Speaker latents not available. Cannot synthesize.", ephemeral=True
        )
        return

    if voice_client.is_playing():
        await interaction.followup.send("Already playing audio.", ephemeral=True)
        return

    tts_reader: Optional[TTSStreamReader] = None
    ffmpeg_source: Optional[FFmpegOpusAudio] = None

    try:
        tts_reader = TTSStreamReader(text, language)

        ffmpeg_source = FFmpegOpusAudio(
            tts_reader,  # type: ignore # Pass our RawIOBase reader here (duck typing works)
            pipe=True,
            before_options=f"-f s16le -ac 1 -ar {XTTS_SAMPLE_RATE}",
            options="-loglevel warning -vn",
        )

        def after_play(error):
            if error:
                print(f"Error during TTS playback: {error}")
            print("Playback finished or stopped. Closing TTS reader.")
            if tts_reader:
                tts_reader.close()

        voice_client.play(ffmpeg_source, after=after_play)
        print(
            "Started voice_client.play() with FFmpegOpusAudio piping TTSStreamReader."  # Keep essential log
        )
        await interaction.followup.send(
            f"Streaming '{text[:50]}...' in {language}.", ephemeral=True
        )
        # Playback runs in the background

    except Exception as e:
        print(f"Error setting up TTS stream: {repr(e)}")  # Keep error logging
        try:
            await interaction.edit_original_response(
                content=f"Failed to start streaming: {repr(e)}"
            )
        except discord.NotFound:
            await interaction.followup.send(
                f"Failed to start streaming: {repr(e)}", ephemeral=True
            )
        # --- Cleanup on setup error ---
        print("Cleaning up resources due to setup error.")
        if voice_client and voice_client.is_playing():
            print("Stopping playback.")
            voice_client.stop()  # This should trigger after_play for tts_reader cleanup
        else:
            # If play never started, cleanup manually
            if tts_reader:
                tts_reader.close()
            # ffmpeg_source cleanup is handled by its __del__


# --- Main Execution ---

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable not set or empty.")
    try:
        bot.run(token)
    except Exception as e:
        print(f"Error running bot: {repr(e)}")
