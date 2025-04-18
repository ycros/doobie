# Discord XTTS-v2 Streaming Bot (Proof of Concept)

This project is a proof-of-concept demonstrating how to integrate Coqui XTTS-v2 text-to-speech with streaming into a Discord voice channel via a bot.

## Features

*   Connects to Discord voice channels.
*   Streams synthesized speech directly using XTTS-v2.
*   Uses `/say` command for TTS generation with language selection.
*   Managed environment using `uv`.

## Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (Python package installer and virtual environment manager)
*   [FFmpeg](https://ffmpeg.org/download.html) (Required by `discord.py[voice]`)
*   Downloaded [XTTS-v2 model files](https://coqui.ai/blog/tts/xtts_v2_release) from Coqui.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Download the XTTS-v2 Model:**
    Download the XTTS-v2 model files and place them in a directory named `XTTS-v2` within the project's root directory. The structure should look like this:
    ```
    .
    ├── XTTS-v2/
    │   ├── config.json
    │   ├── model.pth
    │   ├── vocab.json
    │   └── samples/
    │       └── en_sample.wav  # Or your chosen reference audio
    ├── main.py
    ├── pyproject.toml
    └── ...
    ```
    *Note: Ensure the reference audio path in `main.py` (currently `XTTS-v2/samples/en_sample.wav`) matches your downloaded file.*

3.  **Create and activate virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # On Linux/macOS
    # .\.venv\Scripts\activate  # On Windows
    ```

4.  **Install dependencies:**
    ```bash
    uv sync
    ```
    *(This synchronizes the environment with the project's dependencies, typically using `uv.lock` if present, or `pyproject.toml`)*

5.  **Configure Discord Token:**
    Create a `.env` file in the project root. You can copy the sample file:
    ```bash
    cp .env.sample .env
    ```
    Then, edit the `.env` file and add your Discord bot token:
    ```env
    DISCORD_TOKEN=YOUR_BOT_TOKEN_HERE
    ```

## Running the Bot

Once the setup is complete, run the bot using `uv`:

```bash
uv run main.py
```

The bot should log in and be ready to accept commands.

## Usage

Use the following slash commands in your Discord server:

*   `/connect`: Connects the bot to the voice channel you are currently in.
*   `/disconnect`: Disconnects the bot from its current voice channel.
*   `/say <text> <language>`: Makes the bot speak the provided text in the specified language using XTTS-v2 streaming. Supported languages are available as choices in the command.

## License

This project is licensed under the [Creative Commons Zero v1.0 Universal](LICENSE) license.
