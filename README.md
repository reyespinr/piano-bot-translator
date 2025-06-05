# Piano Bot Translator

A real-time Discord voice channel translator that transcribes and translates speech using state-of-the-art AI models. The bot listens to voice channels, transcribes speech using OpenAI Whisper, detects languages, and translates non-English speech to English using DeepL's API.

## Features

- **Real-time Voice Processing**: Live transcription and translation of Discord voice channel audio
- **Multi-language Support**: Automatic language detection and translation to English
- **Web Dashboard**: Modern Discord-themed web interface for control and monitoring
- **User Management**: Individual user processing toggles for selective filtering
- **Smart Speech Detection**: Context-aware speech processing with adaptive thresholds
- **Session-based Priority**: Improved conversation context through priority filtering
- **Modern Architecture**: Clean FastAPI backend with WebSocket real-time communication

## Architecture

- **Backend**: FastAPI server with Discord.py bot integration
- **Frontend**: Real-time web dashboard with WebSocket communication
- **Audio Processing**: Custom Discord audio sink with real-time speech detection
- **ML Models**: OpenAI Whisper for transcription, DeepL API for translation
- **State Management**: Dataclass-based state management with proper cleanup

## Prerequisites

- Python 3.8 or higher
- Discord bot token (see setup instructions below)
- DeepL API key (free tier available)
- Windows/Linux/macOS with audio support

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/reyespinr/piano-bot-translator.git
cd piano-bot-translator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `discord.py[voice]` + `PyNaCl` - Discord voice channel integration
- `stable-ts` - Enhanced Whisper transcription (auto-installs: openai-whisper, torch, torchaudio, numpy, numba)
- `fastapi` + `uvicorn` - Web server and WebSocket handling  
- `requests` - HTTP client for DeepL API

The optimized requirements.txt only includes essential packages, reducing installation time and potential conflicts.

### 3. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg  # Required for Whisper audio processing
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- FFmpeg is required and should be available in PATH
- Download from [FFmpeg.org](https://ffmpeg.org/download.html) or install via [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`

### 4. Setup Bot Token
1. Create a Discord application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a bot user and copy the bot token
3. Create a file named `token.txt` in the project root
4. Paste your bot token into `token.txt`

### 5. Setup DeepL API (Optional but Recommended)
1. Sign up for a free DeepL API account
2. Update the API key in `utils.py` (line 162)
3. The free tier provides 500,000 characters/month

### 6. Invite Bot to Server
Generate an invite link with these permissions:
- Connect (voice channels)
- Speak (voice channels)
- View Channels
- Read Message History

**Permissions Integer**: `3148800`

**Invite URL Template**:
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_BOT_CLIENT_ID&permissions=3148800&scope=bot
```

## Running the Application

### Development Mode
```bash
python server.py
```

### Production Mode (Ubuntu Server)
```bash
# Install screen for background execution
sudo apt-get install screen

# Start in background session
screen -S piano-bot
python server.py

# Detach with Ctrl+A, D
# Reattach with: screen -r piano-bot
```

### Using systemd (Ubuntu Server)
Create a systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/piano-bot.service
```

Service file content:
```ini
[Unit]
Description=Piano Bot Translator
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/piano-bot-translator
ExecStart=/usr/bin/python3 server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable piano-bot.service
sudo systemctl start piano-bot.service

# Check status
sudo systemctl status piano-bot.service

# View logs
journalctl -u piano-bot.service -f
```

## Usage

1. **Start the Server**: Run `python server.py`
2. **Access Dashboard**: Open `http://localhost:8000` in your browser
3. **Select Server**: Choose your Discord server from the dropdown
4. **Select Channel**: Choose a voice channel to monitor
5. **Join Channel**: Click "Join" to connect the bot
6. **Start Listening**: Click "Listen" to begin transcription/translation
7. **Manage Users**: Toggle individual users on/off as needed

## Configuration

### Audio Processing Settings
Edit `custom_sink.py` to adjust:
- Speech detection thresholds
- Buffer sizes and timing
- Silence detection parameters

### Translation Settings
Edit `utils.py` to modify:
- Language detection sensitivity
- Translation API endpoints
- Transcription model settings

### Model Configuration
By default, the system uses the Whisper "base" model for faster processing. For better accuracy, change the model in `utils.py`:
```python
MODEL_NAME = "large-v3"  # More accurate but slower
```

## API Endpoints

- **WebSocket**: `ws://localhost:8000/ws` - Real-time communication
- **Static Files**: `http://localhost:8000/` - Web dashboard
- **Health Check**: Bot status available through WebSocket

## Troubleshooting

### Common Issues

**"No module named 'discord'"**
```bash
pip install discord.py[voice]
```

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your system PATH
- Windows: `choco install ffmpeg` or download from FFmpeg.org
- Ubuntu: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

**"Bot doesn't respond to voice"**
- Check bot permissions in Discord server
- Ensure proper voice channel permissions
- Verify microphone detection thresholds

**"Translation not working"**
- Verify DeepL API key in `utils.py`
- Check internet connectivity
- Monitor API quota usage

**"WebSocket connection issues"**
- Check firewall settings
- Ensure port 8000 is available
- Verify browser WebSocket support

### Performance Optimization

**For faster transcription:**
- Use smaller Whisper models ("tiny", "base")
- Reduce audio buffer sizes
- Use GPU acceleration if available

**For better accuracy:**
- Use larger models ("large-v3")
- Increase speech detection thresholds
- Enable advanced audio preprocessing

## Development

### Code Structure
```
├── server.py              # Main FastAPI server and Discord bot
├── translator.py          # Voice translation logic
├── custom_sink.py         # Discord audio processing
├── utils.py              # Transcription and translation utilities
├── cleanup.py            # Temporary file management
├── logging_config.py     # Logging configuration
├── frontend/             # Web dashboard
│   ├── index.html        # Main interface
│   ├── app.js           # Frontend logic
│   └── styles.css       # Discord-themed styling
└── requirements.txt      # Python dependencies
```

### State Management
The application uses a dataclass-based state management system (`BotServerState`) that centralizes:
- Bot connection status
- Voice channel information
- User processing states
- Translation history
- WebSocket connections

### Contributing
1. Follow existing code style and patterns
2. Add appropriate logging for debugging
3. Update docstrings for new functions
4. Test voice channel functionality
5. Ensure proper cleanup of resources

## License

This project is open source. Please respect Discord's Terms of Service and API guidelines when using this bot.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [stable-ts](https://github.com/jianfch/stable-ts) for enhanced Whisper processing
- [DeepL](https://www.deepl.com/api) for translation services
- [Discord.py](https://discordpy.readthedocs.io/) for Discord integration
- [FastAPI](https://fastapi.tiangolo.com/) for web framework
