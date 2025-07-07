# Piano Bot Translator

A real-time Discord voice channel translator that transcribes and translates speech using state-of-the-art AI models. The bot listens to voice channels, transcribes speech using OpenAI Whisper, detects languages, and translates non-English speech to English using DeepL's API.

## Features

- **Real-time Voice Processing**: Live transcription and translation of Discord voice channel audio
- **Multi-language Support**: Automatic language detection and translation to English
- **Dual Web Interface**: Modern Discord-themed admin interface + read-only spectator view
- **Individual User Management**: Toggle processing for specific users with real-time controls
- **Smart Speech Detection**: Advanced voice activity detection with adaptive thresholds
- **Multi-tier Model Support**: Configurable accurate/fast model tiers for optimal performance
- **WebSocket Real-time Communication**: Instant updates with auto-reconnect functionality
- **Modular Architecture**: Clean component separation with centralized configuration
- **Production Ready**: Comprehensive logging, error handling, and resource management

## Architecture

- **Backend**: FastAPI server with py-cord Discord bot integration
- **Frontend**: Real-time web dashboard with WebSocket communication
- **Audio Processing**: Modular Discord audio sink with real-time speech detection
- **ML Models**: OpenAI Whisper for transcription, DeepL API for translation
- **State Management**: Dataclass-based state management with centralized configuration
- **WebSocket System**: Dual-mode WebSocket support (admin/spectator) with message routing
- **Modular Design**: Component-based architecture with dedicated managers for different concerns

## Prerequisites

- Python 3.10 or higher
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
- `py-cord` + `PyNaCl` - Modern Discord voice channel integration
- `faster-whisper` - GPU-accelerated Whisper transcription with CUDA support
- `torch` + `torchaudio` - PyTorch ML framework with CUDA support
- `numpy<2.0` - Numerical computing (version locked for compatibility)
- `fastapi` + `uvicorn` - Modern async web server and WebSocket handling
- `websockets` - WebSocket support for real-time communication
- `PyYAML` - Configuration file management
- `requests` - HTTP client for DeepL API

The optimized requirements.txt includes specific versions and CUDA support for better performance and compatibility.

**Recommended installation:**
```bash
# For a clean installation, especially if you've had Whisper issues:
pip uninstall openai-whisper -y
pip install -r requirements.txt
pip install -r requirements-faster-whisper.txt
```

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
2. Update the API key in `translation.py` (search for DeepL API configuration)
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

### Model Configuration
The system now uses a centralized `config.yaml` file for model configuration. Edit this file to adjust:

```yaml
models:
  # Accurate tier - high-quality models for accuracy
  accurate:
    name: "large-v3-turbo"  # or "large-v3", "base"
    count: 1
    device: "cuda"  # or "cpu"
    warmup_timeout: 30
    
  # Fast tier - small, fast models for speed and parallelization  
  fast:
    name: "base"  # or "tiny", "small"
    count: 3
    device: "cuda"
    warmup_timeout: 30
```

### Audio Processing Settings
Edit `audio_sink_core.py` and related audio processing modules to adjust:
- Speech detection thresholds
- Buffer sizes and timing
- Silence detection parameters

### Translation Settings
Edit `translation_utils.py` to modify:
- Language detection sensitivity
- Translation API endpoints
- DeepL API configuration

### Logging Configuration
Edit `config.yaml` to adjust logging levels:
```yaml
logging:
  console_level: 'INFO'  # or 'DEBUG'
  file_level: 'DEBUG'
  max_file_size: 10
  backup_count: 4
```

## API Endpoints

- **WebSocket (Admin)**: `ws://localhost:8000/ws` - Full control interface
- **WebSocket (Spectator)**: `ws://localhost:8000/ws/spectator` - Read-only monitoring
- **Main Dashboard**: `http://localhost:8000/` - Admin web interface
- **Spectator View**: `http://localhost:8000/spectator` - Read-only web interface
- **Static Files**: `http://localhost:8000/*` - Frontend assets

### Interface Modes

**Admin Interface** (`http://localhost:8000/`):
- Full bot control (join/leave channels, start/stop listening)
- Individual user toggle controls
- Real-time transcription and translation display
- Bot status monitoring and configuration

**Spectator Interface** (`http://localhost:8000/spectator`):
- Read-only view of live transcriptions and translations
- Real-time updates without control capabilities
- Perfect for sharing with team members or viewers
- Minimal resource usage for display-only purposes

## Troubleshooting

### Common Issues

**"cannot import name 'dtw_kernel' from 'whisper.triton_ops'"**
- This was a compatibility issue with the old stable-ts implementation
- Solution: The project now uses faster-whisper which doesn't have this issue:
```bash
pip uninstall openai-whisper -y
pip install -r requirements.txt
pip install -r requirements-faster-whisper.txt
```
- If issues persist, ensure CUDA is properly installed for GPU acceleration

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
- Verify DeepL API key in `translation_utils.py`
- Check internet connectivity
- Monitor API quota usage

**"WebSocket connection issues"**
- Check firewall settings
- Ensure port 8000 is available
- Verify browser WebSocket support

### Performance Optimization

**For faster transcription:**
- Use smaller Whisper models in `config.yaml` ("tiny", "base")
- Increase the `fast` tier model count for parallel processing
- Use GPU acceleration by setting `device: "cuda"`
- Reduce audio buffer sizes in audio processing modules

**For better accuracy:**
- Use larger models in `config.yaml` ("large-v3", "large-v3-turbo")
- Use the `accurate` tier for critical transcriptions
- Increase speech detection thresholds in audio processing
- Enable advanced audio preprocessing

## Development

### Code Structure
```
├── server.py                      # Main FastAPI server and Discord bot
├── config.yaml                    # Centralized configuration
├── requirements.txt               # Python dependencies with CUDA support
├── translation_service.py         # Voice translation manager
├── translation_engine.py          # Core translation components
├── translation_utils.py           # Translation and language utilities
├── audio_sink.py                  # Discord audio sink implementation
├── audio_sink_core.py             # Core audio processing logic
├── audio_processing_utils.py      # Audio processing utilities
├── audio_buffer_manager.py        # Audio buffer management
├── audio_session_manager.py       # Audio session handling
├── audio_worker_manager.py        # Audio worker management
├── bot_manager.py                 # Discord bot lifecycle management
├── faster_whisper_manager.py      # Faster-Whisper model management
├── faster_whisper_core.py         # Core faster-whisper components
├── faster_whisper_engine.py       # Faster-Whisper transcription engine
├── faster_whisper_service.py      # Faster-Whisper transcription service
├── websocket_handler.py           # WebSocket connection management
├── websocket_broadcaster.py       # WebSocket message broadcasting
├── websocket_connection_manager.py # WebSocket connection handling
├── websocket_message_router.py    # WebSocket message routing
├── websocket_state_manager.py     # WebSocket state management
├── discord_voice_events.py        # Discord voice event handling
├── discord_channel_manager.py     # Discord channel operations
├── discord_user_manager.py        # Discord user operations
├── transcription_engine.py        # Speech transcription logic
├── transcription_service.py       # Transcription service coordination
├── voice_activity_detector.py     # Voice activity detection
├── cleanup.py                     # Temporary file management
├── logging_config.py              # Logging configuration
├── config_manager.py              # Configuration management
├── code_stats.py                  # Code analysis and statistics
├── frontend/                      # Web dashboard
│   ├── index.html                 # Main admin interface
│   ├── spectator.html             # Read-only spectator interface
│   ├── app.js                     # Frontend logic with auto-reconnect
│   ├── styles.css                 # Discord-themed styling
│   └── favicon.ico                # Site icon
└── logs/                          # Application logs
```

### State Management
The application uses a modular, dataclass-based architecture with centralized state management:

**Core State Classes:**
- `VoiceTranslatorState` - Voice translation state and configuration
- `WebSocketStateManager` - WebSocket client states and user processing settings
- `ModelManager` - ML model loading, warm-up, and lifecycle management
- `AudioSinkState` - Audio processing and buffer management
- `TranscriptionRequest/Result` - Structured transcription handling

**Key State Components:**
- Bot connection status and voice channel information
- User processing states with individual toggle controls
- Real-time transcription and translation data
- WebSocket connection management (admin/spectator modes)
- Model loading status and warm-up progress
- Audio processing buffers and voice activity detection
- Session-based audio management and cleanup

**Modular Architecture:**
- Separation of concerns with dedicated managers for Discord, WebSocket, and ML operations
- Clean interfaces between components for easier testing and maintenance
- Centralized configuration through `config.yaml`
- Proper resource cleanup and error handling
- Component-based design with focused responsibilities

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
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for GPU-accelerated Whisper processing
- [DeepL](https://www.deepl.com/api) for translation services
- [py-cord](https://github.com/Pycord-Development/pycord) for modern Discord integration
- [FastAPI](https://fastapi.tiangolo.com/) for web framework
