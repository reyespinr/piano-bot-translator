"""
Discord Piano Bot Translator - GUI Module

This module provides a graphical user interface for a Discord bot that can
join voice channels, record audio, transcribe speech, and translate it.
It handles audio streaming, voice channel connections, and user interaction
through PyQt5 controls.

The GUI allows users to:
- Connect to Discord voice channels
- Stream audio to these channels
- Record and process audio from users in the channel
- Display transcribed and translated text
- Toggle processing for specific users

Dependencies:
- PyQt5 for the user interface
- discord.py for Discord integration
- sound module for audio processing
"""

import os
import sys
import asyncio
import logging
import discord
# pylint: disable=no-name-in-module
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QFontDatabase, QFontMetrics, QIcon
from PyQt5.QtCore import Qt, QCoreApplication, QEventLoop, QDir, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QWidget,
    QFrame,
    QGridLayout,
    QComboBox,
    QLabel,
    QHBoxLayout,
    QStyledItemDelegate,
    QListView,
    QTextEdit,
    QMessageBox,
    QVBoxLayout
)
# pylint: enable=no-name-in-module
import sound
from custom_sink import RealTimeWaveSink

if getattr(sys, "frozen", False):
    bundle_dir = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))


class Dropdown(QComboBox):
    """
    Enhanced QComboBox that emits signals with both deselected and selected indices.

    This class extends QComboBox to track selection changes and provide
    additional functionality like hiding specific rows.

    Signals:
        changed(object, object): Emitted when selection changes, providing both
                                 the previously selected and newly selected indices.
    """
    changed = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setItemDelegate(QStyledItemDelegate())
        self.setPlaceholderText("None")
        self.setView(QListView())
        self.deselected = None
        self.currentIndexChanged.connect(self.changed_signal)

    def changed_signal(self, selected):
        """
        Signal handler for index changes in the dropdown.

        Args:
            selected (int): The index of the newly selected item
        """
        self.changed.emit(self.deselected, selected)
        self.deselected = selected

    def set_row_hidden(self, idx, hidden):
        """ Hide or show a specific row in the dropdown."""
        self.view().setRowHidden(idx, hidden)


class SVGButton(QPushButton):
    """
    Button with integrated SVG loading indicator.

    This button displays an SVG animation when disabled, providing
    visual feedback for background operations.
    """

    def __init__(self, text=None):
        super().__init__(text)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.svg = QSvgWidget("./assets/loading.svg", self)
        self.svg.setVisible(False)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.svg)

    # pylint: disable=invalid-name
    def setEnabled(self, enabled):
        """Override setEnabled to control SVG visibility based on enabled state.

        This method overrides Qt's setEnabled method and must use camelCase.
        """
        super().setEnabled(enabled)
        self.svg.setVisible(not enabled)

    def set_svg_visible(self, visible):
        """Set the visibility of the SVG widget directly."""
        self.svg.setVisible(visible)


# pylint: disable=too-many-instance-attributes
class Connection:
    """
    Manages a connection to a Discord voice channel.

    This class handles:
    - Device selection
    - Server (guild) selection
    - Channel selection
    - Audio streaming
    - Voice channel connection/disconnection
    - Mute/unmute functionality
    - Recording and processing audio

    It provides the UI elements needed to control these features.
    """

    def __init__(self, layer, parent):
        self.stream = sound.PCMStream()
        self.parent = parent
        self.voice = None
        self.sink = None

        # dropdowns
        self.devices = Dropdown()
        self.servers = Dropdown()
        self.channels = Dropdown()

        for device, idx in parent.devices.items():
            self.devices.addItem(device + "   ", idx)

        # buttons
        self.mute = SVGButton("Mute")
        self.mute.setObjectName("mute")
        self.listen = SVGButton("Listen")
        self.listen.setObjectName("listen")
        self.clear = QPushButton("Clear")
        self.clear.setObjectName("clear")

        # add widgets
        parent.layout.addWidget(self.devices, layer, 0)
        parent.layout.addWidget(self.servers, layer, 1)
        parent.layout.addWidget(self.channels, layer, 2)
        parent.layout.addWidget(self.mute, layer, 3)
        # Fixed: Added to layout
        parent.layout.addWidget(self.listen, layer+1, 2)
        # Fixed: Added to layout
        parent.layout.addWidget(self.clear, layer+1, 3)

        # events
        self.devices.changed.connect(self.change_device)
        self.servers.changed.connect(
            lambda deselected, selected: asyncio.ensure_future(
                self.change_server(deselected, selected)
            )
        )
        self.channels.changed.connect(
            lambda: asyncio.ensure_future(self.change_channel())
        )
        self.mute.clicked.connect(self.toggle_mute)
        self.listen.clicked.connect(self.toggle_listen)
        self.clear.clicked.connect(self.clear_text_boxes)

    @staticmethod
    def resize_combobox(combobox):
        """
        Resize a combobox to fit its contents.

        Calculates the width needed to display all items in the combobox
        and sets the minimum width accordingly.

        Args:
            combobox (QComboBox): The combobox to resize.
        """
        font = combobox.property("font")
        metrics = QFontMetrics(font)
        min_width = 0

        for i in range(combobox.count()):
            size = metrics.horizontalAdvance(combobox.itemText(i))
            min_width = max(min_width, size)

        combobox.setMinimumWidth(min_width + 50)

    def set_enabled(self, enabled):
        """Enable or disable all connection UI elements."""
        self.devices.setEnabled(enabled)
        self.servers.setEnabled(enabled)
        self.channels.setEnabled(enabled)
        self.mute.setEnabled(enabled)
        self.mute.setText("Mute" if enabled else "")

    def set_servers(self, guilds):
        """
        Populate the servers dropdown with Discord guilds (servers).

        Args:
            guilds (list): List of Discord Guild objects to add to the dropdown.
        """
        for guild in guilds:
            self.servers.addItem(guild.name, guild)

    def change_device(self):
        """
        Handle audio device changes.

        Updates the audio stream to use the newly selected audio device
        and restarts audio playback if it was already active.
        """
        try:
            selection = self.devices.currentData()
            self.mute.setText("Mute")

            if self.voice is not None:
                self.voice.stop()
                self.stream.change_device(selection)

                if self.voice.is_connected():
                    self.voice.play(self.stream)
            else:
                self.stream.change_device(selection)

        except (discord.errors.ClientException, AttributeError, ValueError, RuntimeError) as e:
            logging.exception("Error on change_device: %s", e)

    async def change_server(self, deselcted, selected):
        """
        Handle server selection changes.

        Updates the channels dropdown to show voice channels available
        in the selected server.

        Args:
            deselcted (int): Index of the previously selected server in the dropdown.
            selected (int): Index of the newly selected server in the dropdown.
        """
        try:
            selection = self.servers.itemData(selected)

            self.parent.exclude(deselcted, selected)
            self.channels.clear()
            self.channels.addItem("None", None)

            for channel in selection.channels:
                if isinstance(channel, discord.VoiceChannel):
                    self.channels.addItem(channel.name, channel)

            Connection.resize_combobox(self.channels)

        except (discord.errors.HTTPException, AttributeError, ValueError) as e:
            logging.exception("Error on change_server: %s", e)

    async def change_channel(self):
        """
        Handle voice channel selection changes.

        Connects to the selected voice channel or disconnects if "None" is selected.
        Updates the UI to reflect the connection status and populates the list of
        users in the channel.
        """
        try:
            selection = self.channels.currentData()
            self.mute.setText("Mute")
            self.set_enabled(False)

            if selection is not None:
                not_connected = (
                    self.voice is None
                    or self.voice is not None
                    and not self.voice.is_connected()
                )

                if not_connected:
                    self.voice = await selection.connect(timeout=10)
                else:
                    await self.voice.move_to(selection)

                self.parent.vc = self.voice  # Assign the voice client to the GUI instance
                print(f"Connected to voice channel: {self.voice.channel.name}")

                # Populate connected users when joining a channel
                self.parent.connected_users = [
                    member for member in selection.members if member.id != self.parent.bot.user.id
                ]
                # Update the UI with initial user list
                self.parent.update_connected_users(self.parent.connected_users)
                print(
                    f"Initial users: {[user.display_name for user in self.parent.connected_users]}")

                # Automatically initialize the audio stream with the current device
                self.change_device()

            else:
                if self.voice is not None:
                    await self.voice.disconnect()
                    self.parent.vc = None
                    print("Disconnected from the voice channel.")

        except (discord.errors.ClientException, discord.errors.HTTPException,
                asyncio.TimeoutError, AttributeError, RuntimeError) as e:
            logging.exception("Error on change_channel: %s", e)

        finally:
            self.set_enabled(True)

    async def handle_bot_movement(self, new_channel):
        """
        Handle events when the bot is moved between voice channels.

        Updates the connected users list and voice client reference
        when the bot is moved to a new channel or disconnected.

        Args:
            new_channel (discord.VoiceChannel or None): The channel the bot was moved to,
                                                       or None if the bot was disconnected.
        """
        try:
            if new_channel is not None:
                self.parent.vc = self.voice  # Update the voice client
                self.parent.connected_users = [
                    member for member in new_channel.members if member.id != self.parent.bot.user.id
                ]
                print(f"Bot moved to a new channel: {new_channel.name}")
                print(
                    "Updated connected_users list for new channel:"
                    f"{[user.display_name for user in self.parent.connected_users]}")
            else:
                self.parent.connected_users = []
                print("Bot is no longer in a voice channel. Connected users cleared.")
        except (discord.errors.HTTPException, AttributeError) as e:
            logging.exception("Error handling bot movement: %s", e)

    def toggle_mute(self):
        """
        Toggle audio muting for the voice client.

        Pauses or resumes audio playback and updates the mute button text accordingly.
        """
        try:
            if self.voice is not None:
                if self.voice.is_playing():
                    self.voice.pause()
                    self.mute.setText("Resume")
                else:
                    self.voice.resume()
                    self.mute.setText("Mute")
        except discord.errors.ClientException as e:
            logging.exception("Discord client error in toggle_mute: %s", e)
        except AttributeError as e:
            logging.exception("Attribute error in toggle_mute: %s", e)
        except RuntimeError as e:
            logging.exception("Runtime error in toggle_mute: %s", e)

    def toggle_listen(self):
        """ Toggle listening for user input."""
        try:
            if self.parent.is_listening:
                # Stop listening
                self.parent.is_listening = False
                print("Listening toggled OFF.")
                if self.parent.vc:
                    self.parent.vc.stop_recording()
                print("All listeners have been stopped.")

                # Update button text and style
                self.listen.setText("Listen")
                self.listen.setStyleSheet("")  # Reset to default style
            else:
                # Start listening
                self.parent.is_listening = True
                print("Listening toggled ON.")

                # Update button text and style
                self.listen.setText("Stop")
                self.listen.setStyleSheet(
                    "background-color: red; color: white;")

                # Start recording using RealTimeWaveSink
                if self.parent.vc:
                    try:
                        # Create the sink with a reference to the parent GUI
                        sink = RealTimeWaveSink(
                            pause_threshold=1.0,
                            event_loop=asyncio.get_event_loop()
                        )
                        # Add a reference to the parent GUI
                        sink.parent = self.parent

                        # Store the sink instance for emergency access
                        self.sink = sink
                        self.parent.vc.start_recording(
                            sink,
                            self.parent.process_audio_callback,
                            None,
                        )
                        print("Started recording audio.")
                    except (discord.errors.ClientException, RuntimeError) as e:
                        print(f"Error starting recording: {e}")
                        # Reset state
                        self.parent.is_listening = False
                        self.listen.setText("Listen")
                        self.listen.setStyleSheet("")
        except (discord.errors.ClientException, RuntimeError, AttributeError) as e:
            print(f"Error in toggle_listen: {e}")

    def clear_text_boxes(self):
        """Clear the transcribed and translated text boxes."""
        self.parent.transcribed_display.clear()
        self.parent.translated_display.clear()


class TitleBar(QFrame):
    """
    Custom window title bar implementation.

    Provides a custom window title bar with minimize and close buttons,
    replacing the default window decorations.
    """

    def __init__(self, parent):
        super().__init__()
        self.setObjectName("titlebar")
        self.parent = parent
        self.bot = parent.bot

        # layout and window title
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)
        self.setLayout(layout)
        title = QLabel("Piano Translator Bot")

        # buttons and add widgets
        minimize_button = QPushButton("—")
        minimize_button.setObjectName("minimize")
        close_button = QPushButton("✕")
        close_button.setObjectName("close")
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(minimize_button)
        layout.addWidget(close_button)

        # events
        minimize_button.clicked.connect(self.minimize)
        close_button.clicked.connect(
            lambda: asyncio.ensure_future(self.close()))

    async def close(self):
        """Handle the close event for the window."""
        # First, check if listening is active and stop it
        if self.parent.is_listening:
            try:
                print("Stopping recording before closing...")
                self.parent.force_stop_listening()
                await asyncio.sleep(0.5)
            except (discord.errors.ClientException, RuntimeError, asyncio.CancelledError) as e:
                print(f"Error stopping recording during close: {e}")

        # Then proceed with the normal closing routine
        for voice in self.bot.voice_clients:
            try:
                await voice.disconnect()
            except (discord.errors.ClientException, asyncio.CancelledError):
                pass

        # Use the proper method to close the bot instead of accessing protected members
        await self.bot.close()
        self.parent.close()

    def minimize(self):
        """Minimize the window."""
        self.parent.showMinimized()


# pylint: disable=too-many-instance-attributes
class GUI(QMainWindow):
    """
    Main application window for the Discord Piano Bot Translator.

    This class provides the complete user interface for the application, including:
    - Device, server, and channel selection
    - Text display areas for transcriptions and translations
    - User toggle controls
    - Emergency stop functionality
    - Window management

    It handles audio processing, voice channel interactions, and coordinates
    between different components of the application.
    """
    # pylint: disable=too-many-statements

    def __init__(self, app, bot):
        super().__init__()
        QDir.setCurrent(bundle_dir)
        self.app = app

        # window info
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        window_icon = QIcon("./assets/favicon.ico")
        self.setWindowTitle("Piano Translator Bot")
        self.app.setWindowIcon(window_icon)
        self.position = None

        # discord
        self.bot = bot
        self.vc = None  # Initialize the voice client attribute

        # layout
        central = QWidget()
        self.layout = QGridLayout()
        central.setLayout(self.layout)

        # labels
        self.info = QLabel("Connecting...")
        device_lb = QLabel("Devices")
        device_lb.setObjectName("label")
        server_lb = QLabel("Servers     ")
        server_lb.setObjectName("label")
        channel_lb = QLabel("Channels  ")
        channel_lb.setObjectName("label")

        # connections
        self.devices = sound.query_devices()
        self.connections = [Connection(2, self)]
        self.connected_servers = set()

        # UI layout
        self.layout.addWidget(self.info, 0, 0, 1, 4)  # Info label at the top
        self.layout.addWidget(device_lb, 1, 0)  # Device label
        self.layout.addWidget(server_lb, 1, 1)  # Server label
        self.layout.addWidget(channel_lb, 1, 2)  # Channel label

        # Dropdowns and buttons
        self.layout.addWidget(
            self.connections[0].devices, 2, 0)  # Devices dropdown
        self.layout.addWidget(
            self.connections[0].servers, 2, 1)  # Servers dropdown
        # Channels dropdown
        self.layout.addWidget(self.connections[0].channels, 2, 2)
        self.layout.addWidget(self.connections[0].mute, 2, 3)  # Mute button
        self.layout.addWidget(
            self.connections[0].listen, 3, 2)  # Listen button
        self.layout.addWidget(self.connections[0].clear, 3, 3)  # Clear button

        # Text display areas
        self.transcribed_display = QTextEdit()
        self.transcribed_display.setReadOnly(True)
        self.transcribed_display.setPlaceholderText("Transcribed Text")
        self.transcribed_display.setMinimumHeight(400)
        self.layout.addWidget(self.transcribed_display, 5,
                              0, 1, 4)  # Spanning all columns

        self.translated_display = QTextEdit()
        self.translated_display.setReadOnly(True)
        self.translated_display.setPlaceholderText("Translated Text")
        self.translated_display.setMinimumHeight(400)
        self.layout.addWidget(self.translated_display, 6,
                              0, 1, 4)  # Spanning all columns

        # Emergency stop button
        self.emergency_stop = QPushButton("EMERGENCY STOP")
        self.emergency_stop.setObjectName("emergency")
        self.emergency_stop.setStyleSheet(
            "background-color: darkred; color: white; font-weight: bold;")
        self.emergency_stop.clicked.connect(self.force_stop_listening)
        self.layout.addWidget(self.emergency_stop, 3, 0)

        # Add user tracking with processing enabled/disabled state
        self.user_processing_enabled = {}  # Maps user IDs to boolean enabled state

        # Add user toggle panel - place after the emergency stop button
        self.user_toggle_label = QLabel("Active Users:")
        self.user_toggle_label.setObjectName("label")
        # Position next to emergency stop
        self.layout.addWidget(self.user_toggle_label, 3, 1)

        # Scrollable area for user toggles
        self.user_toggle_area = QFrame()
        self.user_toggle_layout = QVBoxLayout(self.user_toggle_area)
        self.user_toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.user_toggle_area.setLayout(self.user_toggle_layout)
        # Position above transcription area
        self.layout.addWidget(self.user_toggle_area, 4, 0, 1, 2)

        # Adjust transcription and translation display positions
        # Move them down by 1 row
        self.layout.removeWidget(self.transcribed_display)
        self.layout.removeWidget(self.translated_display)
        self.layout.addWidget(self.transcribed_display, 5, 0, 1, 4)
        self.layout.addWidget(self.translated_display, 6, 0, 1, 4)

        # Add frame style and background color to user toggle area
        self.user_toggle_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.user_toggle_area.setMinimumHeight(100)  # Ensure minimum height
        self.user_toggle_area.setStyleSheet(
            "background-color: rgba(200, 200, 200, 30);")  # Subtle background

        # build window
        titlebar = TitleBar(self)
        self.setMenuWidget(titlebar)
        self.setCentralWidget(central)
        self.setEnabled(False)

        # load styles
        QFontDatabase.addApplicationFont("./assets/Roboto-Black.ttf")
        with open("./assets/style.qss", "r", encoding="utf-8") as qss:
            self.app.setStyleSheet(qss.read())

        # show window
        self.show()

        # State tracking
        self.is_listening = False
        self.connected_users = []  # List to track users in the voice channel
        self.last_transcription_speaker = ""
        self.last_translation_speaker = ""

    # pylint: disable=invalid-name
    def mousePressEvent(self, event):
        """Handle mouse press events for window dragging.

        Overrides Qt's mousePressEvent method.
        """
        if event.button() == Qt.LeftButton:
            self.position = event.pos()
            event.accept()

    # pylint: disable=invalid-name
    def mouseMoveEvent(self, event):
        """Handle mouse move events for window dragging.

        Overrides Qt's mouseMoveEvent method.
        """
        if self.position is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.position)
            event.accept()

    # pylint: disable=invalid-name
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for window dragging.

        Overrides Qt's mouseReleaseEvent method.
        """
        self.position = None
        event.accept()

    def exclude(self, deselected, selected):
        """
        Manage visibility of servers across multiple connections.

        When a server is selected in one connection dropdown, it is hidden in all other 
        connection dropdowns to prevent the same server from being used in multiple connections.

        Args:
            deselected (int): Index of the previously selected server.
            selected (int): Index of the newly selected server.
        """
        self.connected_servers.add(selected)

        if deselected is not None:
            self.connected_servers.remove(deselected)

        for connection in self.connections:
            connection.servers.set_row_hidden(selected, True)

            if deselected is not None:
                connection.servers.set_row_hidden(deselected, False)

    async def run_qt(self, interval=0.01):
        """
        Run the Qt event loop within an asyncio context.

        This function allows the Qt GUI to remain responsive while running
        alongside asyncio coroutines.

        Args:
            interval (float): Time in seconds between processing Qt events.
        """
        while True:
            QCoreApplication.processEvents(
                QEventLoop.AllEvents, int(interval * 1000))
            await asyncio.sleep(interval)

    async def ready(self):
        """
        Initialize the application once the Discord bot is ready.

        This method is called when the bot has successfully connected to Discord
        and is ready to interact with the API. It populates the server list and
        enables the UI controls.
        """
        await self.bot.wait_until_ready()
        self.info.setText(f"Logged in as: {self.bot.user.name}")
        self.connections[0].set_servers(self.bot.guilds)
        Connection.resize_combobox(self.connections[0].servers)
        self.setEnabled(True)

    def _format_text(self, current_text, current_speaker, last_speaker, new_text):
        """Helper method to format text consistently with fewer branches."""
        if current_text and current_speaker == last_speaker:
            # Same speaker, append to the last line with punctuation handling
            lines = current_text.split("\n")
            last_line = lines[-1]
            # Check for punctuation
            ends_with_punctuation = any(last_line.endswith(p)
                                        for p in ['.', '!', '?'])

            # Format based on punctuation
            if ends_with_punctuation:
                return f"{current_text} {new_text}"
            return f"{current_text}. {new_text}"

        # New speaker or empty current text
        if current_text:
            return f"{current_text}\n{current_speaker}: {new_text}"
        return f"{current_speaker}: {new_text}"

    def update_text_display(self, transcribed_text, translated_text):
        """Append new transcriptions and translations to the text displays"""
        # Parse out the speaker from the incoming texts
        try:
            current_speaker_transcribed = transcribed_text.split(":", 1)[
                0].strip()
            current_text_transcribed = transcribed_text.split(":", 1)[
                1].strip()
            current_speaker_translated = translated_text.split(":", 1)[
                0].strip()
            current_text_translated = translated_text.split(":", 1)[1].strip()
        except IndexError:
            # Fallback if the text doesn't contain a colon
            current_speaker_transcribed = ""
            current_text_transcribed = transcribed_text
            current_speaker_translated = ""
            current_text_translated = translated_text

        # Get current text from displays
        current_transcribed = self.transcribed_display.toPlainText()
        current_translated = self.translated_display.toPlainText()

        # Process text using helper method
        new_transcribed = self._format_text(
            current_transcribed,
            current_speaker_transcribed,
            self.last_transcription_speaker,
            current_text_transcribed
        )

        new_translated = self._format_text(
            current_translated,
            current_speaker_translated,
            self.last_translation_speaker,
            current_text_translated
        )

        # Update last speaker tracking
        self.last_transcription_speaker = current_speaker_transcribed
        self.last_translation_speaker = current_speaker_translated

        # Update display
        self.transcribed_display.setPlainText(new_transcribed)
        self.translated_display.setPlainText(new_translated)

        # Scroll to the bottom
        self.transcribed_display.verticalScrollBar().setValue(
            self.transcribed_display.verticalScrollBar().maximum())
        self.translated_display.verticalScrollBar().setValue(
            self.translated_display.verticalScrollBar().maximum())

    async def process_audio_callback(self, sink, data):  # pylint: disable=unused-argument
        """
        Callback function to process audio data.
        This function must maintain this signature to work with discord.py's voice client.
        """
        print("Finished processing audio.")

    def force_stop_listening(self):
        """Force stop listening without relying on normal UI flow."""
        print("Emergency stop triggered!")

        if self.is_listening and self.vc:
            try:
                # Stop recording directly
                self.vc.stop_recording()
                print("Forced recording to stop.")

                # Reset state
                self.is_listening = False

                # Update any connection button states
                for connection in self.connections:
                    connection.listen.setText("Listen")
                    connection.listen.setStyleSheet("")

                # Kill any pending transcription workers
                for connection in self.connections:
                    if hasattr(connection, "sink") and connection.sink:
                        connection.sink.worker_running = False

                # Show confirmation to user
                QMessageBox.information(
                    self, "Emergency Stop",
                    "Recording has been force-stopped. "
                    "You may need to restart the application if issues persist."
                )

            except (discord.errors.ClientException, RuntimeError, AttributeError) as e:
                print(f"Error during emergency stop: {e}")
                QMessageBox.critical(
                    self, "Error",
                    "Could not stop recording properly. Please close and restart the application."
                )

    def cleanup_resources(self):
        """
        Clean up resources when the application is closing.

        Stops any active recordings, cleans up sinks, and disconnects
        from all voice clients.
        """
        print("Cleaning up resources...")

        if self.is_listening and self.vc:
            try:
                # Stop recording
                self.vc.stop_recording()
                print("Recording stopped during cleanup")

                # Reset state
                self.is_listening = False

                # Clean up sink resources
                for connection in self.connections:
                    if hasattr(connection, "sink") and connection.sink:
                        if hasattr(connection.sink, "cleanup"):
                            connection.sink.cleanup()
                        if hasattr(connection.sink, "worker_running"):
                            connection.sink.worker_running = False

                        # Give worker threads a chance to terminate
                        if hasattr(connection.sink, "worker_thread"):
                            connection.sink.worker_thread.join(timeout=1.0)
            except (discord.errors.ClientException, RuntimeError, AttributeError) as e:
                print(f"Error during cleanup: {e}")

        # Make sure all voice clients are disconnected
        for voice in self.bot.voice_clients:
            try:
                asyncio.run_coroutine_threadsafe(
                    voice.disconnect(), self.bot.loop)
            except (discord.errors.ClientException, RuntimeError) as e:
                print(f"Error disconnecting voice client: {e}")

    # pylint: disable=invalid-name
    def closeEvent(self, event):
        """
        Ensures that all resources are cleaned up and recordings are stopped
        before the application is closed.

        Args:
            event (QCloseEvent): The close event triggered by the user.
        """
        print("Window close event detected")

        # First stop any active recording
        if self.is_listening and self.vc:
            try:
                self.force_stop_listening()
            except (discord.errors.ClientException, RuntimeError, AttributeError) as e:
                print(f"Error stopping recording during close: {e}")

        # Clean up resources
        self.cleanup_resources()

        # Accept the close event and proceed with closing
        event.accept()

    def update_connected_users(self, users):
        """
        Update the UI to display connected users and their toggle controls.

        Creates toggle buttons for each connected user to enable/disable
        processing of their audio.

        Args:
            users (list): List of Discord Member objects
        """
        # Create a brand new layout
        self.user_toggle_layout = QVBoxLayout()
        self.user_toggle_layout.setContentsMargins(0, 0, 0, 0)

        # Delete the old container widget and create a new one
        old_widget = self.user_toggle_area
        self.user_toggle_area = QFrame()
        self.user_toggle_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.user_toggle_area.setMinimumHeight(100)
        self.user_toggle_area.setStyleSheet(
            "background-color: rgba(200, 200, 200, 30);")
        self.user_toggle_area.setLayout(self.user_toggle_layout)

        # Replace the old widget in the grid layout
        self.layout.replaceWidget(old_widget, self.user_toggle_area)

        # Schedule the old widget for deletion
        old_widget.setParent(None)
        old_widget.deleteLater()

        # Show a message if no users
        if not users:
            empty_label = QLabel("No users in channel")
            empty_label.setStyleSheet("color: gray;")
            self.user_toggle_layout.addWidget(empty_label)
            return

        # Rest of your existing user toggle creation code...
        # Create toggle for each user
        seen_user_ids = set()

        for user in users:
            # Skip if we've already processed this user
            if user.id in seen_user_ids:
                continue
            seen_user_ids.add(user.id)

            # Initialize enabled state if not exists
            if user.id not in self.user_processing_enabled:
                self.user_processing_enabled[user.id] = True

            # Create user toggle switch
            user_frame = QFrame()
            user_layout = QHBoxLayout(user_frame)
            user_layout.setContentsMargins(0, 0, 0, 0)

            # User label
            user_label = QLabel(user.display_name)
            user_layout.addWidget(user_label)

            # Toggle switch
            user_toggle = QPushButton()
            user_toggle.setCheckable(True)
            user_toggle.setChecked(self.user_processing_enabled[user.id])
            user_toggle.setText(
                "Enabled" if self.user_processing_enabled[user.id] else "Disabled")
            user_toggle.setStyleSheet(
                "background-color: #4CAF50;" if self.user_processing_enabled[user.id]
                else "background-color: #F44336;"
            )

            # Store user ID in the button (for identifying in callback)
            user_toggle.user_id = user.id

            # Connect toggle button to handler
            user_toggle.clicked.connect(
                lambda checked, btn=user_toggle: self.toggle_user_processing(btn))

            user_layout.addWidget(user_toggle)
            self.user_toggle_layout.addWidget(user_frame)

    def toggle_user_processing(self, button):
        """
        Toggle audio processing for a specific user.

        Args:
            button (QPushButton): The toggle button that was clicked,
                                 containing the user_id attribute
        """
        user_id = button.user_id
        enabled = button.isChecked()
        self.user_processing_enabled[user_id] = enabled

        # Update button appearance
        button.setText("Enabled" if enabled else "Disabled")
        button.setStyleSheet(
            "background-color: #4CAF50;" if enabled else "background-color: #F44336;"
        )

        print(
            f"Audio processing for user {user_id} is now {'enabled' if enabled else 'disabled'}")
