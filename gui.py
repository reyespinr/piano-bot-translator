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
                    except Exception as e:
                        logging.exception("Error starting recording: %s", e)
                        self.parent.is_listening = False
                        self.listen.setText("Listen")
                        self.listen.setStyleSheet("")  # Reset on error

        except Exception as e:
            logging.exception("Error in toggle_listen: %s", e)

    def clear_text_boxes(self):
        """Clear the transcribed and translated text boxes."""
        try:
            self.parent.transcribed_display.clear()
            self.parent.translated_display.clear()
        except Exception as e:
            logging.exception("Error clearing text boxes: %s", e)

    def update_text_display(self, transcribed_text, translated_text):
        """Append new transcriptions and translations to the text displays"""
        # Parse out the speaker from the incoming texts
        try:
            # Check if the text contains speaker information
            if ":" in transcribed_text:
                current_speaker_transcribed = transcribed_text.split(":", 1)[
                    0].strip()
                current_text_transcribed = transcribed_text.split(":", 1)[
                    1].strip()
            else:
                # If no speaker info, use "Unknown" and the full text
                current_speaker_transcribed = "Unknown"
                current_text_transcribed = transcribed_text

            if ":" in translated_text:
                current_speaker_translated = translated_text.split(":", 1)[
                    0].strip()
                current_text_translated = translated_text.split(":", 1)[
                    1].strip()
            else:
                current_speaker_translated = "Unknown"
                current_text_translated = translated_text
        except IndexError:
            # Fallback if the text doesn't contain a colon
            current_speaker_transcribed = "Unknown"
            current_text_transcribed = transcribed_text
            current_speaker_translated = "Unknown"
            current_text_translated = translated_text

        # Get current text from displays
        current_transcribed = self.transcribed_display.toPlainText()
        current_translated = self.translated_display.toPlainText()

        # Process text using improved helper method to combine consecutive messages
        new_transcribed = self._format_text_with_continuity(
            current_transcribed,
            current_speaker_transcribed,
            self.last_transcription_speaker,
            current_text_transcribed
        )

        new_translated = self._format_text_with_continuity(
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

    def _format_text_with_continuity(self, current_text, speaker, last_speaker, new_text):
        """Format text to append to existing text if from the same speaker.

        Args:
            current_text (str): Current text in the display
            speaker (str): Current speaker name
            last_speaker (str): Previous speaker name
            new_text (str): New text to add

        Returns:
            str: Formatted text with continuity for the same speaker
        """
        # If empty, just return the new text with speaker prefix
        if not current_text:
            return f"{speaker}: {new_text}"

        # Split the current text into lines
        lines = current_text.strip().split('\n')
        last_line = lines[-1] if lines else ""

        # Check if the last line starts with the same speaker
        if last_speaker == speaker and last_line.startswith(f"{speaker}:"):
            # Append to the existing line
            lines[-1] = f"{lines[-1]} {new_text}"
        else:
            # Add as a new line
            lines.append(f"{speaker}: {new_text}")

        # Join everything back together
        return '\n'.join(lines)

    # Replace the existing _format_text method with our new improved version
    def _format_text(self, current_text, speaker, last_speaker, new_text):
        """Backwards compatibility wrapper for the new _format_text_with_continuity method"""
        return self._format_text_with_continuity(current_text, speaker, last_speaker, new_text)
