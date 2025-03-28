import os
import sys
import sound
import asyncio
import logging
import discord
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
    QTextEdit  # Add QTextEdit for text display
)
# Import listen and other functions
# Import listen_and_transcribe
from utils import process_audio, listen, transcribe, translate, listen_and_transcribe

if getattr(sys, "frozen", False):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))


class Dropdown(QComboBox):
    changed = pyqtSignal(object, object)

    def __init__(self):
        super(Dropdown, self).__init__()

        self.setItemDelegate(QStyledItemDelegate())
        self.setPlaceholderText("None")
        self.setView(QListView())

        self.deselected = None
        self.currentIndexChanged.connect(self.changed_signal)

    def changed_signal(self, selected):
        self.changed.emit(self.deselected, selected)
        self.deselected = selected

    def setRowHidden(self, idx, hidden):
        self.view().setRowHidden(idx, hidden)


class SVGButton(QPushButton):
    def __init__(self, text=None):
        super(SVGButton, self).__init__(text)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.svg = QSvgWidget("./assets/loading.svg", self)
        self.svg.setVisible(False)

        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.svg)

    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.svg.setVisible(not enabled)


class Connection:
    def __init__(self, layer, parent):
        self.stream = sound.PCMStream()
        self.parent = parent
        self.voice = None

        # dropdowns
        self.devices = Dropdown()
        self.servers = Dropdown()
        self.channels = Dropdown()

        for device, idx in parent.devices.items():
            self.devices.addItem(device + "   ", idx)

        # mute
        self.mute = SVGButton("Mute")
        self.mute.setObjectName("mute")

        # record button
        self.record = SVGButton("Record")
        self.record.setObjectName("record")

        # Rename record button to listen
        self.listen = SVGButton("Listen")
        self.listen.setObjectName("listen")

        # Add clear button
        self.clear = QPushButton("Clear")
        self.clear.setObjectName("clear")

        # add widgets
        parent.layout.addWidget(self.devices, layer, 0)
        parent.layout.addWidget(self.servers, layer, 1)
        parent.layout.addWidget(self.channels, layer, 2)
        parent.layout.addWidget(self.mute, layer, 3)
        parent.layout.addWidget(self.listen, layer, 4)  # Listen button
        parent.layout.addWidget(self.clear, layer, 5)  # Clear button

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
        # Update event connection for listen button
        self.listen.clicked.connect(self.toggle_listen)
        self.clear.clicked.connect(
            self.clear_text_boxes)  # Connect clear button

    @staticmethod
    def resize_combobox(combobox):
        font = combobox.property("font")
        metrics = QFontMetrics(font)
        min_width = 0

        for i in range(combobox.count()):
            size = metrics.horizontalAdvance(combobox.itemText(i))
            if size > min_width:
                min_width = size

        combobox.setMinimumWidth(min_width + 50)

    def setEnabled(self, enabled):
        self.devices.setEnabled(enabled)
        self.servers.setEnabled(enabled)
        self.channels.setEnabled(enabled)

        self.mute.setEnabled(enabled)
        self.mute.setText("Mute" if enabled else "")

    def set_servers(self, guilds):
        for guild in guilds:
            self.servers.addItem(guild.name, guild)

    def change_device(self):
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

        except Exception:
            logging.exception("Error on change_device")

    async def change_server(self, deselcted, selected):
        try:
            selection = self.servers.itemData(selected)

            self.parent.exclude(deselcted, selected)
            self.channels.clear()
            self.channels.addItem("None", None)

            for channel in selection.channels:
                if isinstance(channel, discord.VoiceChannel):
                    self.channels.addItem(channel.name, channel)

            Connection.resize_combobox(self.channels)

        except Exception:
            logging.exception("Error on change_server")

    async def change_channel(self):
        try:
            selection = self.channels.currentData()
            self.mute.setText("Mute")
            self.setEnabled(False)

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

                not_playing = (
                    self.devices.currentData() is not None
                    and not self.voice.is_playing()
                )

                if not_playing:
                    self.voice.play(self.stream)

            else:
                if self.voice is not None:
                    await self.voice.disconnect()

        except Exception:
            logging.exception("Error on change_channel")

        finally:
            self.setEnabled(True)

    def toggle_mute(self):
        try:
            if self.voice is not None:
                if self.voice.is_playing():
                    self.voice.pause()
                    self.mute.setText("Resume")
                else:
                    self.voice.resume()
                    self.mute.setText("Mute")

        except Exception:
            logging.exception("Error on toggle_mute")

    def toggle_listen(self):
        """Start or stop real-time audio processing."""
        try:
            if self.parent.bot.voice_clients:
                vc = self.parent.bot.voice_clients[0]
                if not self.parent.is_listening:
                    self.parent.is_listening = True
                    self.listen.setText("Stop")
                    # Debugging statement
                    print("Now listening in real-time...")
                    asyncio.ensure_future(self.process_audio_with_gui(vc))
                else:
                    self.parent.is_listening = False
                    self.listen.setText("Listen")
                    print("Stopped listening.")  # Debugging statement
        except Exception:
            logging.exception("Error on toggle_listen")

    async def process_audio_with_gui(self, vc):
        """Continuously process audio in real-time and update the GUI."""
        try:
            await listen_and_transcribe(vc, self.parent)
        except Exception:
            logging.exception("Error during process_audio_with_gui")
        finally:
            print("Stopped real-time processing.")  # Debugging statement

    def clear_text_boxes(self):
        """Clear the transcribed and translated text boxes."""
        self.parent.transcribed_display.clear()
        self.parent.translated_display.clear()


class TitleBar(QFrame):
    def __init__(self, parent):
        # title bar
        super(TitleBar, self).__init__()
        self.setObjectName("titlebar")

        # discord
        self.parent = parent
        self.bot = parent.bot

        # layout
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 10)
        self.setLayout(layout)

        # window title
        title = QLabel("Discord Audio Pipe")

        # minimize
        minimize_button = QPushButton("—")
        minimize_button.setObjectName("minimize")
        layout.addWidget(minimize_button)

        # close
        close_button = QPushButton("✕")
        close_button.setObjectName("close")
        layout.addWidget(close_button)

        # add widgets
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(minimize_button)
        layout.addWidget(close_button)

        # events
        minimize_button.clicked.connect(self.minimize)
        close_button.clicked.connect(
            lambda: asyncio.ensure_future(self.close()))

    async def close(self):
        # workaround for logout bug
        for voice in self.bot.voice_clients:
            try:
                await voice.disconnect()
            except Exception:
                pass

        self.bot._closed = True
        await self.bot.ws.close()
        self.parent.close()

    def minimize(self):
        self.parent.showMinimized()


class GUI(QMainWindow):
    def __init__(self, app, bot):
        # app
        super(GUI, self).__init__()
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

        # Remove connection button logic
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

        # Add separate text display areas for transcribed and translated text
        self.transcribed_display = QTextEdit()
        self.transcribed_display.setReadOnly(True)
        self.transcribed_display.setPlaceholderText("Transcribed Text")
        self.layout.addWidget(self.transcribed_display, 4,
                              0, 1, 4)  # Spanning all columns

        self.translated_display = QTextEdit()
        self.translated_display.setReadOnly(True)
        self.translated_display.setPlaceholderText("Translated Text")
        self.layout.addWidget(self.translated_display, 5,
                              0, 1, 4)  # Spanning all columns

        # Remove connection button event

        # build window
        titlebar = TitleBar(self)
        self.setMenuWidget(titlebar)
        self.setCentralWidget(central)
        self.setEnabled(False)

        # load styles
        QFontDatabase.addApplicationFont("./assets/Roboto-Black.ttf")
        with open("./assets/style.qss", "r") as qss:
            self.app.setStyleSheet(qss.read())

        # show window
        self.show()

        self.is_recording = False  # Add recording state
        self.is_listening = False  # Rename recording state to listening

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.position = event.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.position is not None and event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.position = None
        event.accept()

    def setEnabled(self, enabled):
        # Remove connection button enable logic
        for connection in self.connections:
            connection.setEnabled(enabled)

    def add_connection(self):
        # Remove add_connection logic
        pass

    def exclude(self, deselected, selected):
        self.connected_servers.add(selected)

        if deselected is not None:
            self.connected_servers.remove(deselected)

        for connection in self.connections:
            connection.servers.setRowHidden(selected, True)

            if deselected is not None:
                connection.servers.setRowHidden(deselected, False)

    async def run_Qt(self, interval=0.01):
        while True:
            QCoreApplication.processEvents(
                QEventLoop.AllEvents, int(interval * 1000))
            await asyncio.sleep(interval)

    async def ready(self):
        await self.bot.wait_until_ready()

        self.info.setText(f"Logged in as: {self.bot.user.name}")
        self.connections[0].set_servers(self.bot.guilds)
        Connection.resize_combobox(self.connections[0].servers)
        self.setEnabled(True)

    def update_text_display(self, transcribed_text, translated_text):
        """Append new transcriptions and translations to the text displays."""
        current_transcribed = self.transcribed_display.toPlainText()
        current_translated = self.translated_display.toPlainText()

        self.transcribed_display.setPlainText(
            f"{current_transcribed}\n{transcribed_text}".strip()
        )
        self.translated_display.setPlainText(
            f"{current_translated}\n{translated_text}".strip()
        )
