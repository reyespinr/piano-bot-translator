/**
 * Piano Bot Translator - Production Frontend
 * Optimized WebSocket client with auto-reconnect, error handling, and performance optimizations
 */

class PianoBotClient {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnected = false;
        this.messageQueue = [];
        
        // UI state
        this.selectedGuild = null;
        this.selectedChannel = null;
        this.isListening = false;
        this.lastSpeakers = { transcription: null, translation: null };
        
        this.initializeElements();
        this.attachEventListeners();
        this.connect();
    }

    initializeElements() {
        // Cache DOM elements
        this.elements = {
            serverSelect: document.getElementById('server-select'),
            channelSelect: document.getElementById('channel-select'),
            joinButton: document.getElementById('join-btn'),
            leaveButton: document.getElementById('leave-btn'),
            listenButton: document.getElementById('listen-btn'),
            clearButton: document.getElementById('clear-btn'),
            connectionStatus: document.getElementById('connection-status'),
            transcriptionBox: document.getElementById('transcription-box'),
            translationsContainer: document.getElementById('translations-container'),
            botStatus: document.getElementById('bot-status'),
            userListContainer: null // Will be created dynamically
        };

        // Validate required elements exist
        const requiredElements = ['serverSelect', 'channelSelect', 'joinButton', 'leaveButton', 
                                'listenButton', 'clearButton', 'connectionStatus', 
                                'transcriptionBox', 'translationsContainer'];
        
        for (const elementName of requiredElements) {
            if (!this.elements[elementName]) {
                console.error(`Required element not found: ${elementName}`);
            }
        }
        
        // Clear both containers on page load to ensure clean state
        if (this.elements.transcriptionBox) {
            this.elements.transcriptionBox.innerHTML = '';
        }
        if (this.elements.translationsContainer) {
            this.elements.translationsContainer.innerHTML = '';
        }
    }

    attachEventListeners() {
        // Server selection
        this.elements.serverSelect.addEventListener('change', (e) => {
            this.selectedGuild = e.target.value;
            this.handleServerChange();
        });

        // Channel selection
        this.elements.channelSelect.addEventListener('change', (e) => {
            this.selectedChannel = e.target.options[e.target.selectedIndex].text;
            this.elements.joinButton.disabled = !e.target.value;
        });

        // Button events
        this.elements.joinButton.addEventListener('click', () => this.joinChannel());
        this.elements.leaveButton.addEventListener('click', () => this.leaveChannel());
        this.elements.listenButton.addEventListener('click', () => this.toggleListening());
        this.elements.clearButton.addEventListener('click', () => this.clearMessages());

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });

        // Handle window beforeunload
        window.addEventListener('beforeunload', () => {
            if (this.socket) {
                this.socket.close();
            }
        });
    }

    connect() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.socket = new WebSocket(wsUrl);
            this.setupWebSocketHandlers();
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.handleConnectionError();
        }
    }

    setupWebSocketHandlers() {
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('Connected', 'online');
            
            // Clear both message containers and reset speaker tracking on new connection
            this.clearMessages();
            
            // Process queued messages
            this.processMessageQueue();
            
            // Request initial data
            this.sendCommand('get_guilds');
        };

        this.socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.socket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.updateConnectionStatus('Disconnected', 'offline');
            
            if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError();
        };
    }

    scheduleReconnect() {
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        this.reconnectAttempts++;
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        this.updateConnectionStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'offline');
        
        setTimeout(() => {
            if (!this.isConnected) {
                this.connect();
            }
        }, delay);
    }

    handleConnectionError() {
        this.isConnected = false;
        this.updateConnectionStatus('Connection Error', 'offline');
        this.disableControls();
    }

    sendCommand(command, data = {}) {
        const payload = { command, ...data };
        
        if (this.isConnected && this.socket.readyState === WebSocket.OPEN) {
            try {
                this.socket.send(JSON.stringify(payload));
            } catch (error) {
                console.error('Failed to send command:', error);
                this.messageQueue.push(payload);
            }
        } else {
            this.messageQueue.push(payload);
        }
    }

    processMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            try {
                this.socket.send(JSON.stringify(message));
            } catch (error) {
                console.error('Failed to send queued message:', error);
                break;
            }
        }
    }

    handleMessage(message) {
        switch (message.type) {
            case 'status':
                this.handleStatusMessage(message);
                break;
            case 'guilds':
                this.populateServers(message.guilds);
                break;
            case 'channels':
                this.populateChannels(message.channels);
                break;
            case 'response':
                this.handleCommandResponse(message);
                break;
            case 'bot_status':
                this.updateBotStatus(message.ready);
                break;
            case 'transcription':
                this.handleTranscription(message.data);
                break;
            case 'translation':
                this.handleTranslation(message.data);
                break;
            case 'listen_status':
                this.updateListenButton(message.is_listening);
                break;
            case 'users_update':
                this.updateUserList(message.users, message.enabled_states);
                break;
            case 'user_joined':
                this.addUser(message.user, message.enabled);
                break;
            case 'user_left':
                this.removeUser(message.user_id);
                break;
            case 'user_toggle':
                this.updateUserToggle(message.user_id, message.enabled);
                break;
            default:
                console.warn('Unknown message type:', message.type);
        }
    }

    handleStatusMessage(message) {
        if (message.connected_channel) {
            this.updateConnectionStatus(`Connected to: ${message.connected_channel}`, 'online');
            this.enableChannelControls();
        }
        
        if (typeof message.is_listening !== 'undefined') {
            this.updateListenButton(message.is_listening);
        }
        
        if (message.users) {
            this.updateUserList(message.users, message.enabled_states);
        }
        
        if (message.translations?.length) {
            this.batchDisplayMessages(message.translations, 'translation');
        }
    }

    populateServers(guilds) {
        const select = this.elements.serverSelect;
        select.innerHTML = '<option value="">Select a server</option>';
        
        guilds.forEach(guild => {
            const option = document.createElement('option');
            option.value = guild.id;
            option.textContent = guild.name;
            select.appendChild(option);
        });
    }

    populateChannels(channels) {
        const select = this.elements.channelSelect;
        select.innerHTML = '<option value="">Select a channel</option>';
        select.disabled = false;
        
        channels.forEach(channel => {
            const option = document.createElement('option');
            option.value = channel.id;
            option.textContent = channel.name;
            select.appendChild(option);
        });
    }

    handleServerChange() {
        if (this.selectedGuild) {
            this.sendCommand('get_channels', { guild_id: this.selectedGuild });
        } else {
            this.elements.channelSelect.innerHTML = '<option value="">Select a channel</option>';
            this.elements.channelSelect.disabled = true;
            this.elements.joinButton.disabled = true;
        }
    }

    joinChannel() {
        const channelId = this.elements.channelSelect.value;
        if (channelId) {
            this.setButtonLoading(this.elements.joinButton, 'Joining...');
            this.sendCommand('join_channel', { channel_id: channelId });
        }
    }

    leaveChannel() {
        this.setButtonLoading(this.elements.leaveButton, 'Leaving...');
        this.sendCommand('leave_channel');
    }

    toggleListening() {
        if (this.elements.listenButton.disabled) return;
        
        const button = this.elements.listenButton;
        if (this.isListening) {
            this.setButtonLoading(button, 'Stopping...', 'danger');
        } else {
            this.setButtonLoading(button, 'Starting...', 'primary');
        }
        
        this.sendCommand('toggle_listen');
    }

    clearMessages() {
        this.elements.transcriptionBox.innerHTML = '';
        this.elements.translationsContainer.innerHTML = '';
        // CRITICAL: Reset last speakers when clearing
        this.lastSpeakers = { transcription: null, translation: null };
    }

    handleCommandResponse(message) {
        const { command, success, message: responseMessage } = message;
        
        switch (command) {
            case 'join_channel':
                this.resetButtonLoading(this.elements.joinButton, 'Join');
                if (success) {
                    this.enableChannelControls();
                    this.updateConnectionStatus(`Connected to: ${this.selectedChannel}`, 'online');
                } else {
                    this.updateConnectionStatus(`Error: ${responseMessage}`, 'error');
                }
                break;
            case 'leave_channel':
                this.resetButtonLoading(this.elements.leaveButton, 'Leave');
                if (success) {
                    this.disableChannelControls();
                    this.updateConnectionStatus('Not connected to any channel', 'offline');
                    this.clearUserList();
                }
                break;
            case 'toggle_listen':
                this.updateListenButton(message.is_listening);
                break;
        }
    }

    batchDisplayMessages(messages, type) {
        const container = type === 'transcription' ? 
            this.elements.transcriptionBox : this.elements.translationsContainer;
        
        messages.forEach(data => {
            this.displayMessageWithContinuity(container, data, type);
        });
        
        this.scrollToBottom(container);
        this.trimMessages(container);
    }

    displayMessageWithContinuity(container, data, type) {
        const user = data.user;
        const text = data.text;
        const userId = String(data.user_id); // Ensure userId is always a string
        
        // Get the appropriate last speaker
        const lastSpeaker = this.lastSpeakers[type];
        
        // Get existing messages in container
        const lastMessage = container.lastElementChild;
        
        // Check if we can combine with the last message
        // Must be same user AND the last message in the container AND same user ID
        if (lastSpeaker === user && lastMessage && String(lastMessage.dataset.userId) === userId) {
            // Same speaker, append to existing message
            const contentSpan = lastMessage.querySelector('.message-content');
            if (contentSpan) {
                // Add a space before appending the new text
                contentSpan.textContent += ' ' + text.trim();
                this.scrollToBottom(container);
                return;
            }
        }
        
        // Different speaker or no previous message, create new message
        this.createNewMessage(container, user, text, userId);
        
        // Update last speaker for this message type AFTER creating the message
        this.lastSpeakers[type] = user;
        
        this.scrollToBottom(container);
    }

    createNewMessage(container, user, text, userId) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message';
        messageElement.dataset.userId = userId;
        
        const userElement = document.createElement('strong');
        userElement.textContent = user + ': ';
        
        const contentElement = document.createElement('span');
        contentElement.className = 'message-content';
        contentElement.textContent = text.trim();
        
        messageElement.appendChild(userElement);
        messageElement.appendChild(contentElement);
        container.appendChild(messageElement);
    }

    updateListenButton(listening) {
        this.isListening = listening;
        const button = this.elements.listenButton;
        
        if (listening) {
            button.textContent = 'Stop';
            button.className = 'btn danger';
        } else {
            button.textContent = 'Listen';
            button.className = 'btn primary';
        }
        
        button.disabled = false;
    }

    updateBotStatus(ready) {
        if (this.elements.botStatus) {
            this.elements.botStatus.textContent = ready ? 'Online' : 'Offline';
            this.elements.botStatus.className = ready ? 'status-online' : 'status-offline';
        }
    }

    updateConnectionStatus(message, type) {
        this.elements.connectionStatus.textContent = message;
        this.elements.connectionStatus.className = `connection-${type}`;
    }

    setButtonLoading(button, text, className = 'primary') {
        button.textContent = text;
        button.disabled = true;
        button.className = `btn ${className} loading`;
    }

    resetButtonLoading(button, text, className = 'primary') {
        button.textContent = text;
        button.disabled = false;
        button.className = `btn ${className}`;
    }

    enableChannelControls() {
        this.elements.joinButton.disabled = true;
        this.elements.leaveButton.disabled = false;
        this.elements.listenButton.disabled = false;
        this.elements.clearButton.disabled = false;
    }

    disableChannelControls() {
        this.elements.joinButton.disabled = false;
        this.elements.leaveButton.disabled = true;
        this.elements.listenButton.disabled = true;
        this.elements.clearButton.disabled = true;
        this.updateListenButton(false);
    }

    disableControls() {
        Object.values(this.elements).forEach(element => {
            if (element?.disabled !== undefined) {
                element.disabled = true;
            }
        });
    }

    updateUserList(users, enabledStates) {
        const container = this.getUserListContainer();
        container.innerHTML = '<h3 class="user-list-heading">Connected Users</h3>';
        
        if (users?.length) {
            users.forEach(user => {
                const userElement = this.createUserElement(user, enabledStates?.[user.id] ?? true);
                container.appendChild(userElement);
            });
        } else {
            const noUsers = document.createElement('p');
            noUsers.className = 'no-users';
            noUsers.textContent = 'No users connected';
            container.appendChild(noUsers);
        }
    }

    createUserElement(user, enabled) {
        const userItem = document.createElement('div');
        userItem.className = 'user-toggle-item';
        userItem.id = `user-${user.id}`;
        
        const userName = document.createElement('div');
        userName.className = 'user-name';
        userName.textContent = user.name;
        
        const toggleSwitch = this.createToggleSwitch(user.id, enabled);
        
        userItem.appendChild(userName);
        userItem.appendChild(toggleSwitch);
        
        return userItem;
    }

    createToggleSwitch(userId, enabled) {
        const label = document.createElement('label');
        label.className = 'toggle-switch';
        
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = enabled;
        input.dataset.userId = userId;
        input.addEventListener('change', (e) => {
            this.toggleUser(userId, e.target.checked);
        });
        
        const slider = document.createElement('span');
        slider.className = 'toggle-slider';
        
        label.appendChild(input);
        label.appendChild(slider);
        
        return label;
    }

    toggleUser(userId, enabled) {
        this.sendCommand('toggle_user', { user_id: userId, enabled });
    }

    addUser(user, enabled) {
        const container = this.getUserListContainer();
        const noUsersMsg = container.querySelector('.no-users');
        if (noUsersMsg) noUsersMsg.remove();
        
        if (!document.getElementById(`user-${user.id}`)) {
            const userElement = this.createUserElement(user, enabled);
            container.appendChild(userElement);
        }
    }

    removeUser(userId) {
        const userElement = document.getElementById(`user-${userId}`);
        userElement?.remove();
        
        // Show "no users" message if list is empty
        const container = this.getUserListContainer();
        if (!container.querySelector('.user-toggle-item')) {
            const noUsers = document.createElement('p');
            noUsers.className = 'no-users';
            noUsers.textContent = 'No users connected';
            container.appendChild(noUsers);
        }
    }

    updateUserToggle(userId, enabled) {
        const userElement = document.getElementById(`user-${userId}`);
        const toggle = userElement?.querySelector('input[type="checkbox"]');
        if (toggle) {
            toggle.checked = enabled;
        }
    }

    clearUserList() {
        const container = this.getUserListContainer();
        container.innerHTML = '<h3 class="user-list-heading">Connected Users</h3>';
        const noUsers = document.createElement('p');
        noUsers.className = 'no-users';
        noUsers.textContent = 'No users connected';
        container.appendChild(noUsers);
    }

    getUserListContainer() {
        if (!this.elements.userListContainer) {
            this.elements.userListContainer = this.createUserListContainer();
        }
        return this.elements.userListContainer;
    }

    createUserListContainer() {
        let container = document.getElementById('user-list-container');
        if (container) return container;
        
        container = document.createElement('div');
        container.id = 'user-list-container';
        container.className = 'user-list-container';
        
        const controlPanel = document.querySelector('.control-panel');
        if (controlPanel) {
            controlPanel.appendChild(container);
        }
        
        return container;
    }

    pauseUpdates() {
        // Remove buffer timer clearing since we're not using it anymore
        // if (this.batchUpdateTimer) {
        //     cancelAnimationFrame(this.batchUpdateTimer);
        //     this.batchUpdateTimer = null;
        // }
    }

    resumeUpdates() {
        // Remove buffer processing since we're not using it anymore
        // this.processBatchUpdates();
    }

    handleTranscription(data) {
        if (data) {
            // Display immediately instead of buffering for transcriptions
            this.displayMessageWithContinuity(this.elements.transcriptionBox, data, 'transcription');
            this.trimMessages(this.elements.transcriptionBox);
        }
    }

    handleTranslation(data) {
        if (data) {
            // Display immediately instead of buffering for translations
            this.displayMessageWithContinuity(this.elements.translationsContainer, data, 'translation');
            this.trimMessages(this.elements.translationsContainer);
        }
    }

    scrollToBottom(container) {
        container.scrollTop = container.scrollHeight;
    }

    trimMessages(container) {
        const messages = container.children;
        while (messages.length > 100) {
            container.removeChild(messages[0]);
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.pianoBotClient = new PianoBotClient();
});
