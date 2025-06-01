// Get references to key UI elements
const serverSelect = document.getElementById('server-select');
const channelSelect = document.getElementById('channel-select');
const joinButton = document.getElementById('join-btn');
const leaveButton = document.getElementById('leave-btn');
const listenButton = document.getElementById('listen-btn');
const clearButton = document.getElementById('clear-btn');
const connectionStatus = document.getElementById('connection-status');
const transcriptionBox = document.getElementById('transcription-box');
const translationsContainer = document.getElementById('translations-container');

// Global state
let socket;
let selectedGuild = null;
let selectedChannel = null;
let isListening = false;

// Message tracking for continuity
let lastSpeakers = {
    transcription: null,
    translation: null
};

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function() {
        console.log('WebSocket connection established');
        sendCommand('get_guilds');
    };
    
    socket.onmessage = function(event) {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };
    
    socket.onclose = function() {
        console.log('WebSocket connection closed');
        setTimeout(connectWebSocket, 3000);
    };
}

// Send command to server
function sendCommand(command, data = {}) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        const payload = {
            command: command,
            ...data
        };
        socket.send(JSON.stringify(payload));
    } else {
        console.error('WebSocket not connected');
    }
}

// Handle messages from server
function handleWebSocketMessage(message) {
    console.log('Received message:', message);
    
    switch (message.type) {
        case 'status':
            // Handle initial status
            if (message.connected_channel) {
                connectionStatus.textContent = `Connected to: ${message.connected_channel}`;
                joinButton.disabled = true;
                leaveButton.disabled = false;
                listenButton.disabled = false;
                clearButton.disabled = false;
                
                // Show any existing translations
                if (message.translations && message.translations.length) {
                    translationsContainer.innerHTML = '';
                    message.translations.forEach(t => {
                        displayMessage(t, translationsContainer);
                    });
                }
            }
            break;
            
        case 'guilds':
            // Populate server dropdown
            serverSelect.innerHTML = '<option value="">Select a server</option>';
            message.guilds.forEach(guild => {
                const option = document.createElement('option');
                option.value = guild.id;
                option.textContent = guild.name;
                serverSelect.appendChild(option);
            });
            break;
            
        case 'channels':
            // Populate channel dropdown
            channelSelect.innerHTML = '<option value="">Select a channel</option>';
            message.channels.forEach(channel => {
                const option = document.createElement('option');
                option.value = channel.id;
                option.textContent = channel.name;
                channelSelect.appendChild(option);
            });
            break;
            
        case 'response':
            // Handle command responses
            handleCommandResponse(message);
            break;
            
        case 'bot_status':
            // Update bot status indicator
            const statusElement = document.getElementById('bot-status');
            if (statusElement) {
                if (message.ready) {
                    statusElement.textContent = 'Online';
                    statusElement.className = 'status-online';
                } else {
                    statusElement.textContent = 'Offline';
                    statusElement.className = 'status-offline';
                }
            }
            break;
            
        case 'transcription':
            // Display transcription with enhanced logging
            console.log('🎙️ Transcription received:', message.data);
            if (message.data) {
                displayMessageWithContinuity(transcriptionBox, message.data, 'transcription');
            } else {
                console.error('Received empty transcription data');
            }
            break;
            
        case 'translation':
            // Display translation with enhanced logging
            console.log('🌐 Translation received:', message.data);
            if (message.data) {
                displayMessageWithContinuity(translationsContainer, message.data, 'translation');
            } else {
                console.error('Received empty translation data');
            }
            break;
            
        case 'listen_status':
            // Update listen button state
            console.log('🎧 Listen status update received:', message.is_listening);
            updateListenButtonState(message.is_listening);
            break;
            
        case 'users_update':
            // Log and update the user list when we receive user updates
            console.log('Users update received:', message.users);
            if (message.users) {
                updateUserList(message.users, message.enabled_states);
            }
            break;
            
        case 'user_joined':
            // Handle when a new user joins the channel
            console.log('User joined:', message.user);
            if (message.user) {
                addUserToList(message.user, message.enabled);
            }
            break;
            
        case 'user_left':
            // Handle when a user leaves the channel
            console.log('User left:', message.user_id);
            if (message.user_id) {
                removeUserFromList(message.user_id);
            }
            break;
    }
}

// Handle responses to commands
function handleCommandResponse(message) {
    if (message.command === 'join_channel') {
        if (message.success) {
            connectionStatus.textContent = `Connected to: ${selectedChannel}`;
            joinButton.disabled = true;
            leaveButton.disabled = false;
            listenButton.disabled = false;
            clearButton.disabled = false;
            
            // Update bot status to online when successfully connected
            const statusElement = document.getElementById('bot-status');
            if (statusElement) {
                statusElement.textContent = 'Online';
                statusElement.className = 'status-online';
            }
        } else {
            connectionStatus.textContent = `Error: ${message.message}`;
        }
    } else if (message.command === 'leave_channel') {
        if (message.success) {
            connectionStatus.textContent = 'Not connected to any channel';
            joinButton.disabled = false;
            leaveButton.disabled = true;
            listenButton.disabled = true;
            clearButton.disabled = true;
            
            // Update bot status to offline when disconnected
            const statusElement = document.getElementById('bot-status');
            if (statusElement) {
                statusElement.textContent = 'Offline';
                statusElement.className = 'status-offline';
            }
        }
    } else if (message.command === 'toggle_listen') {
        console.log('Toggle listen response:', message);
        updateListenButtonState(message.is_listening);
    }
}

// Display a message in a container with improved logging
function displayMessage(messageData, container) {
    if (!messageData || !container) {
        console.error('Invalid message data or container:', messageData, container);
        return;
    }
    
    // Create message element
    const div = document.createElement('div');
    div.className = 'message';
    div.innerHTML = `<strong>${messageData.user}:</strong> ${messageData.text}`;
    
    // Add to container
    container.appendChild(div);
    
    // Scroll to the bottom
    container.scrollTop = container.scrollHeight;
    
    // Log for debugging
    console.log(`Message displayed in ${container.id}: ${messageData.user}: ${messageData.text}`);
}

// Add this new function for displaying messages with continuity
function displayMessageWithContinuity(container, data, messageType) {
    const user = data.user;
    const text = data.text;
    const userId = data.user_id;
    
    // Get the appropriate last speaker
    const lastSpeakerKey = messageType === 'transcription' ? 'transcription' : 'translation';
    const lastSpeaker = lastSpeakers[lastSpeakerKey];
    
    // Get existing messages in container
    const messages = container.querySelectorAll('.message');
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
    
    // Debugging
    console.log(`Processing ${messageType} from ${user} (${userId})`);
    console.log(`Last speaker was: ${lastSpeaker}`);
    
    if (lastSpeaker === user && lastMessage) {
        // Same speaker, append to existing message
        console.log("🔄 Combining with previous message");
        const contentSpan = lastMessage.querySelector('.message-content');
        
        // Check if content span exists
        if (contentSpan) {
            contentSpan.textContent += " " + text.trim();
            console.log(`Combined message: ${contentSpan.textContent}`);
        } else {
            console.error("Error: No content span found in last message");
            // Fallback: create new message
            createNewMessage(container, user, text);
        }
    } else {
        // Different speaker, create new message
        console.log("➕ Creating new message");
        createNewMessage(container, user, text);
    }
    
    // Update last speaker
    lastSpeakers[lastSpeakerKey] = user;
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
    
    console.log(`Message displayed in ${container.id}: ${user}: ${text}`);
}

// Helper function to create new message element
function createNewMessage(container, user, text) {
    const messageElement = document.createElement('div');
    messageElement.className = 'message';
    
    const userElement = document.createElement('strong');
    userElement.textContent = user + ": ";
    
    const contentElement = document.createElement('span');
    contentElement.className = 'message-content';
    contentElement.textContent = text.trim();
    
    messageElement.appendChild(userElement);
    messageElement.appendChild(contentElement);
    container.appendChild(messageElement);
}

// Update listen button state with visual feedback
function updateListenButtonState(listening) {
    console.log(`Updating listen button state: ${listening}`);
    
    // Update global state
    isListening = listening;
    
    // Update button appearance
    if (listening) {
        listenButton.textContent = 'Stop';
        listenButton.classList.add('listening');
        listenButton.style.backgroundColor = '#F04747';
        listenButton.style.color = 'white';
        listenButton.style.fontWeight = 'bold';
    } else {
        listenButton.textContent = 'Listen';
        listenButton.classList.remove('listening');
        listenButton.style.backgroundColor = '';
        listenButton.style.color = '';
        listenButton.style.fontWeight = '';
    }
    
    // Re-enable the button
    listenButton.disabled = false;
}

// Function to update the user list display
function updateUserList(users, enabledStates) {
    // Get or create the user list container
    const userListContainer = document.getElementById('user-list-container') || createUserListContainer();
    userListContainer.innerHTML = '';
    
    // Add heading
    const heading = document.createElement('h3');
    heading.textContent = 'Connected Users';
    heading.className = 'user-list-heading';
    userListContainer.appendChild(heading);
    
    // Create user entries
    if (users && users.length > 0) {
        users.forEach(user => {
            const userItem = createUserToggleItem(user, enabledStates ? enabledStates[user.id] : true);
            userListContainer.appendChild(userItem);
        });
    } else {
        // Show a message if no users
        const noUsers = document.createElement('p');
        noUsers.textContent = 'No users connected';
        noUsers.className = 'no-users';
        userListContainer.appendChild(noUsers);
    }
}

// Helper function to create a user toggle item
function createUserToggleItem(user, enabled = true) {
    const userItem = document.createElement('div');
    userItem.className = 'user-toggle-item';
    userItem.id = `user-${user.id}`;
    
    // User name/avatar
    const userName = document.createElement('div');
    userName.className = 'user-name';
    userName.textContent = user.name;
    
    // Toggle switch
    const toggleSwitch = document.createElement('label');
    toggleSwitch.className = 'toggle-switch';
    
    const toggleInput = document.createElement('input');
    toggleInput.type = 'checkbox';
    toggleInput.checked = enabled;
    toggleInput.setAttribute('data-user-id', user.id);
    toggleInput.addEventListener('change', function() {
        toggleUserProcessing(user.id, this.checked);
    });
    
    const toggleSlider = document.createElement('span');
    toggleSlider.className = 'toggle-slider';
    
    toggleSwitch.appendChild(toggleInput);
    toggleSwitch.appendChild(toggleSlider);
    
    userItem.appendChild(userName);
    userItem.appendChild(toggleSwitch);
    
    return userItem;
}

// Function to create the user list container if it doesn't exist
function createUserListContainer() {
    // Check if container already exists
    let container = document.getElementById('user-list-container');
    if (container) return container;
    
    // Create container
    container = document.createElement('div');
    container.id = 'user-list-container';
    container.className = 'user-list-container panel';
    
    // Find the main content container
    const contentContainer = document.querySelector('.container');
    if (contentContainer) {
        // Create the content layout if it doesn't exist yet
        let contentLayout = document.querySelector('.content');
        if (!contentLayout) {
            contentLayout = document.createElement('div');
            contentLayout.className = 'content';
            contentContainer.appendChild(contentLayout);
        }
        
        // Create left column for controls if it doesn't exist
        let leftColumn = document.querySelector('.control-panel');
        if (!leftColumn) {
            leftColumn = document.createElement('div');
            leftColumn.className = 'control-panel';
            contentLayout.appendChild(leftColumn);
        }
        
        // Append user list to the left column - under controls
        leftColumn.appendChild(container);
    } else {
        // Fallback - add to body
        document.body.appendChild(container);
    }
    
    return container;
}

// Function to toggle user processing
function toggleUserProcessing(userId, enabled) {
    console.log(`Toggling user ${userId} processing to ${enabled}`);
    sendCommand('toggle_user', {
        user_id: userId,
        enabled: enabled
    });
}

// Function to add a user to the list
function addUserToList(user, enabled = true) {
    const userListContainer = document.getElementById('user-list-container') || createUserListContainer();
    
    // Check if user already exists
    if (document.getElementById(`user-${user.id}`)) return;
    
    // Add user
    const userItem = createUserToggleItem(user, enabled);
    userListContainer.appendChild(userItem);
}

// Function to remove a user from the list
function removeUserFromList(userId) {
    const userItem = document.getElementById(`user-${userId}`);
    if (userItem) {
        userItem.remove();
    }
}

// Event Listeners
serverSelect.addEventListener('change', function() {
    selectedGuild = this.value;
    if (selectedGuild) {
        sendCommand('get_channels', { guild_id: selectedGuild });
    } else {
        channelSelect.innerHTML = '<option value="">Select a channel</option>';
        channelSelect.disabled = true;
    }
});

channelSelect.addEventListener('change', function() {
    selectedChannel = this.options[this.selectedIndex].text;
    joinButton.disabled = !this.value;
});

joinButton.addEventListener('click', function() {
    if (channelSelect.value) {
        sendCommand('join_channel', { channel_id: channelSelect.value });
    }
});

leaveButton.addEventListener('click', function() {
    sendCommand('leave_channel');
});

listenButton.addEventListener('click', function() {
    if (listenButton.disabled) return;
    
    // Temporarily disable the button
    listenButton.disabled = true;
    
    // Provide immediate visual feedback with correct messaging
    if (!isListening) {
        // Going from not listening to listening
        listenButton.textContent = 'Starting...';
        listenButton.style.backgroundColor = '#F04747';
        listenButton.style.color = 'white';
    } else {
        // Going from listening to not listening
        listenButton.textContent = 'Stopping...';
        // Keep the red styling during the "Stopping..." phase
        listenButton.style.backgroundColor = '#F04747';
        listenButton.style.color = 'white';
    }
    
    // Send the command
    sendCommand('toggle_listen');
});

clearButton.addEventListener('click', function() {
    transcriptionBox.innerHTML = '';
    translationsContainer.innerHTML = '';
});

// Initialize the app on page load
document.addEventListener('DOMContentLoaded', function() {
    // Make sure message containers exist
    if (!transcriptionBox) {
        console.error('Transcription box not found!');
    }
    if (!translationsContainer) {
        console.error('Translations container not found!');
    }
    
    // Connect to WebSocket
    connectWebSocket();
});
