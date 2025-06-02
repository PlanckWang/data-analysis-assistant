let sessionId = null;
        let currentProvider = 'openai';
        let availableProviders = [];
        let isProcessing = false;

        // Initialize session
        async function initSession() {
            try {
                const response = await fetch('/api/v1/session', { method: 'POST' });
                const parsedResponse = await response.json();
                if (parsedResponse.status === 'success' && parsedResponse.data) {
                    sessionId = parsedResponse.data.session_id;
                    availableProviders = parsedResponse.data.available_providers;
                    currentProvider = parsedResponse.data.default_provider;
                    document.getElementById('sessionInfo').textContent = `会话 ID: ${sessionId.substring(0, 8)}...`;
                } else {
                    throw new Error(parsedResponse.message || 'Failed to initialize session data.');
                }

                // Setup providers

                // Setup providers
                setupProviders();

            } catch (error) {
                console.error('Failed to initialize session:', error);
                showError('初始化失败，请刷新页面重试');
            }
        }

        // Setup provider buttons
        function setupProviders() {
            const selector = document.getElementById('providerSelector');
            selector.innerHTML = '';

            const providerNames = {
                'openai': 'ChatGPT',
                'anthropic': 'Claude',
                'deepseek': 'DeepSeek'
            };

            availableProviders.forEach(provider => {
                const button = document.createElement('button');
                button.className = `provider-button ${provider === currentProvider ? 'active' : ''}`;
                button.textContent = providerNames[provider] || provider;
                button.onclick = () => switchProvider(provider);
                selector.appendChild(button);
            });
        }

        // Switch provider
        async function switchProvider(provider) {
            if (provider === currentProvider) return;

            try {
                const response = await fetch('/api/v1/provider/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, provider })
                });
                const parsedResponse = await response.json();

                if (response.ok && parsedResponse.status === 'success') {
                    currentProvider = parsedResponse.data.provider;
                    setupProviders(); // Re-render buttons with new active provider
                    addSystemMessage(`已切换到 ${currentProvider.toUpperCase()} 模型`);
                } else {
                    showError(parsedResponse.detail || '切换模型失败');
                }
            } catch (error) {
                showError('切换模型失败: ' + error.message);
            }
        }

        // File upload handling
        function setupFileUpload() {
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');

            if (!uploadArea || !fileInput) {
                console.error('Upload elements not found');
                return;
            }

            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');

                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileUpload(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileUpload(e.target.files[0]);
                }
            });
        }

        // Setup message input handling
        function setupMessageInput() {
            const messageInput = document.getElementById('messageInput');

            if (!messageInput) {
                console.error('Message input not found');
                return;
            }

            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = this.scrollHeight + 'px';
                updateSendButton();
            });

            messageInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        // Handle file upload
        async function handleFileUpload(file) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            addUserMessage(`上传文件: ${file.name}`);
            showTypingIndicator();

            try {
                const response = await fetch('/api/v1/upload', {
                    method: 'POST',
                    body: formData
                });

                const parsedResponse = await response.json();
                hideTypingIndicator();

                if (response.ok && parsedResponse.status === 'success' && parsedResponse.data && parsedResponse.data.result) {
                    const uploadData = parsedResponse.data.result;
                    if (uploadData.error) {
                        addAssistantMessage(`上传失败: ${uploadData.error}`);
                    } else {
                        updateDatasetsList(uploadData);
                        addAssistantMessage(`文件 "${uploadData.dataset_name}" 上传成功！\n\n数据集信息：\n- 行数: ${uploadData.shape[0]}\n- 列数: ${uploadData.shape[1]}\n- 内存占用: ${uploadData.memory_usage.toFixed(2)} MB\n\n您可以开始询问关于这个数据集的问题了。`);
                    }
                } else {
                     addAssistantMessage(`文件上传失败: ${parsedResponse.detail || response.statusText || '未知错误'}`);
                }

            } catch (error) {
                hideTypingIndicator();
                addAssistantMessage(`上传出错: ${error.message || '网络错误'}`);
            }
        }

        // Update datasets list
        function updateDatasetsList(datasetInfo) {
            const container = document.getElementById('datasetsList');

            // Create dataset item if it doesn't exist
            let item = document.querySelector(`[data-dataset="${datasetInfo.dataset_name}"]`);
            if (!item) {
                item = document.createElement('div');
                item.className = 'dataset-item active';
                item.dataset.dataset = datasetInfo.dataset_name;
                container.appendChild(item);
            }

            item.innerHTML = `
                <strong>${datasetInfo.dataset_name}</strong><br>
                <small class="text-muted">
                    ${datasetInfo.shape[0]} 行 × ${datasetInfo.shape[1]} 列
                </small>
            `;

            // Remove empty message
            const emptyMsg = container.querySelector('.text-muted.small');
            if (emptyMsg) emptyMsg.remove();
        }

        // Send message
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || isProcessing) return;

            isProcessing = true;
            input.value = '';
            input.style.height = 'auto';
            updateSendButton();

            addUserMessage(message);
            showTypingIndicator();

            try {
                const response = await fetch('/api/v1/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message,
                        provider: currentProvider
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let messageElement = null;

                hideTypingIndicator();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.substring(6));

                                if (data.type === 'content') {
                                    assistantMessage += data.content;
                                    if (!messageElement) {
                                        messageElement = addAssistantMessage(assistantMessage, true);
                                    } else {
                                        updateMessage(messageElement, assistantMessage);
                                    }
                                } else if (data.type === 'tool_result') {
                                    displayToolResult(data.result);
                                } else if (data.type === 'error') {
                                    addAssistantMessage(`错误: ${data.error}`);
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                            }
                        }
                    }
                }

            } catch (error) {
                hideTypingIndicator();
                addAssistantMessage(`发生错误: ${error.message}`);
            } finally {
                isProcessing = false;
                updateSendButton();
            }
        }

        // Display tool results
        function displayToolResult(result) {
            const container = document.getElementById('messagesContainer');
            const lastMessage = container.lastElementChild;

            if (result.result && result.result.image) {
                // Display image visualization
                const img = document.createElement('img');
                img.src = result.result.image;
                img.className = 'visualization';
                lastMessage.querySelector('.message-content').appendChild(img);
            } else if (result.result && result.result.plot_json) {
                // Display Plotly visualization
                const plotDiv = document.createElement('div');
                plotDiv.className = 'visualization';
                lastMessage.querySelector('.message-content').appendChild(plotDiv);

                const plotData = JSON.parse(result.result.plot_json);
                Plotly.newPlot(plotDiv, plotData.data, plotData.layout);
            } else {
                // Display other results
                const resultDiv = document.createElement('div');
                resultDiv.className = 'tool-result';
                resultDiv.innerHTML = `<strong>工具: ${result.tool}</strong><br>` +
                    `<pre>${JSON.stringify(result.result, null, 2)}</pre>`;
                lastMessage.querySelector('.message-content').appendChild(resultDiv);
            }
        }

        // Message handling functions
        function addUserMessage(content) {
            const container = document.getElementById('messagesContainer');
            const message = document.createElement('div');
            message.className = 'message user';
            message.innerHTML = `
                <div class="message-content">
                    ${marked.parse(content)}
                </div>
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
            `;
            container.appendChild(message);
            scrollToBottom();
        }

        function addAssistantMessage(content, isStreaming = false) {
            const container = document.getElementById('messagesContainer');
            const message = document.createElement('div');
            message.className = 'message assistant';
            message.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    ${marked.parse(content)}
                </div>
            `;
            container.appendChild(message);
            scrollToBottom();

            if (isStreaming) {
                return message;
            }
        }

        function updateMessage(messageElement, content) {
            const contentDiv = messageElement.querySelector('.message-content');
            contentDiv.innerHTML = marked.parse(content);
            scrollToBottom();
        }

        function addSystemMessage(content) {
            const container = document.getElementById('messagesContainer');
            const message = document.createElement('div');
            message.className = 'text-center text-muted small my-3';
            message.textContent = content;
            container.appendChild(message);
            scrollToBottom();
        }

        function showTypingIndicator() {
            const container = document.getElementById('messagesContainer');
            const indicator = document.createElement('div');
            indicator.className = 'message assistant typing-indicator-message';
            indicator.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            container.appendChild(indicator);
            scrollToBottom();
        }

        function hideTypingIndicator() {
            const indicator = document.querySelector('.typing-indicator-message');
            if (indicator) indicator.remove();
        }

        function scrollToBottom() {
            const container = document.getElementById('messagesContainer');
            container.scrollTop = container.scrollHeight;
        }

        function showError(message) {
            const container = document.getElementById('messagesContainer');
            const error = document.createElement('div');
            error.className = 'alert alert-danger';
            error.textContent = message;
            container.appendChild(error);
            scrollToBottom();
        }

        // Input handling
        function updateSendButton() {
            const sendButton = document.getElementById('sendButton');
            const messageInput = document.getElementById('messageInput');

            if (!sendButton || !messageInput) return;

            const hasContent = messageInput.value.trim().length > 0;
            sendButton.disabled = !hasContent || isProcessing;
        }

        // Export chat
        async function exportChat() {
            try {
                const response = await fetch(`/api/v1/history/${sessionId}`);
                const parsedResponse = await response.json();

                if (response.ok && parsedResponse.status === 'success' && parsedResponse.data) {
                    const blob = new Blob([JSON.stringify(parsedResponse.data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `chat_history_${new Date().toISOString()}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                } else {
                    showError(parsedResponse.detail || '导出失败');
                }
            } catch (error) {
                showError('导出失败: ' + error.message);
            }
        }

        // Clear chat
        function clearChat() {
            if (confirm('确定要清空所有对话记录吗？')) {
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                addAssistantMessage('对话已清空。请问有什么可以帮助您的？');
            }
        }

        // Initialize on load
        window.addEventListener('load', () => {
            initSession();
            updateSendButton();

            // Configure marked options
            marked.setOptions({
                breaks: true,
                gfm: true
            });

            // Setup file upload and message input handling
            setupFileUpload();
            setupMessageInput();
        });
