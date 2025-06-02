let sessionId = null;
        let currentProvider = 'openai';
        let availableProviders = [];
        let isProcessing = false;

        // Element selectors
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const datasetSelect = document.getElementById('datasetSelect');
        const switchDatasetButton = document.getElementById('switchDatasetButton');
        const chatLog = document.getElementById('messagesContainer'); // Assuming this is the main chat log

        // Function to re-initialize session (reloads the page)
        function reinitializeSession() {
            location.reload();
        }

        function disableAllInputs() {
            if (messageInput) messageInput.disabled = true;
            if (sendButton) sendButton.disabled = true;
            if (fileInput) fileInput.disabled = true;
            if (uploadArea) uploadArea.style.pointerEvents = 'none'; // To prevent drag/drop and click
            if (datasetSelect) datasetSelect.disabled = true;
            if (switchDatasetButton) switchDatasetButton.disabled = true;
            document.querySelectorAll('.provider-button').forEach(button => button.disabled = true);
            isProcessing = true; // Prevent sending messages if session is expired
            updateSendButton(); // Reflect disabled state on send button if it uses isProcessing
        }

        function enableAllInputs() {
            if (messageInput) messageInput.disabled = false;
            // sendButton enabled by updateSendButton based on input
            if (fileInput) fileInput.disabled = false;
            if (uploadArea) uploadArea.style.pointerEvents = 'auto';
            if (datasetSelect) datasetSelect.disabled = false; // Will be re-enabled by populateDatasetSelect if datasets exist
            if (switchDatasetButton) switchDatasetButton.disabled = false; // Same as above
            document.querySelectorAll('.provider-button').forEach(button => button.disabled = false);
            isProcessing = false;
            updateSendButton();
        }


        // Initialize session
        async function initSession() {
            disableAllInputs(); // Disable inputs while initializing
            try {
                const response = await fetch('/api/v1/session', { method: 'POST' });
                const parsedResponse = await response.json();
                if (response.ok && parsedResponse.status === 'success' && parsedResponse.data) {
                    sessionId = parsedResponse.data.session_id;
                    availableProviders = parsedResponse.data.available_providers;
                    currentProvider = parsedResponse.data.default_provider;
                    document.getElementById('sessionInfo').textContent = `会话 ID: ${sessionId.substring(0, 8)}...`;
                    enableAllInputs(); // Enable inputs after successful session init
                    setupProviders();
                    await fetchDatasetList(); // Fetch datasets after session is up
                } else {
                    throw new Error(parsedResponse.message || parsedResponse.detail || '未能初始化会话数据。');
                }
            } catch (error) {
                console.error('初始化会话失败:', error);
                addSystemMessage(`初始化会话失败: ${error.message}. 请刷新页面重试.`);
                disableAllInputs(); // Keep inputs disabled on failure
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

                hideTypingIndicator();

                if (!response.ok) {
                    if (response.status === 401 || response.status === 403) {
                        addSystemMessage('您的会话已过期，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                        disableAllInputs();
                        throw new Error('Session expired');
                    }
                    const errorData = await response.json().catch(() => ({ detail: '上传文件时发生未知服务端错误。' }));
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                }

                const parsedResponse = await response.json();
                if (parsedResponse.status === 'success' && parsedResponse.data && parsedResponse.data.result) {
                    const uploadData = parsedResponse.data.result;
                    if (uploadData.error) {
                        addAssistantMessage(`上传失败: ${uploadData.error}`);
                    } else {
                        // The old updateDatasetsList might be removed or repurposed.
                        // The new dropdown will be updated by fetchDatasetList.
                        addAssistantMessage(`文件 "${uploadData.dataset_name}" 上传成功！\n\n数据集信息：\n- 行数: ${uploadData.shape[0]}\n- 列数: ${uploadData.shape[1]}\n- 内存占用: ${uploadData.memory_usage.toFixed(2)} MB\n\n您可以开始询问关于这个数据集的问题了。`);
                        await fetchDatasetList(); // Refresh dataset dropdown
                    }
                } else {
                     addAssistantMessage(`文件上传失败: ${parsedResponse.message || parsedResponse.detail || '解析响应失败'}`);
                }

            } catch (error) {
                hideTypingIndicator();
                if (error.message !== 'Session expired') {
                    addAssistantMessage(`上传出错: ${error.message || '网络错误'}`);
                }
            } finally {
                isProcessing = false;
                updateSendButton();
            }
        }

        // Populate dataset select dropdown
        function populateDatasetSelect(datasetListData) {
            if (!datasetSelect || !switchDatasetButton) return;

            datasetSelect.innerHTML = ''; // Clear existing options
            const datasetsListDiv = document.getElementById('datasetsList');
            if (datasetsListDiv) datasetsListDiv.innerHTML = ''; // Clear the old list display area

            if (!datasetListData || !datasetListData.datasets || datasetListData.datasets.length === 0) {
                const option = document.createElement('option');
                option.value = "";
                option.textContent = "-- 无可用数据集 --";
                datasetSelect.appendChild(option);
                datasetSelect.disabled = true;
                switchDatasetButton.disabled = true;
                 if (datasetsListDiv) { // Update the old list display area as well
                    const p = document.createElement('p');
                    p.className = 'text-muted small';
                    p.textContent = '尚未加载数据集';
                    datasetsListDiv.appendChild(p);
                }
                return;
            }

            datasetListData.datasets.forEach(ds => {
                const option = document.createElement('option');
                option.value = ds.name;
                option.textContent = `${ds.name} (${ds.shape[0]}x${ds.shape[1]})`;
                if (ds.is_current || ds.name === datasetListData.current_dataset) {
                    option.selected = true;
                    option.textContent += ' (当前)';
                }
                datasetSelect.appendChild(option);

                // Optionally, if you still want to populate the old datasetsList div:
                if (datasetsListDiv) {
                    const item = document.createElement('div');
                    item.className = `dataset-item ${ds.is_current || ds.name === datasetListData.current_dataset ? 'active' : ''}`;
                    item.dataset.dataset = ds.name;
                    item.innerHTML = `<strong>${ds.name}</strong><br><small class="text-muted">${ds.shape[0]} 行 × ${ds.shape[1]} 列</small>`;
                    datasetsListDiv.appendChild(item);
                }
            });
            datasetSelect.disabled = false;
            switchDatasetButton.disabled = false;
        }

        // Fetch dataset list from server
        async function fetchDatasetList() {
            if (!sessionId) return;

            try {
                const response = await fetch(`/api/v1/datasets/${sessionId}`);
                if (!response.ok) {
                    if (response.status === 401 || response.status === 403) {
                        addSystemMessage('您的会话已过期，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                        disableAllInputs();
                        return; // Stop further processing
                    }
                    throw new Error(`获取数据集列表失败: ${response.statusText} (${response.status})`);
                }
                const result = await response.json();
                if (result.status === 'success' && result.data) {
                    populateDatasetSelect(result.data);
                } else {
                    throw new Error(result.message || result.detail || '获取数据集列表时服务器返回错误');
                }
            } catch (error) {
                console.error('获取数据集列表错误:', error);
                addSystemMessage(`获取数据集列表时出错: ${error.message}`);
                populateDatasetSelect(null); // Clear or disable list on error
            }
        }

        // Handle dataset switch
        async function handleSwitchDataset() {
            if (!sessionId || !datasetSelect.value) {
                addSystemMessage('请先选择一个数据集进行切换。');
                return;
            }
            const selectedDataset = datasetSelect.value;

            try {
                switchDatasetButton.disabled = true;
                switchDatasetButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 切换中...';

                const response = await fetch('/api/v1/datasets/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, dataset_name: selectedDataset })
                });

                if (!response.ok) {
                    if (response.status === 401 || response.status === 403) {
                        addSystemMessage('您的会话已过期，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                        disableAllInputs();
                        return; // Stop further processing
                    }
                    const errorResult = await response.json().catch(() => ({detail: "切换数据集时发生未知服务端错误。"}));
                    throw new Error(errorResult.detail || `切换数据集失败: ${response.statusText} (${response.status})`);
                }
                const result = await response.json();
                if (result.status === 'success' && result.data && result.data.success) {
                    addSystemMessage(`已成功切换到数据集: ${result.data.current_dataset}`);
                    await fetchDatasetList(); // Refresh dataset list and current status
                } else {
                    throw new Error(result.message || (result.data && result.data.message) || '切换数据集失败，服务器返回否定响应。');
                }
            } catch (error) {
                console.error('切换数据集错误:', error);
                addSystemMessage(`切换数据集时出错: ${error.message}`);
            } finally {
                switchDatasetButton.disabled = false; // Re-enable, populateDatasetSelect will manage final state
                switchDatasetButton.innerHTML = '<i class="fas fa-exchange-alt"></i> 切换数据集';
            }
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

                if (!response.ok) {
                    if (response.status === 401 || response.status === 403) {
                        hideTypingIndicator();
                        addSystemMessage('您的会话已过期，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                        disableAllInputs();
                        throw new Error('Session expired before stream started');
                    }
                    const errorText = await response.text();
                    hideTypingIndicator();
                    addAssistantMessage(`建立流式连接失败: ${errorText || response.statusText}`);
                    isProcessing = false; // Reset processing state
                    updateSendButton(); // Update button based on new state
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

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
                                    console.error('Stream error from server:', data.error);
                                    if (data.error.toLowerCase().includes('session not found') ||
                                        data.error.toLowerCase().includes('session has expired') ||
                                        data.error.toLowerCase().includes('会话未找到') ||
                                        data.error.toLowerCase().includes('redis might be down')) {
                                        addSystemMessage('您的会话已过期或无效，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                                        disableAllInputs();
                                        if (reader) reader.cancel().catch(e => console.error("Error cancelling reader:", e));
                                        return;
                                    } else {
                                        // Use addSystemMessage for server-side stream errors to make them distinct
                                        addSystemMessage(`服务器处理错误: ${data.error}`);
                                    }
                                }
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                                addAssistantMessage(`解析流数据时出错: ${e.message}`);
                            }
                        }
                    }
                }

            } catch (error) {
                hideTypingIndicator();
                if (error.message !== 'Session expired before stream started' && !error.message.toLowerCase().includes('session expired')) {
                    addAssistantMessage(`发生错误: ${error.message}`);
                }
            } finally {
                isProcessing = false;
                updateSendButton(); // This will re-enable send button if messageInput has content
            }
        }

        function escapeHtml(unsafe) {
            if (unsafe === null || unsafe === undefined) return '';
            return String(unsafe)
                 .replace(/&/g, "&amp;")
                 .replace(/</g, "&lt;")
                 .replace(/>/g, "&gt;")
                 .replace(/"/g, "&quot;")
                 .replace(/'/g, "&#039;");
        }

        // Function to trigger file download
        function triggerDownload(content, filename, contentType) {
            const blob = new Blob([content], { type: contentType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // Export table data to CSV
        function exportTableToCsv(data, filename = 'export.csv') {
            if (!Array.isArray(data) || data.length === 0) {
                console.warn('No data to export for CSV or data is not an array.');
                addSystemMessage('无法导出 CSV：没有数据或数据格式不正确。');
                return;
            }
             // Ensure all items in data are objects, for safety, though renderStructuredData implies this for tables
            if (typeof data[0] !== 'object' || data[0] === null) {
                console.warn('CSV export expects an array of objects.');
                addSystemMessage('无法导出 CSV：数据格式不是对象数组。');
                return;
            }

            const headers = Object.keys(data[0]);
            let csvContent = headers.join(',') + '\n';

            data.forEach(row => {
                const values = headers.map(header => {
                    let value = row[header];
                    if (value === null || value === undefined) {
                        value = '';
                    } else if (typeof value === 'object') {
                        value = JSON.stringify(value); // Stringify nested objects/arrays
                    }
                    const stringValue = String(value);
                    if (stringValue.includes('"') || stringValue.includes(',')) {
                        return `"${stringValue.replace(/"/g, '""')}"`;
                    }
                    return stringValue;
                });
                csvContent += values.join(',') + '\n';
            });

            triggerDownload(csvContent, filename, 'text/csv;charset=utf-8;');
        }

        // Export object data to JSON
        function exportObjectToJson(data, filename = 'export.json') {
            if (typeof data !== 'object' || data === null) {
                console.warn('No data to export for JSON or data is not an object.');
                addSystemMessage('无法导出 JSON：没有数据或数据格式不正确。');
                return;
            }
            try {
                const jsonContent = JSON.stringify(data, null, 2);
                triggerDownload(jsonContent, filename, 'application/json;charset=utf-8;');
            } catch (e) {
                console.error('Error stringifying JSON for export:', e);
                addSystemMessage(`无法导出 JSON: ${e.message}`);
            }
        }

        function renderStructuredData(data, depth = 0, originalData = null) {
            // originalData is used to pass the top-level data for export buttons
            // If called externally, originalData should be the same as data for the initial call.
            if (depth === 0 && originalData === null) {
                originalData = data;
            }

            let contentHtml = ''; // Stores the HTML for the data itself

            if (data === null || data === undefined) {
                contentHtml = '<span class="json-null">null</span>';
            } else if (typeof data === 'string') {
                if (data.includes('\n')) {
                     contentHtml = `<pre class="json-string">${escapeHtml(data)}</pre>`;
                } else {
                    contentHtml = `<span class="json-string">"${escapeHtml(data)}"</span>`;
                }
            } else if (typeof data === 'number') {
                contentHtml = `<span class="json-number">${data}</span>`;
            } else if (typeof data === 'boolean') {
                contentHtml = `<span class="json-boolean">${data}</span>`;
            } else if (Array.isArray(data)) {
                if (data.length === 0) {
                    contentHtml = '[]';
                } else if (depth === 0 && data.length > 0 && typeof data[0] === 'object' && data[0] !== null && !Array.isArray(data[0])) {
                    // Table rendering (Array of Objects)
                    let table = '<table class="json-table">';
                    const headers = Array.from(new Set(data.flatMap(obj => Object.keys(obj))));
                    if (headers.length > 0) {
                        table += '<thead><tr>';
                        headers.forEach(header => table += `<th>${escapeHtml(header)}</th>`);
                        table += '</tr></thead>';
                    }
                    table += '<tbody>';
                    data.forEach(row => {
                        table += '<tr>';
                        headers.forEach(header => {
                            table += `<td>${renderStructuredData(row[header], depth + 1, row[header])}</td>`;
                        });
                        table += '</tr>';
                    });
                    table += '</tbody></table>';
                    contentHtml = table;
                } else { // Simple array rendering
                    let list = '<ul class="json-array">';
                    data.forEach(item => {
                        list += `<li>${renderStructuredData(item, depth + 1, item)}</li>`;
                    });
                    list += '</ul>';
                    contentHtml = list;
                }
            } else if (typeof data === 'object') {
                if (Object.keys(data).length === 0) {
                    contentHtml = '{}';
                } else if (data.error && Object.keys(data).length === 1 && depth === 0) {
                    // This is a specific error object like {"error": "message"}
                    contentHtml = `<div class="json-error-server tool-error-message">${escapeHtml(data.error)}</div>`;
                }
                 else {
                    let dl = '<dl class="json-object">';
                    for (const key in data) {
                        if (data.hasOwnProperty(key)) {
                            dl += `<dt>${escapeHtml(key)}:</dt>`;
                            dl += `<dd>${renderStructuredData(data[key], depth + 1, data[key])}</dd>`;
                        }
                    }
                    dl += '</dl>';
                    contentHtml = dl;
                }
            } else {
                contentHtml = `<span class="json-fallback">${escapeHtml(String(data))}</span>`;
            }

            // If at depth 0, wrap content and add export buttons
            if (depth === 0) {
                let wrapperDiv = document.createElement('div');
                wrapperDiv.className = 'structured-data-wrapper';
                wrapperDiv.innerHTML = contentHtml;

                // Add export buttons based on the type of originalData
                if (Array.isArray(originalData) && originalData.length > 0 && typeof originalData[0] === 'object' && originalData[0] !== null && !Array.isArray(originalData[0])) {
                    const exportCsvButton = document.createElement('button');
                    exportCsvButton.textContent = '导出为 CSV';
                    exportCsvButton.className = 'export-button export-csv-button btn btn-sm btn-outline-primary mt-2';
                    exportCsvButton.onclick = function() {
                        exportTableToCsv(originalData, `table_export_${Date.now()}.csv`);
                    };
                    wrapperDiv.appendChild(exportCsvButton);
                } else if (typeof originalData === 'object' && originalData !== null && !Array.isArray(originalData)) {
                    // Also add for error objects if they are the top-level data
                     if (Object.keys(originalData).length > 0) { // Don't add for empty objects
                        const exportJsonButton = document.createElement('button');
                        exportJsonButton.textContent = '导出为 JSON';
                        exportJsonButton.className = 'export-button export-json-button btn btn-sm btn-outline-secondary mt-2';
                        exportJsonButton.onclick = function() {
                            exportObjectToJson(originalData, `data_export_${Date.now()}.json`);
                        };
                        wrapperDiv.appendChild(exportJsonButton);
                    }
                }
                return wrapperDiv.outerHTML;
            }

            return contentHtml; // For nested calls, return only the content HTML
        }


        // Display tool results
        // The 'result' parameter here is the 'result' field from the tool_results array in chatbot_app.py
        // which is the direct output of an MCP tool.
        function displayToolResult(toolCallResult) {
            const container = document.getElementById('messagesContainer');
            const lastMessage = container.lastElementChild;

            if (!lastMessage || !lastMessage.classList.contains('assistant')) {
                console.error("Could not find the last assistant message to append tool result to.");
                addSystemMessage(`工具 ${escapeHtml(toolCallResult.tool)} 执行结果: ${escapeHtml(JSON.stringify(toolCallResult.result))}`);
                return;
            }

            const messageContentDiv = lastMessage.querySelector('.message-content');
            if (!messageContentDiv) {
                console.error("Could not find message-content div in the last assistant message.");
                lastMessage.innerHTML += `<div class="tool-result"><strong>工具: ${escapeHtml(toolCallResult.tool)}</strong><br><pre>${escapeHtml(JSON.stringify(toolCallResult.result))}</pre></div>`;
                return;
            }

            const toolResultDiv = document.createElement('div');
            toolResultDiv.className = 'tool-result';
            toolResultDiv.innerHTML = `<div class="tool-name"><strong>工具: ${escapeHtml(toolCallResult.tool)}</strong></div>`;

            const resultData = toolCallResult.result;

            if (typeof resultData === 'string') {
                const contentDiv = document.createElement('div');
                contentDiv.className = 'tool-output-string';
                contentDiv.innerHTML = `<pre>${escapeHtml(resultData)}</pre>`;
                toolResultDiv.appendChild(contentDiv);
            } else if (typeof resultData === 'object' && resultData !== null) {
                if (resultData.error) {
                    const errorDisplay = document.createElement('div');
                    errorDisplay.className = 'tool-error-message';
                    errorDisplay.innerHTML = `<strong>工具错误:</strong> <pre>${escapeHtml(resultData.error)}</pre>`;
                     // Add export button for the error object itself
                    const exportErrorButton = document.createElement('button');
                    exportErrorButton.textContent = '导出错误详情 (JSON)';
                    exportErrorButton.className = 'export-button export-json-button btn btn-sm btn-outline-danger mt-1';
                    exportErrorButton.onclick = function() {
                        exportObjectToJson(resultData, `error_report_${toolCallResult.tool}_${Date.now()}.json`);
                    };
                    errorDisplay.appendChild(exportErrorButton);
                    toolResultDiv.appendChild(errorDisplay);
                } else if (resultData.image) {
                    const img = document.createElement('img');
                    img.src = resultData.image;
                    img.className = 'visualization';
                    toolResultDiv.appendChild(img);
                } else if (resultData.plot_json) {
                    const plotDivId = `plotly-${Date.now()}-${Math.random().toString(36).substring(2,7)}`;
                    const plotDiv = document.createElement('div');
                    plotDiv.id = plotDivId;
                    plotDiv.className = 'visualization plotly-plot';
                    toolResultDiv.appendChild(plotDiv);
                    try {
                        const plotSpec = JSON.parse(resultData.plot_json);
                        Plotly.newPlot(plotDivId, plotSpec.data, plotSpec.layout, {responsive: true});
                    } catch (e) {
                        console.error("Error parsing Plotly JSON:", e);
                        plotDiv.innerHTML = `<div class="tool-error-message">Plotly JSON 解析错误: <pre>${escapeHtml(e.message)}</pre></div>`;
                    }
                } else {
                    // For general structured data, renderStructuredData returns HTML string including buttons
                    toolResultDiv.innerHTML += renderStructuredData(resultData, 0, resultData);
                }
            } else if (resultData === null || resultData === undefined) {
                const nullInfo = document.createElement('div');
                nullInfo.className = 'tool-output-null';
                nullInfo.textContent = '工具没有返回任何内容。';
                toolResultDiv.appendChild(nullInfo);
            } else {
                const fallbackDiv = document.createElement('div');
                fallbackDiv.className = 'tool-output-fallback';
                fallbackDiv.innerHTML = `<pre>${escapeHtml(JSON.stringify(resultData, null, 2))}</pre>`;
                toolResultDiv.appendChild(fallbackDiv);
            }

            messageContentDiv.appendChild(toolResultDiv);
            scrollToBottom();
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

                if (!response.ok) {
                    if (response.status === 401 || response.status === 403) {
                        addSystemMessage('您的会话已过期，请刷新页面或点击<a href="#" onclick="reinitializeSession()">这里</a>重新开始。');
                        disableAllInputs(); // Disable inputs as session is gone
                        throw new Error('Session expired');
                    }
                    const errorData = await response.json().catch(() => ({ detail: '导出聊天记录时发生未知服务端错误。' }));
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                }

                const parsedResponse = await response.json();

                if (parsedResponse.status === 'success' && parsedResponse.data) {
                    const blob = new Blob([JSON.stringify(parsedResponse.data, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `chat_history_${new Date().toISOString()}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                } else {
                    showError(parsedResponse.message || parsedResponse.detail || '导出失败');
                }
            } catch (error) {
                 if (error.message !== 'Session expired') {
                    showError('导出失败: ' + error.message);
                }
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
