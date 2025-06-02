"""Web interface for Data Analysis Assistant"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Data Analysis Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, EXPORT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Session management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "history": [],
            "datasets": []
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)
    
    def add_to_history(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                **message,
                "timestamp": datetime.now().isoformat()
            })

session_manager = SessionManager()

# Request/Response models
class AnalysisRequest(BaseModel):
    session_id: str
    tool: str
    parameters: Dict[str, Any]

class ChatMessage(BaseModel):
    session_id: str
    message: str

# MCP Client connection
mcp_client = None
mcp_session = None

async def init_mcp_client():
    """Initialize MCP client connection"""
    global mcp_client, mcp_session
    
    try:
        server_script = str(BASE_DIR / "src" / "server.py")
        
        async with stdio_client(["python", server_script]) as (read, write):
            async with ClientSession(read, write) as session:
                mcp_session = session
                await session.initialize()
                
                # Get available tools
                tools = await session.list_tools()
                logger.info(f"Connected to MCP server. Available tools: {[t.name for t in tools]}")
                
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        raise

# Routes
@app.on_event("startup")
async def startup_event():
    """Start MCP client on app startup"""
    asyncio.create_task(init_mcp_client())
    
    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface"""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        # Create a basic index.html if it doesn't exist
        await create_default_index()
    
    async with aiofiles.open(index_path, mode='r') as f:
        content = await f.read()
    return HTMLResponse(content=content)

@app.post("/api/session")
async def create_session():
    """Create a new analysis session"""
    session_id = session_manager.create_session()
    return {"session_id": session_id}

@app.post("/api/upload")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload a file for analysis"""
    try:
        # Validate session
        if not session_manager.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Call MCP tool to process file
        if mcp_session:
            result = await mcp_session.call_tool(
                "upload_file",
                arguments={
                    "file_path": str(file_path),
                    "file_name": file.filename
                }
            )
            
            # Add to session history
            session_manager.add_to_history(session_id, {
                "type": "file_upload",
                "filename": file.filename,
                "result": result
            })
            
            return {"success": True, "result": result}
        else:
            raise HTTPException(status_code=503, detail="MCP server not connected")
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """Execute an analysis tool"""
    try:
        # Validate session
        if not session_manager.get_session(request.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Call MCP tool
        if mcp_session:
            result = await mcp_session.call_tool(
                request.tool,
                arguments=request.parameters
            )
            
            # Add to session history
            session_manager.add_to_history(request.session_id, {
                "type": "analysis",
                "tool": request.tool,
                "parameters": request.parameters,
                "result": result
            })
            
            return {"success": True, "result": result}
        else:
            raise HTTPException(status_code=503, detail="MCP server not connected")
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Get session history"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"history": session["history"]}

@app.get("/api/tools")
async def list_tools():
    """List available MCP tools"""
    if mcp_session:
        tools = await mcp_session.list_tools()
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                for tool in tools
            ]
        }
    else:
        raise HTTPException(status_code=503, detail="MCP server not connected")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process based on message type
            if data["type"] == "analyze":
                if mcp_session:
                    result = await mcp_session.call_tool(
                        data["tool"],
                        arguments=data.get("parameters", {})
                    )
                    
                    await websocket.send_json({
                        "type": "result",
                        "tool": data["tool"],
                        "result": result
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "MCP server not connected"
                    })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def create_default_index():
    """Create a default index.html file"""
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析助手</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .chat-container {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .visualization {
            max-width: 100%;
            margin: 10px 0;
        }
        .tool-button {
            margin: 5px;
        }
        .sidebar {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
    </style>
</head>
<body>
    <div class="container-fluid mt-4">
        <h1 class="text-center mb-4">
            <i class="fas fa-chart-line"></i> 数据分析助手
        </h1>
        
        <div class="row">
            <!-- Left Sidebar -->
            <div class="col-md-3">
                <div class="sidebar">
                    <h5>数据集</h5>
                    <div id="datasetList" class="mb-3">
                        <p class="text-muted">尚未上传数据集</p>
                    </div>
                    
                    <div class="mb-3">
                        <label for="fileUpload" class="form-label">上传文件</label>
                        <input type="file" class="form-control" id="fileUpload" accept=".csv,.xlsx,.xls">
                    </div>
                    
                    <hr>
                    
                    <h5>快速操作</h5>
                    <div id="quickActions">
                        <button class="btn btn-sm btn-outline-primary tool-button" onclick="callTool('data_overview')">
                            <i class="fas fa-info-circle"></i> 数据概览
                        </button>
                        <button class="btn btn-sm btn-outline-primary tool-button" onclick="callTool('descriptive_statistics')">
                            <i class="fas fa-calculator"></i> 描述统计
                        </button>
                        <button class="btn btn-sm btn-outline-primary tool-button" onclick="callTool('correlation_analysis')">
                            <i class="fas fa-project-diagram"></i> 相关性分析
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Main Chat Area -->
            <div class="col-md-6">
                <div class="chat-container" id="chatContainer">
                    <div class="message assistant-message">
                        <strong>助手:</strong> 欢迎使用数据分析助手！请上传CSV或Excel文件开始分析。
                    </div>
                </div>
                
                <div class="input-group mt-3">
                    <input type="text" class="form-control" id="userInput" placeholder="输入您的问题或命令...">
                    <button class="btn btn-primary" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i> 发送
                    </button>
                </div>
            </div>
            
            <!-- Right Sidebar -->
            <div class="col-md-3">
                <div class="sidebar">
                    <h5>可视化选项</h5>
                    <div id="visualizationOptions">
                        <button class="btn btn-sm btn-outline-success tool-button" onclick="showVisualizationModal('histogram')">
                            <i class="fas fa-chart-bar"></i> 直方图
                        </button>
                        <button class="btn btn-sm btn-outline-success tool-button" onclick="showVisualizationModal('scatter')">
                            <i class="fas fa-chart-scatter"></i> 散点图
                        </button>
                        <button class="btn btn-sm btn-outline-success tool-button" onclick="showVisualizationModal('box')">
                            <i class="fas fa-box"></i> 箱线图
                        </button>
                        <button class="btn btn-sm btn-outline-success tool-button" onclick="showVisualizationModal('heatmap')">
                            <i class="fas fa-th"></i> 热力图
                        </button>
                    </div>
                    
                    <hr>
                    
                    <h5>导出选项</h5>
                    <button class="btn btn-sm btn-outline-secondary" onclick="exportHistory()">
                        <i class="fas fa-download"></i> 导出分析历史
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Visualization Modal -->
    <div class="modal fade" id="visualizationModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="visualizationModalLabel">创建可视化</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="visualizationModalBody">
                    <!-- Dynamic content based on visualization type -->
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-primary" onclick="createVisualization()">创建</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        let sessionId = null;
        let currentDataset = null;
        let columns = [];
        
        // Initialize session
        async function initSession() {
            try {
                const response = await fetch('/api/session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                console.log('Session initialized:', sessionId);
            } catch (error) {
                console.error('Failed to initialize session:', error);
            }
        }
        
        // File upload handler
        document.getElementById('fileUpload').addEventListener('change', async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);
            
            try {
                addMessage('用户', `上传文件: ${file.name}`, 'user');
                
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentDataset = result.result.dataset_name;
                    columns = result.result.columns;
                    updateDatasetList(result.result);
                    addMessage('助手', `文件上传成功！数据集包含 ${result.result.shape[0]} 行和 ${result.result.shape[1]} 列。`, 'assistant');
                } else {
                    addMessage('助手', `上传失败: ${result.error}`, 'assistant');
                }
            } catch (error) {
                addMessage('助手', `上传错误: ${error.message}`, 'assistant');
            }
        });
        
        // Call analysis tool
        async function callTool(toolName, parameters = {}) {
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        tool: toolName,
                        parameters: parameters
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResult(toolName, result.result);
                } else {
                    addMessage('助手', `分析失败: ${result.error}`, 'assistant');
                }
            } catch (error) {
                addMessage('助手', `分析错误: ${error.message}`, 'assistant');
            }
        }
        
        // Display analysis results
        function displayResult(toolName, result) {
            if (result.error) {
                addMessage('助手', `错误: ${result.error}`, 'assistant');
                return;
            }
            
            let message = '';
            
            switch (toolName) {
                case 'data_overview':
                    message = `数据概览:\\n- 行数: ${result.shape.rows}\\n- 列数: ${result.shape.columns}\\n- 缺失值总数: ${result.missing_values.total}\\n- 重复行: ${result.duplicated_rows}`;
                    break;
                
                case 'descriptive_statistics':
                    message = '描述性统计已完成。';
                    // Could add a table here
                    break;
                
                case 'correlation_analysis':
                    message = `相关性分析完成。发现 ${result.significant_correlations.length} 个显著相关。`;
                    break;
                
                case 'plot_histogram':
                case 'plot_scatter':
                case 'plot_box':
                case 'plot_correlation_heatmap':
                    if (result.image) {
                        const img = document.createElement('img');
                        img.src = result.image;
                        img.className = 'visualization';
                        addMessageWithElement('助手', img, 'assistant');
                        return;
                    }
                    break;
            }
            
            if (message) {
                addMessage('助手', message, 'assistant');
            }
        }
        
        // Add message to chat
        function addMessage(sender, text, type) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function addMessageWithElement(sender, element, type) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> `;
            messageDiv.appendChild(element);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Update dataset list
        function updateDatasetList(datasetInfo) {
            const datasetList = document.getElementById('datasetList');
            datasetList.innerHTML = `
                <div class="alert alert-info">
                    <strong>${datasetInfo.dataset_name}</strong><br>
                    形状: ${datasetInfo.shape[0]} × ${datasetInfo.shape[1]}<br>
                    内存: ${datasetInfo.memory_usage.toFixed(2)} MB
                </div>
            `;
        }
        
        // Show visualization modal
        function showVisualizationModal(type) {
            if (!currentDataset) {
                alert('请先上传数据集！');
                return;
            }
            
            const modal = new bootstrap.Modal(document.getElementById('visualizationModal'));
            const modalBody = document.getElementById('visualizationModalBody');
            
            // Set modal content based on type
            switch (type) {
                case 'histogram':
                    modalBody.innerHTML = `
                        <div class="mb-3">
                            <label class="form-label">选择列</label>
                            <select class="form-select" id="vizColumn">
                                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">箱数</label>
                            <input type="number" class="form-control" id="vizBins" value="30">
                        </div>
                    `;
                    modalBody.dataset.vizType = 'histogram';
                    break;
                
                case 'scatter':
                    modalBody.innerHTML = `
                        <div class="mb-3">
                            <label class="form-label">X轴</label>
                            <select class="form-select" id="vizXColumn">
                                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Y轴</label>
                            <select class="form-select" id="vizYColumn">
                                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                            </select>
                        </div>
                    `;
                    modalBody.dataset.vizType = 'scatter';
                    break;
                
                // Add more visualization types...
            }
            
            modal.show();
        }
        
        // Create visualization
        async function createVisualization() {
            const modalBody = document.getElementById('visualizationModalBody');
            const vizType = modalBody.dataset.vizType;
            let parameters = {};
            
            switch (vizType) {
                case 'histogram':
                    parameters = {
                        column: document.getElementById('vizColumn').value,
                        bins: parseInt(document.getElementById('vizBins').value)
                    };
                    await callTool('plot_histogram', parameters);
                    break;
                
                case 'scatter':
                    parameters = {
                        x_column: document.getElementById('vizXColumn').value,
                        y_column: document.getElementById('vizYColumn').value
                    };
                    await callTool('plot_scatter', parameters);
                    break;
            }
            
            // Close modal
            bootstrap.Modal.getInstance(document.getElementById('visualizationModal')).hide();
        }
        
        // Send message
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage('用户', message, 'user');
            input.value = '';
            
            // Process user message (could add NLP here)
            processUserMessage(message);
        }
        
        // Process user message
        function processUserMessage(message) {
            // Simple keyword matching for now
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('概览') || lowerMessage.includes('overview')) {
                callTool('data_overview');
            } else if (lowerMessage.includes('统计') || lowerMessage.includes('statistics')) {
                callTool('descriptive_statistics');
            } else if (lowerMessage.includes('相关') || lowerMessage.includes('correlation')) {
                callTool('correlation_analysis');
            } else {
                addMessage('助手', '抱歉，我不理解您的请求。请尝试使用左侧的快速操作按钮。', 'assistant');
            }
        }
        
        // Export history
        async function exportHistory() {
            try {
                const response = await fetch(`/api/history/${sessionId}`);
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `analysis_history_${new Date().toISOString()}.json`;
                a.click();
            } catch (error) {
                alert('导出失败: ' + error.message);
            }
        }
        
        // Initialize on load
        window.addEventListener('load', () => {
            initSession();
            
            // Enter key handler
            document.getElementById('userInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        });
    </script>
</body>
</html>"""
    
    TEMPLATES_DIR.mkdir(exist_ok=True)
    async with aiofiles.open(TEMPLATES_DIR / "index.html", mode='w', encoding='utf-8') as f:
        await f.write(html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)