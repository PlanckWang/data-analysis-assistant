"""Chatbot-style Web Interface for Data Analysis Assistant"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel
from dotenv import load_dotenv

from .llm_providers import LLMProviderFactory, Message

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Data Analysis Chatbot", version="1.0.0")

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
for dir_path in [UPLOAD_DIR, EXPORT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Session management
class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.context = {
            "datasets": {},
            "current_dataset": None,
            "analysis_history": []
        }
        self.llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    
    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
    
    def get_conversation_context(self) -> str:
        """Get context about current data and analysis"""
        context_parts = []
        
        if self.context["current_dataset"]:
            context_parts.append(f"当前数据集: {self.context['current_dataset']}")
        
        if self.context["datasets"]:
            datasets_info = []
            for name, info in self.context["datasets"].items():
                datasets_info.append(f"- {name}: {info.get('shape', 'N/A')}")
            context_parts.append("已加载的数据集:\n" + "\n".join(datasets_info))
        
        if self.context["analysis_history"]:
            recent_analyses = self.context["analysis_history"][-5:]
            analyses_info = [f"- {a['type']} at {a['timestamp']}" for a in recent_analyses]
            context_parts.append("最近的分析:\n" + "\n".join(analyses_info))
        
        return "\n\n".join(context_parts) if context_parts else "尚未加载任何数据集。"

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ChatSession(session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self.sessions.get(session_id)

session_manager = SessionManager()

# Request models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: Optional[str] = None

class ProviderSwitchRequest(BaseModel):
    session_id: str
    provider: str

# MCP Client management
class MCPManager:
    def __init__(self):
        self.session = None
        self.tools = {}
        self.lock = asyncio.Lock()
    
    async def connect(self):
        """Connect to MCP server"""
        async with self.lock:
            if self.session is not None:
                return
            
            try:
                # Create server parameters
                server_params = StdioServerParameters(
                    command="python -m src.server",
                    cwd=str(BASE_DIR)
                )
                
                read, write = await stdio_client(server_params).__aenter__()
                self.session = ClientSession(read, write)
                await self.session.__aenter__()
                await self.session.initialize()
                
                # Get available tools
                tools_response = await self.session.list_tools()
                for tool in tools_response.tools:
                    self.tools[tool.name] = tool
                
                logger.info(f"Connected to MCP server. Available tools: {list(self.tools.keys())}")
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                self.session = None
                raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool"""
        if not self.session:
            await self.connect()
        
        if not self.session:
            raise Exception("MCP server not connected")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result
    
    def get_tools_description(self) -> str:
        """Get a description of available tools for the LLM"""
        if not self.tools:
            return "No tools available"
        
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"- {name}: {tool.description}"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                params = tool.inputSchema.get('properties', {})
                if params:
                    param_list = [f"{k} ({v.get('type', 'any')})" for k, v in params.items()]
                    desc += f"\n  Parameters: {', '.join(param_list)}"
            descriptions.append(desc)
        
        return "Available tools:\n" + "\n".join(descriptions)

mcp_manager = MCPManager()

# System prompt for the LLM
SYSTEM_PROMPT = """你是一个专业的数据分析助手。你可以帮助用户：
1. 上传和管理CSV、Excel数据文件
2. 进行数据探索和统计分析
3. 创建各种数据可视化图表
4. 回答关于数据的问题

当用户需要进行数据分析时，你可以调用以下工具：
{tools_description}

请根据用户的需求，智能地选择合适的工具进行分析。回复时要简洁明了，使用中文。

当需要调用工具时，请使用以下格式：
<tool_call>
{
    "tool": "tool_name",
    "arguments": {
        "param1": "value1",
        "param2": "value2"
    }
}
</tool_call>

记住：
- 在分析数据前，确保已经上传了数据文件
- 根据用户的问题选择合适的分析方法
- 解释分析结果时要通俗易懂
- 如果用户的请求不明确，请礼貌地询问更多细节"""

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize MCP connection on startup"""
    try:
        await mcp_manager.connect()
    except Exception as e:
        logger.error(f"Failed to initialize MCP connection: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chatbot interface"""
    index_path = TEMPLATES_DIR / "chatbot.html"
    if not index_path.exists():
        await create_chatbot_interface()
    
    async with aiofiles.open(index_path, mode='r', encoding='utf-8') as f:
        content = await f.read()
    return HTMLResponse(content=content)

@app.post("/api/session")
async def create_session():
    """Create a new chat session"""
    session_id = session_manager.create_session()
    available_providers = LLMProviderFactory.get_available_providers()
    return {
        "session_id": session_id,
        "available_providers": available_providers,
        "default_provider": os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    session_id_query: str = Query(None, alias="session_id")
):
    """Upload a file for analysis"""
    # 使用查询参数或表单参数中的session_id
    session_id = session_id or session_id_query
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Call MCP tool to process file
        result = await mcp_manager.call_tool(
            "upload_file",
            {
                "file_path": str(file_path),
                "file_name": file.filename
            }
        )
        
        # Update session context
        if "error" not in result:
            dataset_name = result.get("dataset_name", file.filename)
            session.context["datasets"][dataset_name] = result
            session.context["current_dataset"] = dataset_name
        
        return {"success": True, "result": result}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat messages"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add user message to history
    session.add_message("user", request.message)
    
    # Switch provider if requested
    if request.provider:
        session.llm_provider = request.provider
    
    # Prepare messages for LLM
    system_message = SYSTEM_PROMPT.format(
        tools_description=mcp_manager.get_tools_description()
    )
    
    # Add context about current data
    context_message = f"当前数据状态:\n{session.get_conversation_context()}"
    
    messages = [
        Message(role="system", content=system_message),
        Message(role="system", content=context_message)
    ] + session.messages[-10:]  # Keep last 10 messages for context
    
    try:
        # Create LLM provider
        provider = LLMProviderFactory.create(session.llm_provider)
        
        # Get LLM response
        full_response = ""
        async for chunk in provider.chat(messages, stream=True):
            full_response += chunk
        
        # Check if the response contains tool calls
        tool_results = []
        if "<tool_call>" in full_response:
            tool_results = await process_tool_calls(full_response, session)
        
        # Add assistant message to history
        session.add_message("assistant", full_response)
        
        return {
            "response": full_response,
            "tool_results": tool_results,
            "provider": session.llm_provider
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def generate():
        try:
            # Add user message
            session.add_message("user", request.message)
            
            # Switch provider if requested
            if request.provider:
                session.llm_provider = request.provider
            
            # Prepare messages
            system_message = SYSTEM_PROMPT.format(
                tools_description=mcp_manager.get_tools_description()
            )
            context_message = f"当前数据状态:\n{session.get_conversation_context()}"
            
            messages = [
                Message(role="system", content=system_message),
                Message(role="system", content=context_message)
            ] + session.messages[-10:]
            
            # Create LLM provider
            provider = LLMProviderFactory.create(session.llm_provider)
            
            # Stream response
            full_response = ""
            async for chunk in provider.chat(messages, stream=True):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Process tool calls if any
            if "<tool_call>" in full_response:
                tool_results = await process_tool_calls(full_response, session)
                for result in tool_results:
                    yield f"data: {json.dumps({'type': 'tool_result', 'result': result})}\n\n"
            
            # Add to history
            session.add_message("assistant", full_response)
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/provider/switch")
async def switch_provider(request: ProviderSwitchRequest):
    """Switch LLM provider for a session"""
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    available = LLMProviderFactory.get_available_providers()
    if request.provider not in available:
        raise HTTPException(status_code=400, detail=f"Provider {request.provider} not available")
    
    session.llm_provider = request.provider
    return {"success": True, "provider": request.provider}

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages
        ],
        "context": session.context
    }

async def process_tool_calls(response: str, session: ChatSession) -> List[Dict[str, Any]]:
    """Extract and process tool calls from LLM response"""
    import re
    
    tool_results = []
    
    # Find all tool calls in the response
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            # Parse JSON
            tool_data = json.loads(match.strip())
            tool_name = tool_data.get("tool")
            arguments = tool_data.get("arguments", {})
            
            # Call the tool
            result = await mcp_manager.call_tool(tool_name, arguments)
            
            # Update session context if needed
            if tool_name == "upload_file" and "error" not in result:
                dataset_name = result.get("dataset_name")
                session.context["datasets"][dataset_name] = result
                session.context["current_dataset"] = dataset_name
            
            # Add to analysis history
            session.context["analysis_history"].append({
                "type": tool_name,
                "timestamp": datetime.now().isoformat(),
                "result_summary": str(result)[:200]
            })
            
            tool_results.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            tool_results.append({
                "tool": "error",
                "error": str(e)
            })
    
    return tool_results

async def create_chatbot_interface():
    """Create the chatbot HTML interface"""
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析聊天助手</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #007bff;
            --secondary-color: #6c757d;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            margin: 0;
            padding: 0;
        }
        
        .chat-container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 300px;
            background-color: white;
            border-right: 1px solid #dee2e6;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            background-color: white;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .message.user .message-content {
            background-color: var(--primary-color);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-content {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .message.user .message-avatar {
            background-color: var(--primary-color);
            color: white;
            order: 1;
        }
        
        .message.assistant .message-avatar {
            background-color: #f0f0f0;
            color: #666;
        }
        
        .input-area {
            background-color: white;
            border-top: 1px solid #dee2e6;
            padding: 20px;
        }
        
        .input-group-custom {
            display: flex;
            align-items: flex-end;
            gap: 10px;
        }
        
        .message-input {
            flex: 1;
            border: 1px solid #dee2e6;
            border-radius: 24px;
            padding: 10px 20px;
            resize: none;
            outline: none;
            font-size: 14px;
            line-height: 1.5;
            max-height: 120px;
            overflow-y: auto;
        }
        
        .message-input:focus {
            border-color: var(--primary-color);
        }
        
        .send-button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .send-button:hover {
            background-color: #0056b3;
        }
        
        .send-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
        }
        
        .upload-area.dragover {
            border-color: var(--primary-color);
            background-color: #e7f3ff;
        }
        
        .provider-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .provider-button {
            flex: 1;
            padding: 8px;
            border: 1px solid #dee2e6;
            background-color: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 12px;
        }
        
        .provider-button.active {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .provider-button:hover:not(.active) {
            background-color: #f8f9fa;
        }
        
        .dataset-item {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .dataset-item:hover {
            background-color: #e9ecef;
        }
        
        .dataset-item.active {
            background-color: #e7f3ff;
            border: 1px solid var(--primary-color);
        }
        
        .tool-result {
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid var(--info-color);
        }
        
        .visualization {
            max-width: 100%;
            margin-top: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 10px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #999;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% {
                opacity: 0.3;
                transform: scale(0.8);
            }
            30% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .error-message {
            color: var(--danger-color);
            font-size: 14px;
            margin-top: 5px;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 2px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <h5 class="mb-0">
                    <i class="fas fa-chart-line"></i> 数据分析助手
                </h5>
            </div>
            
            <div class="sidebar-content">
                <!-- Provider Selection -->
                <div class="mb-4">
                    <h6 class="text-muted mb-2">AI 模型选择</h6>
                    <div class="provider-selector" id="providerSelector">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
                
                <!-- File Upload -->
                <div class="mb-4">
                    <h6 class="text-muted mb-2">上传数据文件</h6>
                    <div class="upload-area" id="uploadArea">
                        <i class="fas fa-cloud-upload-alt fa-2x text-muted"></i>
                        <p class="mb-0 mt-2">点击或拖拽文件到此处</p>
                        <small class="text-muted">支持 CSV, Excel 格式</small>
                        <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
                    </div>
                </div>
                
                <!-- Datasets -->
                <div class="mb-4">
                    <h6 class="text-muted mb-2">已加载数据集</h6>
                    <div id="datasetsList">
                        <p class="text-muted small">尚未加载数据集</p>
                    </div>
                </div>
                
                <!-- Quick Actions -->
                <div>
                    <h6 class="text-muted mb-2">快速操作</h6>
                    <button class="btn btn-sm btn-outline-primary w-100 mb-2" onclick="exportChat()">
                        <i class="fas fa-download"></i> 导出对话记录
                    </button>
                    <button class="btn btn-sm btn-outline-secondary w-100" onclick="clearChat()">
                        <i class="fas fa-trash"></i> 清空对话
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Chat Header -->
            <div class="chat-header">
                <div>
                    <h6 class="mb-0">数据分析对话</h6>
                    <small class="text-muted" id="sessionInfo">会话 ID: -</small>
                </div>
                <div>
                    <span class="badge bg-success" id="connectionStatus">
                        <i class="fas fa-circle"></i> 已连接
                    </span>
                </div>
            </div>
            
            <!-- Messages Container -->
            <div class="messages-container" id="messagesContainer">
                <div class="message assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <p>您好！我是您的数据分析助手。</p>
                        <p>我可以帮助您：</p>
                        <ul>
                            <li>📊 分析CSV和Excel数据文件</li>
                            <li>📈 创建各种数据可视化图表</li>
                            <li>🔍 进行统计分析和数据探索</li>
                            <li>💡 回答关于数据的问题</li>
                        </ul>
                        <p>请上传数据文件开始分析，或直接向我提问！</p>
                    </div>
                </div>
            </div>
            
            <!-- Input Area -->
            <div class="input-area">
                <div class="input-group-custom">
                    <textarea 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="输入消息... (Shift+Enter 换行)"
                        rows="1"
                    ></textarea>
                    <button class="send-button" id="sendButton" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        let sessionId = null;
        let currentProvider = 'openai';
        let availableProviders = [];
        let isProcessing = false;
        
        // Initialize session
        async function initSession() {
            try {
                const response = await fetch('/api/session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                availableProviders = data.available_providers;
                currentProvider = data.default_provider;
                
                document.getElementById('sessionInfo').textContent = `会话 ID: ${sessionId.substring(0, 8)}...`;
                
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
                const response = await fetch('/api/provider/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, provider })
                });
                
                if (response.ok) {
                    currentProvider = provider;
                    setupProviders();
                    addSystemMessage(`已切换到 ${provider.toUpperCase()} 模型`);
                }
            } catch (error) {
                showError('切换模型失败');
            }
        }
        
        // File upload handling - moved to setupFileUpload function
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
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                hideTypingIndicator();
                
                if (result.success && result.result) {
                    if (result.result.error) {
                        addAssistantMessage(`上传失败: ${result.result.error}`);
                    } else {
                        updateDatasetsList(result.result);
                        addAssistantMessage(`文件 "${result.result.dataset_name}" 上传成功！\n\n数据集信息：\n- 行数: ${result.result.shape[0]}\n- 列数: ${result.result.shape[1]}\n- 内存占用: ${result.result.memory_usage.toFixed(2)} MB\n\n您可以开始询问关于这个数据集的问题了。`);
                    }
                } else {
                    addAssistantMessage('文件上传失败，请重试。');
                }
                
            } catch (error) {
                hideTypingIndicator();
                addAssistantMessage(`上传出错: ${error.message}`);
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
                const response = await fetch('/api/chat/stream', {
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
                const response = await fetch(`/api/history/${sessionId}`);
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `chat_history_${new Date().toISOString()}.json`;
                a.click();
                URL.revokeObjectURL(url);
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
            // Initialize session
            initSession();
            
            // Setup event listeners after DOM is loaded
            setupFileUpload();
            setupMessageInput();
            
            // Update UI
            updateSendButton();
            
            // Configure marked options
            marked.setOptions({
                breaks: true,
                gfm: true
            });
        });
    </script>
</body>
</html>"""
    
    async with aiofiles.open(TEMPLATES_DIR / "chatbot.html", mode='w', encoding='utf-8') as f:
        await f.write(html_content)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", 8000))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)