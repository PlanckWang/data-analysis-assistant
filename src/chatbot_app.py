"""
Main FastAPI application for the Data Analysis Chatbot.

This module sets up the FastAPI application, including:
- API routes for session management, file uploads, chat interactions, provider switching, and history.
- WebSocket endpoint for real-time chat.
- Custom exception handlers for consistent error responses.
- Structlog integration for structured logging.
- Request ID middleware for tracing.
- MCP (Machine Control Protocol) client management for backend tool execution.
- Redis-based session management.
- Pydantic models for request/response validation and serialization.
- Static file serving for the chatbot interface.
"""

import asyncio
import json
import logging # Still needed for some structlog stdlib interactions
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List as PyList, Optional, TypeVar, Generic, Literal, AsyncGenerator
from dataclasses import asdict

import aiofiles
import redis.asyncio as aioredis
import structlog # For logger initialization
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel, Field
from dotenv import load_dotenv # Used by config if .env file is present, but not directly here
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseCallNext
from starlette.requests import Request as StarletteRequest


from .llm_providers import LLMProviderFactory, Message
from .config import settings
from . import exceptions

# --- Structlog Configuration ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

json_formatter = structlog.stdlib.ProcessorFormatter(
    processor=structlog.processors.JSONRenderer(),
    foreign_pre_chain=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.format_exc_info,
    ]
)

handler = logging.StreamHandler()
handler.setFormatter(json_formatter)
root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO) # Default log level

logger = structlog.get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analysis Chatbot",
    version="1.0.0",
    description="A chatbot interface for data analysis, powered by LLMs and MCP."
)

# --- Request ID Middleware ---
class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request and log context."""
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCallNext):
        """
        Clears context variables, binds a new request_id, and processes the request.
        The request_id is also added to the response headers.
        """
        structlog.contextvars.clear_contextvars()
        request_id: str = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestContextMiddleware)

# Global Redis client
redis_client: Optional[aioredis.Redis] = None


# Pydantic Models for API Responses
T = TypeVar('T')

class SuccessResponse(BaseModel, Generic[T]):
    """Generic success response model for API endpoints."""
    status: Literal["success"] = Field(default="success", description="Indicates the successful status of the response.")
    data: Optional[T] = Field(default=None, description="The actual data payload of the response, if any.")
    message: Optional[str] = Field(default=None, description="An optional message providing further details about the success.")

class SessionInitData(BaseModel):
    """Data model for session initialization response."""
    session_id: str = Field(description="The unique identifier for the newly created session.")
    available_providers: PyList[str] = Field(description="A list of available LLM provider names.")
    default_provider: str = Field(description="The default LLM provider configured for new sessions.")

class UploadResultData(BaseModel):
    """Data model for file upload response."""
    result: PyDict[str, Any] = Field(description="A dictionary containing the result from the file upload processing, typically including file metadata and initial analysis.")

class ChatCompletionData(BaseModel):
    """Data model for chat completion response (non-streaming)."""
    response: str = Field(description="The complete response string from the LLM.")
    tool_results: PyList[PyDict[str, Any]] = Field(default_factory=list, description="A list of results from any tools called by the LLM.")
    provider: str = Field(description="The LLM provider that generated the response.")

class ProviderSwitchData(BaseModel):
    """Data model for provider switch response."""
    provider: str = Field(description="The name of the newly active LLM provider for the session.")

class ChatHistoryData(BaseModel):
    """Data model for chat history response."""
    messages: PyList[PyDict[str,str]] = Field(description="A list of chat messages, each with 'role' and 'content'.")
    context: PyDict[str, Any] = Field(description="The current context associated with the chat session (e.g., loaded datasets).")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
class ChatSession:
    """
    Represents a user's chat session, including messages, context, and LLM provider settings.
    """
    def __init__(self, session_id: str) -> None:
        """
        Initializes a new ChatSession.

        Args:
            session_id: The unique identifier for this session.
        """
        self.session_id: str = session_id
        self.created_at: datetime = datetime.now()
        self.messages: PyList[Message] = []
        self.context: Dict[str, Any] = {
            "datasets": {},          # Stores information about loaded datasets
            "current_dataset": None, # Name of the currently active dataset
            "analysis_history": []   # History of analysis operations
        }
        self.llm_provider: str = settings.DEFAULT_LLM_PROVIDER

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the ChatSession instance to a dictionary for storage (e.g., in Redis).

        Returns:
            A dictionary representation of the session.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [asdict(msg) for msg in self.messages],
            "context": self.context,
            "llm_provider": self.llm_provider,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """
        Deserializes a dictionary into a ChatSession instance.

        Args:
            data: A dictionary containing session data, typically from Redis.

        Returns:
            A ChatSession instance populated with data from the dictionary.
        """
        session = cls(session_id=data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.messages = [Message(role=msg_data["role"], content=msg_data["content"]) for msg_data in data["messages"]]
        session.context = data["context"]
        session.llm_provider = data["llm_provider"]
        return session
    
    def add_message(self, role: str, content: str) -> None:
        """
        Adds a new message to the session's message history.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The text content of the message.
        """
        self.messages.append(Message(role=role, content=content))
    
    def get_conversation_context(self) -> str:
        """
        Generates a string summarizing the current conversation context for the LLM.
        This includes information about loaded datasets and recent analyses.

        Returns:
            A string providing context about the current data state.
        """
        context_parts: PyList[str] = []
        
        if self.context.get("current_dataset"):
            context_parts.append(f"当前数据集: {self.context['current_dataset']}")
        
        if self.context["datasets"]:
            datasets_info = []
            for name, info in self.context["datasets"].items():
                datasets_info.append(f"- {name}: {info.get('shape', 'N/A')}")
            context_parts.append("已加载的数据集:\n" + "\n".join(datasets_info)) # type: ignore
        
        if self.context.get("analysis_history"):
            recent_analyses: PyList[Dict[str, Any]] = self.context["analysis_history"][-5:]
            analyses_info: PyList[str] = [f"- {a.get('type', 'N/A')} at {a.get('timestamp', 'N/A')}" for a in recent_analyses]
            context_parts.append("最近的分析:\n" + "\n".join(analyses_info))
        
        return "\n\n".join(context_parts) if context_parts else "尚未加载任何数据集。"

class SessionManager:
    """
    Manages chat sessions using a Redis backend.

    Responsibilities include creating, retrieving, and updating session data.
    """
    def __init__(self, redis_client: aioredis.Redis) -> None:
        """
        Initializes the SessionManager with a Redis client instance.

        Args:
            redis_client: An initialized `redis.asyncio.Redis` client.
        """
        self.redis: aioredis.Redis = redis_client
    
    async def create_session(self) -> str:
        """
        Creates a new chat session, stores it in Redis, and returns the session ID.

        Returns:
            The unique identifier (UUID string) for the newly created session.
        """
        session_id: str = str(uuid.uuid4())
        session = ChatSession(session_id)
        session_data_json: str = json.dumps(session.to_dict())
        await self.redis.set(f"session:{session_id}", session_data_json, ex=settings.SESSION_EXPIRE_SECONDS)
        logger.info("session_created", session_id=session_id)
        return session_id
    
    async def get_session(self, session_id: str) -> ChatSession: # Changed Optional[ChatSession] to ChatSession
        """
        Retrieves a chat session from Redis by its ID.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession instance.

        Raises:
            exceptions.SessionNotFoundError: If the session with the given ID is not found.
        """
        session_data_json: Optional[bytes] = await self.redis.get(f"session:{session_id}")
        if not session_data_json:
            # Logger is in the exception handler now for SessionNotFoundError
            raise exceptions.SessionNotFoundError(session_id)
        session_data: Dict[str, Any] = json.loads(session_data_json)
        return ChatSession.from_dict(session_data)

    async def update_session(self, session: ChatSession) -> None:
        """
        Updates an existing chat session in Redis.

        Args:
            session: The ChatSession instance to update. The session's data will be
                     serialized and stored, refreshing its expiration time.
        """
        session_data_json: str = json.dumps(session.to_dict())
        await self.redis.set(f"session:{session.session_id}", session_data_json, ex=settings.SESSION_EXPIRE_SECONDS)
        logger.debug("session_updated", session_id=session.session_id)

session_manager: Optional[SessionManager] = None # Initialized in startup_event

# Request models
class ChatRequest(BaseModel):
    """Request model for chat messages."""
    session_id: str = Field(description="The ID of the current chat session.")
    message: str = Field(description="The user's message content.")
    provider: Optional[str] = Field(default=None, description="Optional LLM provider name to switch to for this request.")

class ProviderSwitchRequest(BaseModel):
    """Request model for switching LLM providers."""
    session_id: str = Field(description="The ID of the current chat session.")
    provider: str = Field(description="The name of the LLM provider to switch to.")

# MCP Client management
class MCPManager:
    """
    Manages the connection and interaction with the MCP (Machine Control Protocol) server.
    Handles connecting to the server, calling tools, and managing the lifecycle
    of the MCP client session and underlying process.
    """
    def __init__(self) -> None:
        """Initializes the MCPManager."""
        self.session_cm = None  # ClientSession context manager
        self.mcp_process_cm = None  # stdio_client context manager
        self.tools = {}
        self.lock = asyncio.Lock() # Ensures thread-safe operations on connection state
    
    async def connect(self) -> None:
        """
        Establishes a connection to the MCP server if not already connected.
        Initializes the MCP client session and retrieves available tools.

        Raises:
            exceptions.MCPConnectionError: If connection to the MCP server fails.
        """
        async with self.lock:
            if self.session_cm is not None:
                logger.debug("mcp_already_connected")
                return
            
            logger.info("mcp_connecting", command=settings.MCP_SERVER_COMMAND, cwd=str(settings.BASE_DIR))
            try:
                # Create server parameters
                server_params = StdioServerParameters(
                    command=settings.MCP_SERVER_COMMAND,
                    cwd=str(settings.BASE_DIR)
                )
                
                self.mcp_process_cm = stdio_client(server_params)
                read, write = await self.mcp_process_cm.__aenter__()

                self.session_cm = ClientSession(read, write)
                await self.session_cm.__aenter__()
                await self.session_cm.initialize()
                
                # Get available tools
                tools_response = await self.session_cm.list_tools() # type: ignore # session_cm is ClientSession
                self.tools = {tool.name: tool for tool in tools_response.tools}
                
                logger.info("mcp_connected_successfully", available_tools=list(self.tools.keys()))
                
            except Exception as e:
                logger.error("mcp_connection_attempt_failed", error=str(e), exc_info=True)
                # Clean up resources
                if self.session_cm:
                    try:
                        await self.session_cm.__aexit__(None, None, None)
                    except Exception as e_exit:
                        logger.error(f"Error during session_cm.__aexit__ in connect error handler: {e_exit}")
                    self.session_cm = None
                if self.mcp_process_cm:
                    try:
                        await self.mcp_process_cm.__aexit__(None, None, None)
                    except Exception as e_exit:
                        logger.error(f"Error during mcp_process_cm.__aexit__ in connect error handler: {e_exit}")
                    self.mcp_process_cm = None
                self.tools = {}
                # Raise a more specific error
                raise exceptions.MCPConnectionError(f"Failed to connect to MCP server: {str(e)}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Calls a specific tool on the MCP server with the given arguments.
        Establishes connection if one does not exist.

        Args:
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The result from the MCP tool execution.

        Raises:
            exceptions.MCPConnectionError: If connection to MCP server is not established.
            exceptions.ToolExecutionError: If an error occurs during tool execution.
        """
        if not self.session_cm:
            try:
                await self.connect()
            except exceptions.MCPConnectionError as e:
                logger.error("mcp_connection_failed_before_tool_call", tool_name=tool_name, error=str(e))
                raise # Re-raise if connect itself failed with MCPConnectionError
            except Exception as e: # Catch other unexpected errors during connect
                logger.error("mcp_unexpected_connection_error_before_tool_call", tool_name=tool_name, error=str(e), exc_info=True)
                raise exceptions.MCPConnectionError(f"Unexpected connection attempt error before calling tool '{tool_name}': {str(e)}")

        if not self.session_cm: # Re-check after connect attempt
            logger.error("mcp_still_not_connected_before_tool_call", tool_name=tool_name)
            raise exceptions.MCPConnectionError(f"MCP server not connected after connect attempt, cannot call tool '{tool_name}'.")
        
        try:
            logger.info("mcp_calling_tool", tool_name=tool_name, arguments=arguments)
            result: Any = await self.session_cm.call_tool(tool_name, arguments) # type: ignore
            logger.info("mcp_tool_call_successful", tool_name=tool_name)
            return result
        except Exception as e: # Catch errors during the actual tool call
            logger.error("mcp_tool_execution_error", tool_name=tool_name, arguments=arguments, error=str(e), exc_info=True)
            raise exceptions.ToolExecutionError(tool_name, str(e))

    async def close(self) -> None:
        """
        Gracefully closes the MCP client session and the underlying stdio process.
        """
        async with self.lock:
            logger.info("mcp_closing_connection")
            if self.session_cm:
                try:
                    await self.session_cm.__aexit__(None, None, None) # type: ignore
                    logger.info("mcp_client_session_closed")
                except Exception as e:
                    logger.error("mcp_client_session_close_error", error=str(e), exc_info=True)
                finally:
                    self.session_cm = None

            if self.mcp_process_cm:
                try:
                    await self.mcp_process_cm.__aexit__(None, None, None)
                    logger.info("MCP stdio_client process closed.")
                except Exception as e:
                    logger.error(f"Error closing MCP stdio_client process: {e}")
                finally:
                    self.mcp_process_cm = None # type: ignore

            self.tools = {}
            logger.info("mcp_connection_closed_and_tools_cleared")

    def get_tools_description(self) -> str:
        """
        Generates a string describing all available MCP tools, for use in LLM prompts.

        Returns:
            A formatted string listing available tools and their parameters.
            Returns "No tools available" if the tools list is empty.
        """
        if not self.tools:
            return "No tools available." # Added a period for consistency
        
        descriptions: PyList[str] = []
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

# Routes
@app.on_event("startup")
async def startup_event() -> None:
    """
    FastAPI startup event handler.
    Initializes the Redis client, session manager, and MCP manager connection.
    Logs success or failure for critical service connections.
    """
    global redis_client, session_manager
    try:
        logger.info("application_startup_initiated", service="redis")
        redis_client = aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
            encoding="utf-8", decode_responses=True # Added for easier string handling
        )
        await redis_client.ping() # type: ignore # Ping is available
        logger.info("redis_connection_successful")
        session_manager = SessionManager(redis_client)
        logger.info("session_manager_initialized")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e), exc_info=True)
        redis_client = None
        session_manager = None
        # Consider if app should hard fail to start if Redis is unavailable.
        # For now, it will continue, but session-dependent routes will fail.

    if not redis_client or not session_manager:
        logger.critical("redis_or_session_manager_not_initialized_aborting_startup")
        raise RuntimeError("Failed to connect to Redis and initialize session manager. Application cannot start.")

    try:
        logger.info("application_startup_initiated", service="mcp_manager")
        await mcp_manager.connect()
        logger.info("mcp_manager_connection_attempted") # connect() logs success/failure internally
    except Exception as e: # Catch if connect() itself throws an unhandled error
        logger.error("mcp_manager_initial_connection_failed", error=str(e), exc_info=True)

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    FastAPI shutdown event handler.
    Gracefully closes connections to the MCP server and Redis.
    """
    global redis_client
    logger.info("application_shutdown_initiated")
    await mcp_manager.close()
    if redis_client:
        logger.info("redis_closing_connection")
        await redis_client.close()
        logger.info("redis_connection_closed")

# Exception Handlers
@app.exception_handler(exceptions.SessionNotFoundError)
async def session_not_found_exception_handler(request: Request, exc: exceptions.SessionNotFoundError) -> JSONResponse:
    """Handles SessionNotFoundError by returning a 404 response."""
    logger.warn("session_not_found", session_id=exc.session_id, path=str(request.url.path), message=exc.message)
    return JSONResponse(
        status_code=404,
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(exceptions.MCPConnectionError)
async def mcp_connection_exception_handler(request: Request, exc: exceptions.MCPConnectionError) -> JSONResponse:
    """Handles MCPConnectionError by returning a 503 Service Unavailable response."""
    logger.error("mcp_connection_error", path=str(request.url.path), message=exc.message, exc_info=True)
    return JSONResponse(
        status_code=503,
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(exceptions.LLMProviderError)
async def llm_provider_exception_handler(request: Request, exc: exceptions.LLMProviderError) -> JSONResponse:
    """Handles LLMProviderError by returning a 502 Bad Gateway response."""
    logger.error("llm_provider_error", provider_name=exc.provider_name, path=str(request.url.path), message=exc.message, exc_info=True)
    return JSONResponse(
        status_code=502,
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(exceptions.ToolExecutionError)
async def tool_execution_exception_handler(request: Request, exc: exceptions.ToolExecutionError) -> JSONResponse:
    """Handles ToolExecutionError by returning a 500 response."""
    logger.error("tool_execution_error", tool_name=exc.tool_name, original_error=exc.original_error, path=str(request.url.path), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(exceptions.FileUploadError)
async def file_upload_exception_handler(request: Request, exc: exceptions.FileUploadError) -> JSONResponse:
    """Handles FileUploadError and its subclasses, returning appropriate HTTP status codes."""
    logger.warn("file_upload_error_handler", error_type=type(exc).__name__, path=str(request.url.path), message=exc.message)
    status_code: int = 400 # Default for base FileUploadError
    if isinstance(exc, exceptions.FileSizeExceededError):
        status_code = 413 # Payload Too Large
    elif isinstance(exc, exceptions.InvalidFileTypeError):
        status_code = 415 # Unsupported Media Type
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(exceptions.DataAnalysisException)
async def data_analysis_exception_handler(request: Request, exc: exceptions.DataAnalysisException) -> JSONResponse:
    """Generic handler for DataAnalysisException if not caught by more specific ones."""
    logger.error("data_analysis_exception_unhandled", path=str(request.url.path), message=exc.message, exc_info=True)
    return JSONResponse(
        status_code=500, # Or 400 if it's more often client-side related
        content={"detail": exc.message, "type": type(exc).__name__},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """FastAPI's default HTTPException handler re-raised for explicitness or custom standard."""
    # This ensures that HTTPExceptions we raise directly are also logged if desired,
    # or formatted consistently if FastAPI's default is not preferred.
    # For now, just log it, FastAPI will handle the response itself.
    logger.warn("http_exception_occurred", status_code=exc.status_code, detail=exc.detail, path=str(request.url.path))
    # Re-raise to use FastAPI's default handling for HTTPExceptions
    # return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    raise exc

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches any unhandled exceptions and returns a generic 500 error."""
    logger.error("unhandled_exception", path=str(request.url.path), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "type": type(exc).__name__},
    )

@app.get("/") # Not an API endpoint, so no /v1 or SuccessResponse
async def root():
    """Serve the chatbot interface"""
    return FileResponse(settings.TEMPLATES_DIR / "chatbot.html")

@app.post("/api/v1/session", response_model=SuccessResponse[SessionInitData])
async def create_session():
    """Create a new chat session"""
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available. Redis might be down.")
        session_id = await session_manager.create_session()
        available_providers = LLMProviderFactory.get_available_providers()

        return SuccessResponse[SessionInitData](
            data=SessionInitData(
                session_id=session_id,
                available_providers=available_providers,
                default_provider=settings.DEFAULT_LLM_PROVIDER
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /api/v1/session: {e}", exc_info=True)
        raise

@app.post("/api/v1/upload", response_model=SuccessResponse[UploadResultData])
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    session_id_query: str = Query(None, alias="session_id")
):
    """Upload a file for analysis"""
    session_id = session_id or session_id_query
    if not session_id:
        # This is a client error, HTTPException is appropriate
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available.")

    session = await session_manager.get_session(session_id) # SessionNotFoundError handled by middleware
    # No need to check `if not session:` here if SessionNotFoundError is raised and handled

    # File Extension Check
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise exceptions.InvalidFileTypeError(
            filename=file.filename,
            extension=file_extension
        )

    # MIME Type Check
    if file.content_type not in settings.ALLOWED_UPLOAD_MIME_TYPES:
        raise exceptions.InvalidFileTypeError(
            filename=file.filename,
            content_type=file.content_type
        )
    
    try:
        # Read file content for size check and writing
        content = await file.read()

        # File Size Check
        if len(content) > settings.MAX_UPLOAD_FILE_SIZE_BYTES:
            raise exceptions.FileSizeExceededError(
                filename=file.filename,
                max_size_mb=settings.MAX_UPLOAD_FILE_SIZE_BYTES / (1024 * 1024)
            )

        # Save uploaded file
        file_path = settings.UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as f:
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
            await session_manager.update_session(session)
        
        # Original response was {"success": True, "result": result}
        # We need to map this to UploadResultData which expects `result` field
        return SuccessResponse[UploadResultData](data=UploadResultData(result=result))
        
    except HTTPException: # Re-raise HTTPExceptions (like those from file validation)
        raise
    except Exception as e: # Catch other unexpected errors
        logger.error(f"Unexpected error in /api/v1/upload: {e}", exc_info=True)
        # Let the generic handler defined with @app.exception_handler(Exception) handle this
        raise

@app.post("/api/v1/chat", response_model=SuccessResponse[ChatCompletionData])
async def chat(request: ChatRequest):
    """Process chat messages"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available.")
    session = await session_manager.get_session(request.session_id) # SessionNotFoundError handled by middleware
    
    # Add user message to history
    session.add_message("user", request.message)
    
    # Switch provider if requested
    if request.provider:
        session.llm_provider = request.provider
    
    await session_manager.update_session(session) # Update session after message add and provider switch

    # Prepare messages for LLM
    system_message = settings.SYSTEM_PROMPT.format(
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
        await session_manager.update_session(session)
        
        return SuccessResponse[ChatCompletionData](
            data=ChatCompletionData(
                response=full_response,
                tool_results=tool_results,
                provider=session.llm_provider
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error in /api/v1/chat: {e}", exc_info=True)
        raise

@app.post("/api/v1/chat/stream") # Path versioning for stream
async def chat_stream(request: ChatRequest):
    """Stream chat responses"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available.")
    session = await session_manager.get_session(request.session_id)
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available.")
    session = await session_manager.get_session(request.session_id) # SessionNotFoundError handled by middleware
    
    async def generate():
        try:
            # Add user message
            session.add_message("user", request.message)
            
            # Switch provider if requested
            if request.provider:
                session.llm_provider = request.provider
            
            await session_manager.update_session(session) # Update session

            # Prepare messages
            system_message = settings.SYSTEM_PROMPT.format(
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
            await session_manager.update_session(session) # Update session
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/v1/provider/switch", response_model=SuccessResponse[ProviderSwitchData])
async def switch_provider(request: ProviderSwitchRequest):
    """Switch LLM provider for a session"""
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available.")
        session = await session_manager.get_session(request.session_id)

        available = LLMProviderFactory.get_available_providers()
        if request.provider not in available:
            raise HTTPException(status_code=400, detail=f"Provider {request.provider} not available")

        session.llm_provider = request.provider
        await session_manager.update_session(session)
        # Original response: {"success": True, "provider": request.provider}
        return SuccessResponse[ProviderSwitchData](data=ProviderSwitchData(provider=request.provider))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /api/v1/provider/switch: {e}", exc_info=True)
        raise

@app.get("/api/v1/history/{session_id}", response_model=SuccessResponse[ChatHistoryData])
async def get_history(session_id: str):
    """Get chat history"""
    try:
        if not session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available.")
        session = await session_manager.get_session(session_id)

        # Original response structure matches ChatHistoryData
        return SuccessResponse[ChatHistoryData](
            data=ChatHistoryData(
                messages=[{"role": msg.role, "content": msg.content} for msg in session.messages],
                context=session.context
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /api/v1/history/{session_id}: {e}", exc_info=True)
        raise

async def process_tool_calls(response: str, session: ChatSession) -> List[PyList[PyDict[str, Any]]]:
    """Extract and process tool calls from LLM response"""
    import re
    
    tool_results = []
    
    # Find all tool calls in the response
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            # Attempt to parse JSON and process the tool call
            try:
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
                    if session_manager:
                        await session_manager.update_session(session)

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

            except json.JSONDecodeError as e:
                logger.error("json_decode_error_processing_tool_call",
                             tool_match_content=match.strip()[:500],
                             error=str(e),
                             exc_info=True)
                tool_results.append({
                    "tool": "error_parsing_tool_call_json",
                    "error_message": f"Failed to parse JSON for tool call: {str(e)}",
                    "raw_content_snippet": match.strip()[:200]
                })
            
        except Exception as e:
            # This general except block catches errors from mcp_manager.call_tool or other unexpected issues
            logger.error(f"Error processing tool call: {e}", exc_info=True) # Added exc_info=True for more details
            # Determine tool_name safely for logging and result
            parsed_tool_name = "unknown_tool"
            if 'tool_data' in locals() and isinstance(tool_data, dict): # Check if tool_data was defined and is a dict
                parsed_tool_name = tool_data.get("tool", "unknown_tool_from_parsed_data")
            elif 'match' in locals(): # If tool_data parsing failed, try to get info from match
                 # Basic check if match might contain a tool name, very rough
                if '"tool":' in match.strip()[:100]: # Check a small part of the match string
                    try:
                        # Attempt a very cautious parse just for the tool name if possible
                        # This is risky, as the JSON is known to be invalid
                        # A more robust way would be regex, but for now, keep it simple
                        # For safety, this part is omitted to avoid new errors.
                        # parsed_tool_name will remain "unknown_tool" or based on prior state.
                        pass
                    except: # nosec
                        pass # Ignore if this cautious parse fails
            
            tool_results.append({
                "tool": parsed_tool_name, # Use the safely determined tool name
                "error": str(e)
            })
            
    return tool_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)