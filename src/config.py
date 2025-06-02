"""
Application configuration settings using Pydantic Settings.

This module defines a `Settings` class that loads configuration values from
environment variables and .env files. It includes settings for paths,
file uploads, LLM providers, MCP server, general server settings, and Redis.
It also includes a validator to create necessary directories on startup.
"""
from pathlib import Path
from typing import Set, List as PyList # Renamed List to PyList to avoid conflict if any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator, Field

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """
    # Path configurations
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent, description="Base directory of the application.")
    STATIC_DIR: Path = Field(description="Directory for static files.")
    TEMPLATES_DIR: Path = Field(description="Directory for HTML templates.")
    UPLOAD_DIR: Path = Field(description="Directory for user uploaded files.")
    EXPORT_DIR: Path = Field(description="Directory for exported files or data.")

    # File upload settings (general, used by server.py)
    ALLOWED_EXTENSIONS: Set[str] = Field(default={'.csv', '.xlsx', '.xls'}, description="Set of allowed file extensions for general server-side processing.")

    # File upload security settings (specific for /api/upload endpoint in chatbot_app.py)
    MAX_UPLOAD_FILE_SIZE_BYTES: int = Field(default=25 * 1024 * 1024, description="Maximum allowed file size for uploads in bytes (default: 25MB).")
    ALLOWED_UPLOAD_EXTENSIONS: Set[str] = Field(default={'.csv', '.xlsx', '.xls'}, description="Set of allowed file extensions for API uploads.")
    ALLOWED_UPLOAD_MIME_TYPES: Set[str] = Field(
        default={
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/csv'
        },
        description="Set of allowed MIME types for API uploads."
    )

    # LLM and MCP settings
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider to use (e.g., 'openai', 'anthropic').")
    MCP_SERVER_COMMAND: str = Field(default="python -m src.server", description="Command to start the MCP server.")
    SYSTEM_PROMPT: str = Field(
        default="""你是一个专业的数据分析助手。你可以帮助用户：
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

    # Server settings
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # Redis settings
    REDIS_HOST: str = Field(default="localhost", description="Hostname for the Redis server.")
    REDIS_PORT: int = Field(default=6379, description="Port number for the Redis server.")
    REDIS_DB: int = Field(default=0, description="Redis database number to use.")
    SESSION_EXPIRE_SECONDS: int = Field(default=3600, description="Session expiration time in seconds (default: 1 hour).")

    # Pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",                # Load .env file
        env_file_encoding="utf-8",      # Encoding for .env file
        extra="ignore"                  # Ignore extra fields from environment/dotenv
    )

    # Re-define path configurations that depend on BASE_DIR if they are not automatically handled
    # Pydantic v2 should handle this with default_factory for BASE_DIR and direct usage for others.
    # However, if issues arise, use @model_validator or properties.
    # For now, assuming direct usage post BASE_DIR initialization is fine.
    # Let's adjust the structure slightly to ensure BASE_DIR is resolved before others use it.

    @model_validator(mode='after')
    def _resolve_paths_and_create_dirs(self) -> 'Settings':
        """
        Resolves paths relative to BASE_DIR and creates necessary directories.
        This validator runs after the model is initialized.
        """
        # Resolve paths that depend on BASE_DIR
        # If BASE_DIR itself could be from .env, this ensures paths are correct.
        # For this class, BASE_DIR is fixed, so this is more for robust dynamic paths.
        self.STATIC_DIR = self.BASE_DIR / "static"
        self.TEMPLATES_DIR = self.BASE_DIR / "templates"
        self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        self.EXPORT_DIR = self.BASE_DIR / "exports"

        paths_to_create: PyList[Path] = [
            self.STATIC_DIR,
            self.TEMPLATES_DIR,
            self.UPLOAD_DIR,
            self.EXPORT_DIR,
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
        return self

# Instantiate settings
settings = Settings()
# Attempt to print a setting to ensure it's loaded (for debugging, can be removed)
# print(f"Config loaded: BASE_DIR is {settings.BASE_DIR}")
