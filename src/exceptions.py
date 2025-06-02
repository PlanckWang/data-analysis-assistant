"""
Custom exception classes for the Data Analysis Assistant application.

These exceptions provide more specific error information than built-in
exceptions and can be caught by dedicated handlers in the FastAPI application
to return appropriate HTTP responses.
"""

class DataAnalysisException(Exception):
    """Base class for exceptions in this application."""
    def __init__(self, message: str = "An error occurred during data analysis."):
        self.message = message
        super().__init__(self.message)

class SessionNotFoundError(DataAnalysisException):
    """Raised when a session ID is not found in storage."""
    def __init__(self, session_id: str):
        """
        Initializes SessionNotFoundError.

        Args:
            session_id: The ID of the session that was not found.
        """
        super().__init__(message=f"Session with ID '{session_id}' not found.")
        self.session_id = session_id

class DatasetNotFoundError(DataAnalysisException):
    """Raised when a dataset is not found or not loaded in the current session."""
    def __init__(self, dataset_name: str = None):
        """
        Initializes DatasetNotFoundError.

        Args:
            dataset_name: Optional name of the dataset that was not found.
                          If None, implies no dataset is currently loaded.
        """
        if dataset_name:
            super().__init__(message=f"Dataset '{dataset_name}' not found or not loaded.")
        else:
            super().__init__(message="No dataset loaded. Please upload a file first.")
        self.dataset_name = dataset_name

class InvalidToolArgumentError(DataAnalysisException):
    """Raised when an argument provided to an MCP tool is invalid."""
    def __init__(self, tool_name: str, argument_name: str, reason: str):
        """
        Initializes InvalidToolArgumentError.

        Args:
            tool_name: The name of the tool for which the argument was invalid.
            argument_name: The name of the invalid argument.
            reason: The reason why the argument is considered invalid.
        """
        super().__init__(message=f"Invalid argument '{argument_name}' for tool '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.argument_name = argument_name
        self.reason = reason

class ToolExecutionError(DataAnalysisException):
    """Raised when an error occurs during the execution of an MCP tool."""
    def __init__(self, tool_name: str, original_error: str):
        """
        Initializes ToolExecutionError.

        Args:
            tool_name: The name of the tool that failed to execute.
            original_error: A string representation of the original error.
        """
        super().__init__(message=f"Error executing tool '{tool_name}': {original_error}")
        self.tool_name = tool_name
        self.original_error = original_error

class MCPConnectionError(DataAnalysisException):
    """Raised when there's an issue connecting to or communicating with the MCP server."""
    def __init__(self, message: str = "Error connecting to or communicating with MCP server."):
        """
        Initializes MCPConnectionError.

        Args:
            message: A descriptive message for the connection error.
        """
        super().__init__(message=message)

class FileUploadError(DataAnalysisException):
    """Base class for errors that occur during file upload."""
    def __init__(self, message: str = "File upload failed."):
        super().__init__(message=message)

class FileSizeExceededError(FileUploadError):
    """Raised when an uploaded file exceeds the configured size limit."""
    def __init__(self, filename: str, max_size_mb: float):
        """
        Initializes FileSizeExceededError.

        Args:
            filename: The name of the file that exceeded the size limit.
            max_size_mb: The maximum allowed file size in megabytes.
        """
        super().__init__(message=f"File '{filename}' exceeds maximum size of {max_size_mb:.2f} MB.")
        self.filename = filename
        self.max_size_mb = max_size_mb

class InvalidFileTypeError(FileUploadError):
    """Raised when an uploaded file has an unsupported type (extension or MIME type)."""
    def __init__(self, filename: str, content_type: str = None, extension: str = None):
        """
        Initializes InvalidFileTypeError.

        Args:
            filename: The name of the file with the unsupported type.
            content_type: Optional. The detected MIME type of the file.
            extension: Optional. The detected file extension.
        """
        detail = f"File '{filename}' has an unsupported type."
        if content_type:
            detail += f" (Content-Type: {content_type})"
        if extension:
            detail += f" (Extension: {extension})"
        super().__init__(message=detail)
        self.filename = filename
        self.content_type = content_type
        self.extension = extension

class LLMProviderError(DataAnalysisException):
    """Raised for errors related to LLM provider interactions."""
    def __init__(self, provider_name: str, message: str):
        super().__init__(message=f"Error with LLM provider '{provider_name}': {message}")
        self.provider_name = provider_name
