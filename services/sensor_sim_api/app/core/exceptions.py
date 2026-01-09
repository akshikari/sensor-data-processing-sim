"""Domain-level exceptions for business logic errors."""


class BaseException(Exception):
    """Base exception for all domain-level errors."""

    pass


class ResourceNotFoundError(BaseException):
    """Raised when a requested resource doesn't exist."""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} with ID '{resource_id}' not found")


class ResourceAlreadyExistsError(BaseException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, resource_type: str, identifier: str):
        self.resource_type = resource_type
        self.identifier = identifier
        super().__init__(
            f"{resource_type} with identifier '{identifier}' already exists"
        )


class ValidationError(BaseException):
    """Raised when business validation fails."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message)


class InvalidReferenceError(BaseException):
    """Raised when a foreign key reference is invalid."""

    def __init__(self, field: str, value: str, referenced_type: str):
        self.field = field
        self.value = value
        self.referenced_type = referenced_type
        super().__init__(f"Invalid {field}: {referenced_type} '{value}' does not exist")


class ResourceArchivedError(BaseException):
    """Raised when attempting to operate on an archived resource."""

    def __init__(self, resource_type: str, resource_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} '{resource_id}' is archived")


class DatabaseError(BaseException):
    """Raised when database operations fail due to infrastructure issues."""

    def __init__(self, operation: str, original_error: Exception | None = None):
        """
        Initialize database error.

        :param operation: Description of the operation that failed
            (e.g., "create accelerometer", "query sensor types")
        :param original_error: The underlying database exception for logging/debugging
        """
        self.operation = operation
        self.original_error = original_error

        message = f"Database operation failed: {operation}"
        if original_error:
            # Include error details for debugging, but sanitize sensitive info
            error_msg = str(original_error)
            # Don't expose full connection strings or credentials
            if "password" in error_msg.lower() or "conn" in error_msg.lower():
                message = f"Database operation failed: {operation} (connection error)"
            else:
                message = f"Database operation failed: {operation} ({error_msg})"

        super().__init__(message)
