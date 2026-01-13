"""Domain-level exceptions for business logic errors."""


class BaseException(Exception):
    """Base exception for all domain-level errors."""

    pass


class ResourceNotFoundError(BaseException):
    """Raised when a requested resource doesn't exist."""

    def __init__(self, resource_type: str, resource_id: str):
        """Initialize resource not found error.

        :param resource_type: The database object type of the nonexistent or archived resource
        :param resource_id: The ID of the nonexistent or archived resource
        """
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} with ID '{resource_id}' not found")


class ResourceAlreadyExistsError(BaseException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, resource_type: str, identifier: str):
        """Initialize resource already exists error.

        :param resource_type: The dtabase object type of the existing resource
        :param identifier: The unique ID of the already existing resource
        """
        self.resource_type = resource_type
        self.identifier = identifier
        super().__init__(
            f"{resource_type} with identifier '{identifier}' already exists"
        )


class ValidationError(BaseException):
    """Raised when domain validation fails."""

    def __init__(self, message: str, field: str | None = None):
        """Initialize domain logic validaiton error.

        :param message: Message detailing the specific validation that failed
        :param field: (Optional) field for which the validation failed.
        """
        self.field = field
        super().__init__(message)


class InvalidReferenceError(BaseException):
    """Raised when a foreign key reference is invalid."""

    def __init__(self, field: str, value: str, referenced_type: str):
        """Initialize invalid foreign key reference error.

        :param field: The field attribute of the incorrectly referenced resource.
        :param value: The invalid value of the referenced resource
        :param referenced_type: The database object type of the incorrectly referenced resource
        """
        self.field = field
        self.value = value
        self.referenced_type = referenced_type
        super().__init__(f"Invalid {field}: {referenced_type} '{value}' does not exist")


class ResourceArchivedError(BaseException):
    """Raised when attempting to operate on an archived resource."""

    def __init__(self, resource_type: str, resource_id: str):
        """
        Initialize archived resource error.

        :param resource type: The database object type of the archived resource
        :param resource_id: The ID of the archived resource
        """
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} '{resource_id}' is archived")


class DatabaseError(BaseException):
    """Raised when database operations fail due to infrastructure issues."""

    def __init__(self, operation: str, original_error: Exception | None = None):
        """Initialize database error.

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
