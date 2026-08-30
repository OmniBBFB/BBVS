class BBVSError(Exception):
    """Base error with a message suitable for the command line."""


class DependencyError(BBVSError):
    """A required executable or optional package is unavailable."""


class CommandError(BBVSError):
    """An external command failed."""
