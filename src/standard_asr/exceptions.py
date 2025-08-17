# src/standard_asr/exceptions.py

class StandardASRError(Exception):
    """Base exception class for all errors raised by the standard_asr library."""
    pass

class ConfigError(StandardASRError, ValueError):
    """Raised when there is an error in the user-provided configuration."""
    pass

class TranscriptionError(StandardASRError):
    """Raised when an error occurs during the transcription process."""
    pass