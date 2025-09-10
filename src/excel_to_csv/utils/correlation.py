"""Correlation ID management for tracking operations across components.

This module provides utilities for generating and managing correlation IDs
that allow tracking related operations throughout the system.
"""

import contextvars
import uuid
from typing import Optional


class CorrelationContext:
    """Context manager for correlation ID tracking across async operations."""
    
    _context: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id')
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set the correlation ID for the current context.
        
        Args:
            correlation_id: The correlation ID to set
        """
        cls._context.set(correlation_id)
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get the correlation ID from the current context.
        
        Returns:
            The current correlation ID, or None if not set
        """
        try:
            return cls._context.get()
        except LookupError:
            return None
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate a new UUID-based correlation ID.
        
        Returns:
            A new correlation ID string
        """
        return str(uuid.uuid4())
    
    @classmethod
    def ensure_correlation_id(cls) -> str:
        """Ensure a correlation ID exists, generating one if needed.
        
        Returns:
            The current or newly generated correlation ID
        """
        correlation_id = cls.get_correlation_id()
        if correlation_id is None:
            correlation_id = cls.generate_correlation_id()
            cls.set_correlation_id(correlation_id)
        return correlation_id
    
    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize context manager with optional correlation ID.
        
        Args:
            correlation_id: Optional correlation ID. If None, generates new one.
        """
        self.correlation_id = correlation_id or self.generate_correlation_id()
        self.token = None
    
    def __enter__(self) -> str:
        """Enter context and set correlation ID."""
        self.token = self._context.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore previous correlation ID."""
        if self.token is not None:
            self._context.reset(self.token)
        else:
            # If no previous token, clear the context
            try:
                self._context.set(None)
            except Exception:
                pass