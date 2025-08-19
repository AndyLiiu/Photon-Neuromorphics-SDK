"""
Basic Error Handling Utilities
==============================
"""

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

def safe_execute(func: Callable) -> Callable:
    """Decorator to add basic error handling to functions."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    
    return wrapper

def validate_not_none(value: Any, name: str) -> Any:
    """Basic validation that value is not None."""
    if value is None:
        raise ValueError(f"Parameter {name} cannot be None")
    return value