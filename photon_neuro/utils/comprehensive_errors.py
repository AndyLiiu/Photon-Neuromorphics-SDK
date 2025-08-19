"""
Comprehensive Error Handling System
===================================
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"

class ErrorContext:
    """Context information for error handling."""
    
    def __init__(self, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.operation = operation
        self.severity = severity
        self.retry_count = 0
        self.max_retries = 3
        self.recovery_strategy = RecoveryStrategy.RETRY

def with_error_recovery(context: ErrorContext):
    """Decorator for comprehensive error handling with recovery."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[Any]:
            last_exception = None
            
            for attempt in range(context.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Operation {context.operation} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    context.retry_count = attempt
                    
                    if context.severity == ErrorSeverity.CRITICAL:
                        logger.critical(f"Critical error in {context.operation}: {e}")
                        raise
                    
                    if attempt < context.max_retries and context.recovery_strategy == RecoveryStrategy.RETRY:
                        wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                        logger.warning(f"Retrying {context.operation} in {wait_time:.1f}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        continue
                    
                    break
            
            # Final error handling
            if context.recovery_strategy == RecoveryStrategy.FAIL_FAST:
                raise last_exception
            elif context.recovery_strategy == RecoveryStrategy.IGNORE:
                logger.warning(f"Ignoring error in {context.operation}: {last_exception}")
                return None
            else:
                logger.error(f"Operation {context.operation} failed after {context.max_retries + 1} attempts: {last_exception}")
                return None
        
        return wrapper
    return decorator