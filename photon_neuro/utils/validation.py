"""
Input Validation Utilities
==========================
"""

from typing import Any, Union, List, Callable
import numpy as np

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

def validate_array_shape(arr: np.ndarray, expected_shape: tuple, name: str = "array") -> np.ndarray:
    """Validate array has expected shape."""
    if arr.shape != expected_shape:
        raise ValidationError(f"{name} shape {arr.shape} does not match expected {expected_shape}")
    return arr

def validate_positive_number(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """Validate that number is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value

def validate_range(value: Union[int, float], min_val: float, max_val: float, name: str = "value") -> Union[int, float]:
    """Validate that value is within specified range."""
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    return value