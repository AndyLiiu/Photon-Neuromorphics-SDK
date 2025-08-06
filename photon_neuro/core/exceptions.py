"""
Custom exception classes for Photon Neuromorphics SDK.

This module provides comprehensive error handling with custom exception
classes for different error types and graceful degradation mechanisms.
"""

import traceback
import logging
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
import torch
import numpy as np


class PhotonicError(Exception):
    """Base exception class for all photonic-related errors."""
    
    def __init__(self, message: str, component: Optional[str] = None,
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.component = component
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None
        
    def __str__(self):
        error_str = f"PhotonicError: {super().__str__()}"
        if self.component:
            error_str += f" (Component: {self.component})"
        if self.error_code:
            error_str += f" (Code: {self.error_code})"
        return error_str
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'component': self.component,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }


class SimulationError(PhotonicError):
    """Errors related to simulation execution."""
    
    def __init__(self, message: str, simulation_type: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.simulation_type = simulation_type
        self.parameters = parameters or {}


class ComponentError(PhotonicError):
    """Errors related to photonic components."""
    
    def __init__(self, message: str, component_type: Optional[str] = None,
                 component_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component_type = component_type
        self.component_id = component_id


class HardwareError(PhotonicError):
    """Errors related to hardware interfaces."""
    
    def __init__(self, message: str, hardware_type: Optional[str] = None,
                 connection_string: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.hardware_type = hardware_type
        self.connection_string = connection_string


class CalibrationError(PhotonicError):
    """Errors related to calibration procedures."""
    
    def __init__(self, message: str, calibration_type: Optional[str] = None,
                 calibration_step: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.calibration_type = calibration_type
        self.calibration_step = calibration_step


class PowerBudgetError(PhotonicError):
    """Errors related to power budget analysis."""
    
    def __init__(self, message: str, power_type: Optional[str] = None,
                 threshold_exceeded: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.power_type = power_type
        self.threshold_exceeded = threshold_exceeded or {}


class ValidationError(PhotonicError):
    """Errors related to parameter validation."""
    
    def __init__(self, message: str, parameter_name: Optional[str] = None,
                 expected_type: Optional[type] = None, received_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.received_value = received_value


class ConvergenceError(PhotonicError):
    """Errors related to numerical convergence failures."""
    
    def __init__(self, message: str, algorithm: Optional[str] = None,
                 max_iterations: Optional[int] = None, final_error: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.final_error = final_error


class ThermalError(PhotonicError):
    """Errors related to thermal management."""
    
    def __init__(self, message: str, temperature_limit: Optional[float] = None,
                 current_temperature: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.temperature_limit = temperature_limit
        self.current_temperature = current_temperature


class NoiseModelError(PhotonicError):
    """Errors related to noise modeling."""
    
    def __init__(self, message: str, noise_type: Optional[str] = None,
                 noise_parameters: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.noise_type = noise_type
        self.noise_parameters = noise_parameters or {}


class DataIntegrityError(PhotonicError):
    """Errors related to data integrity checks."""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 corruption_details: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.corruption_details = corruption_details or {}


class ErrorRecovery:
    """Error recovery and graceful degradation mechanisms."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_strategies = {}
        self.fallback_values = {}
        self.error_history: List[PhotonicError] = []
        
    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
        
    def set_fallback_value(self, parameter_name: str, fallback_value: Any):
        """Set fallback value for a parameter."""
        self.fallback_values[parameter_name] = fallback_value
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle error with registered recovery strategy."""
        self.error_history.append(error)
        
        # Log error
        self.logger.error(f"Error occurred: {error}", extra={'context': context})
        
        # Try registered recovery strategy
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                
        # Try parent class strategies
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    return strategy(error, context)
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
                    
        # Default recovery: return None or raise
        self.logger.error(f"No recovery strategy found for {error_type}")
        return None
        
    def get_fallback_value(self, parameter_name: str, default: Any = None) -> Any:
        """Get fallback value for a parameter."""
        return self.fallback_values.get(parameter_name, default)
        
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()


def safe_execution(fallback_value: Any = None, 
                  raise_on_failure: bool = False,
                  error_types: tuple = (Exception,)):
    """Decorator for safe execution with error handling."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error in {func.__name__}: {e}")
                
                if raise_on_failure:
                    raise
                    
                return fallback_value
                
        return wrapper
    return decorator


def validate_parameter(param_name: str, value: Any, 
                      expected_type: type = None,
                      valid_range: tuple = None,
                      valid_values: list = None,
                      allow_none: bool = False) -> Any:
    """Validate parameter with comprehensive checks."""
    
    # Check for None
    if value is None:
        if allow_none:
            return value
        else:
            raise ValidationError(
                f"Parameter '{param_name}' cannot be None",
                parameter_name=param_name,
                received_value=value
            )
    
    # Check type
    if expected_type is not None:
        if isinstance(expected_type, (list, tuple)):
            if not any(isinstance(value, t) for t in expected_type):
                raise ValidationError(
                    f"Parameter '{param_name}' must be one of types {expected_type}, got {type(value)}",
                    parameter_name=param_name,
                    expected_type=expected_type,
                    received_value=value
                )
        else:
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"Parameter '{param_name}' must be of type {expected_type}, got {type(value)}",
                    parameter_name=param_name,
                    expected_type=expected_type,
                    received_value=value
                )
    
    # Check valid range
    if valid_range is not None and hasattr(value, '__gt__'):
        min_val, max_val = valid_range
        if not (min_val <= value <= max_val):
            raise ValidationError(
                f"Parameter '{param_name}' must be in range [{min_val}, {max_val}], got {value}",
                parameter_name=param_name,
                received_value=value
            )
    
    # Check valid values
    if valid_values is not None:
        if value not in valid_values:
            raise ValidationError(
                f"Parameter '{param_name}' must be one of {valid_values}, got {value}",
                parameter_name=param_name,
                received_value=value
            )
    
    return value


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Check tensor for NaN, Inf, and other issues."""
    
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"{name} must be a torch.Tensor, got {type(tensor)}",
            parameter_name=name,
            expected_type=torch.Tensor,
            received_value=type(tensor)
        )
    
    if torch.isnan(tensor).any():
        raise DataIntegrityError(
            f"{name} contains NaN values",
            data_type="tensor",
            corruption_details={'has_nan': True, 'shape': tensor.shape}
        )
    
    if torch.isinf(tensor).any():
        raise DataIntegrityError(
            f"{name} contains infinite values",
            data_type="tensor",
            corruption_details={'has_inf': True, 'shape': tensor.shape}
        )
    
    return tensor


def check_array_validity(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Check numpy array for NaN, Inf, and other issues."""
    
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            f"{name} must be a numpy.ndarray, got {type(array)}",
            parameter_name=name,
            expected_type=np.ndarray,
            received_value=type(array)
        )
    
    if np.isnan(array).any():
        raise DataIntegrityError(
            f"{name} contains NaN values",
            data_type="array",
            corruption_details={'has_nan': True, 'shape': array.shape}
        )
    
    if np.isinf(array).any():
        raise DataIntegrityError(
            f"{name} contains infinite values",
            data_type="array",
            corruption_details={'has_inf': True, 'shape': array.shape}
        )
    
    return array


def robust_division(numerator: Any, denominator: Any, epsilon: float = 1e-12) -> Any:
    """Perform division with protection against division by zero."""
    
    if isinstance(denominator, (int, float)):
        if abs(denominator) < epsilon:
            raise ValidationError(
                f"Division by zero or near-zero value: {denominator}",
                parameter_name="denominator",
                received_value=denominator
            )
    elif isinstance(denominator, (torch.Tensor, np.ndarray)):
        if hasattr(denominator, 'abs'):
            small_values = denominator.abs() < epsilon
        else:
            small_values = np.abs(denominator) < epsilon
            
        if small_values.any():
            raise ValidationError(
                f"Division by zero or near-zero values detected in denominator",
                parameter_name="denominator",
                received_value="array with small values"
            )
    
    return numerator / denominator


def safe_sqrt(value: Any, epsilon: float = 1e-12) -> Any:
    """Compute square root with protection against negative values."""
    
    if isinstance(value, (int, float)):
        if value < -epsilon:
            raise ValidationError(
                f"Cannot compute square root of negative value: {value}",
                parameter_name="value",
                received_value=value
            )
        return np.sqrt(max(0, value))
    elif isinstance(value, torch.Tensor):
        if (value < -epsilon).any():
            raise ValidationError(
                f"Cannot compute square root of tensor with negative values",
                parameter_name="value",
                received_value="tensor with negative values"
            )
        return torch.sqrt(torch.clamp(value, min=0))
    elif isinstance(value, np.ndarray):
        if (value < -epsilon).any():
            raise ValidationError(
                f"Cannot compute square root of array with negative values",
                parameter_name="value",
                received_value="array with negative values"
            )
        return np.sqrt(np.maximum(0, value))
    else:
        raise ValidationError(
            f"Unsupported type for safe_sqrt: {type(value)}",
            parameter_name="value",
            expected_type=(int, float, torch.Tensor, np.ndarray),
            received_value=type(value)
        )


def safe_log(value: Any, epsilon: float = 1e-12) -> Any:
    """Compute logarithm with protection against non-positive values."""
    
    if isinstance(value, (int, float)):
        if value <= epsilon:
            raise ValidationError(
                f"Cannot compute logarithm of non-positive value: {value}",
                parameter_name="value",
                received_value=value
            )
        return np.log(value)
    elif isinstance(value, torch.Tensor):
        if (value <= epsilon).any():
            raise ValidationError(
                f"Cannot compute logarithm of tensor with non-positive values",
                parameter_name="value",
                received_value="tensor with non-positive values"
            )
        return torch.log(torch.clamp(value, min=epsilon))
    elif isinstance(value, np.ndarray):
        if (value <= epsilon).any():
            raise ValidationError(
                f"Cannot compute logarithm of array with non-positive values",
                parameter_name="value",
                received_value="array with non-positive values"
            )
        return np.log(np.maximum(epsilon, value))
    else:
        raise ValidationError(
            f"Unsupported type for safe_log: {type(value)}",
            parameter_name="value",
            expected_type=(int, float, torch.Tensor, np.ndarray),
            received_value=type(value)
        )


# Create global error recovery instance
global_error_recovery = ErrorRecovery()

# Default recovery strategies
def default_simulation_recovery(error: SimulationError, context: Dict[str, Any]) -> Any:
    """Default recovery strategy for simulation errors."""
    logger = logging.getLogger(__name__)
    logger.warning(f"Simulation failed, using fallback simulation: {error}")
    
    # Return empty result with correct structure
    if 'expected_output_shape' in context:
        return torch.zeros(context['expected_output_shape'], dtype=torch.complex64)
    return None


def default_component_recovery(error: ComponentError, context: Dict[str, Any]) -> Any:
    """Default recovery strategy for component errors."""
    logger = logging.getLogger(__name__)
    logger.warning(f"Component failed, using unity transmission: {error}")
    
    # Return identity transformation
    if 'input_field' in context:
        return context['input_field']
    return torch.ones(1, dtype=torch.complex64)


def default_hardware_recovery(error: HardwareError, context: Dict[str, Any]) -> Any:
    """Default recovery strategy for hardware errors."""
    logger = logging.getLogger(__name__)
    logger.warning(f"Hardware interface failed, using simulation mode: {error}")
    
    # Switch to simulation mode
    return {'mode': 'simulation', 'hardware_available': False}


# Register default recovery strategies
global_error_recovery.register_recovery_strategy(SimulationError, default_simulation_recovery)
global_error_recovery.register_recovery_strategy(ComponentError, default_component_recovery)
global_error_recovery.register_recovery_strategy(HardwareError, default_hardware_recovery)

# Set default fallback values
global_error_recovery.set_fallback_value('wavelength', 1550e-9)
global_error_recovery.set_fallback_value('temperature', 25.0)
global_error_recovery.set_fallback_value('optical_power', 1e-3)
global_error_recovery.set_fallback_value('insertion_loss', 0.1)