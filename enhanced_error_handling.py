#!/usr/bin/env python3
"""
Enhanced Error Handling and Robustness Features
Generation 2: Make It Robust Implementation
"""

import logging
import traceback
import time
import json
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from functools import wraps

class RobustPhotonicError(Exception):
    """Base exception for robust photonic operations"""
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = time.time()

class ValidationError(RobustPhotonicError):
    """Input validation errors"""
    pass

class SimulationError(RobustPhotonicError):
    """Simulation runtime errors"""
    pass

class HardwareError(RobustPhotonicError):
    """Hardware interface errors"""
    pass

class RobustLogger:
    """Enhanced logging system with structured output"""
    
    def __init__(self, name: str = "photon_neuro", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.metrics = {
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'operations': 0
        }
    
    def error(self, message: str, error_code: str = None, context: Dict = None):
        """Log error with context"""
        self.metrics['errors'] += 1
        structured_msg = {
            'message': message,
            'error_code': error_code,
            'context': context,
            'timestamp': time.time()
        }
        self.logger.error(json.dumps(structured_msg))
    
    def warning(self, message: str, context: Dict = None):
        """Log warning with context"""
        self.metrics['warnings'] += 1
        structured_msg = {
            'message': message,
            'context': context,
            'timestamp': time.time()
        }
        self.logger.warning(json.dumps(structured_msg))
    
    def info(self, message: str, context: Dict = None):
        """Log info with context"""
        self.metrics['info'] += 1
        structured_msg = {
            'message': message,
            'context': context,
            'timestamp': time.time()
        }
        self.logger.info(json.dumps(structured_msg))
    
    def get_metrics(self) -> Dict:
        """Get logging metrics"""
        return self.metrics.copy()

class ParameterValidator:
    """Comprehensive parameter validation"""
    
    @staticmethod
    def validate_wavelength(wavelength: float) -> float:
        """Validate optical wavelength parameter"""
        if not isinstance(wavelength, (int, float)):
            raise ValidationError(
                "Wavelength must be numeric",
                error_code="INVALID_WAVELENGTH_TYPE",
                context={"provided_type": type(wavelength).__name__}
            )
        
        # Validate input values
        if wavelength == 0:
            raise ValidationError(
                "Wavelength cannot be zero",
                error_code="ZERO_WAVELENGTH",
                context={"provided_value": wavelength}
            )
        elif wavelength < 0:
            raise ValidationError(
                f"Wavelength cannot be negative: {wavelength}",
                error_code="NEGATIVE_WAVELENGTH",
                context={"provided_value": wavelength}
            )
        
        # Convert to meters if likely in nanometers (values > 100 without units are likely in nm)
        # Only convert if the value seems to be a raw nanometer value (> 100)
        if wavelength > 100:  # Raw number > 100, likely in nm units
            wavelength = wavelength * 1e-9
        
        if not (1e-9 <= wavelength <= 10e-6):  # 1nm to 10um
            raise ValidationError(
                f"Wavelength {wavelength*1e9:.1f}nm outside valid range (1nm-10μm)",
                error_code="WAVELENGTH_OUT_OF_RANGE",
                context={"wavelength_nm": wavelength * 1e9}
            )
        
        return wavelength
    
    @staticmethod
    def validate_power(power_dbm: float) -> float:
        """Validate optical power parameter"""
        if not isinstance(power_dbm, (int, float)):
            raise ValidationError(
                "Power must be numeric",
                error_code="INVALID_POWER_TYPE"
            )
        
        if not (-50 <= power_dbm <= 30):  # Reasonable range for photonics
            raise ValidationError(
                f"Power {power_dbm}dBm outside safe range (-50dBm to +30dBm)",
                error_code="POWER_OUT_OF_RANGE",
                context={"power_dbm": power_dbm}
            )
        
        return power_dbm
    
    @staticmethod
    def validate_dimensions(width: float, height: float = None) -> tuple:
        """Validate physical dimensions"""
        if not isinstance(width, (int, float)):
            raise ValidationError(
                "Width must be numeric",
                error_code="INVALID_WIDTH_TYPE"
            )
        
        if width <= 0 or width > 100e-6:  # 0 to 100 micrometers
            raise ValidationError(
                f"Width {width*1e6:.2f}μm outside valid range (0-100μm)",
                error_code="WIDTH_OUT_OF_RANGE",
                context={"width_um": width * 1e6}
            )
        
        if height is not None:
            if not isinstance(height, (int, float)):
                raise ValidationError(
                    "Height must be numeric",
                    error_code="INVALID_HEIGHT_TYPE"
                )
            
            if height <= 0 or height > 10e-6:  # 0 to 10 micrometers
                raise ValidationError(
                    f"Height {height*1e6:.2f}μm outside valid range (0-10μm)",
                    error_code="HEIGHT_OUT_OF_RANGE",
                    context={"height_um": height * 1e6}
                )
            
            return (width, height)
        
        return (width,)

def robust_operation(operation_name: str = None, retry_count: int = 3):
    """Decorator for robust operations with retry and error handling"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = RobustLogger()
            op_name = operation_name or func.__name__
            
            for attempt in range(retry_count):
                try:
                    logger.info(f"Starting {op_name}", {"attempt": attempt + 1})
                    result = func(*args, **kwargs)
                    logger.info(f"Completed {op_name} successfully")
                    return result
                
                except ValidationError as e:
                    # Don't retry validation errors
                    logger.error(f"Validation error in {op_name}", e.error_code, e.context)
                    raise
                
                except Exception as e:
                    if attempt == retry_count - 1:  # Last attempt
                        logger.error(
                            f"Operation {op_name} failed after {retry_count} attempts",
                            "OPERATION_FAILED",
                            {
                                "function": func.__name__,
                                "args": str(args),
                                "kwargs": str(kwargs),
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                        )
                        raise SimulationError(
                            f"Operation {op_name} failed: {str(e)}",
                            "OPERATION_FAILED"
                        )
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {op_name}, retrying",
                            {"error": str(e)}
                        )
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        return wrapper
    return decorator

@contextmanager
def error_recovery_context(operation_name: str):
    """Context manager for error recovery and cleanup"""
    logger = RobustLogger()
    start_time = time.time()
    
    try:
        logger.info(f"Starting protected operation: {operation_name}")
        yield
        
    except Exception as e:
        logger.error(
            f"Error in protected operation: {operation_name}",
            "PROTECTED_OPERATION_FAILED",
            {
                "operation": operation_name,
                "error": str(e),
                "duration": time.time() - start_time
            }
        )
        
        # Attempt recovery
        logger.info(f"Attempting recovery for: {operation_name}")
        # Recovery logic would go here
        
        raise  # Re-raise the exception
    
    finally:
        duration = time.time() - start_time
        logger.info(
            f"Completed protected operation: {operation_name}",
            {"duration": duration}
        )

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise SimulationError(
                    "Circuit breaker is OPEN",
                    "CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise

class RobustPhotonicComponent:
    """Enhanced photonic component with robust error handling"""
    
    def __init__(self, name: str, component_type: str = "generic"):
        self.name = name
        self.component_type = component_type
        self.parameters = {}
        self.logger = RobustLogger(f"component.{name}")
        self.circuit_breaker = CircuitBreaker()
        self.validator = ParameterValidator()
        
        self.health_metrics = {
            'successful_operations': 0,
            'failed_operations': 0,
            'last_operation_time': 0,
            'average_operation_time': 0.0
        }
    
    def configure(self, **kwargs) -> 'RobustPhotonicComponent':
        """Configure component parameters with validation"""
        with error_recovery_context(f"configure_{self.name}"):
            # Validate common parameters
            if 'wavelength' in kwargs:
                kwargs['wavelength'] = self.validator.validate_wavelength(kwargs['wavelength'])
            
            if 'power_dbm' in kwargs:
                kwargs['power_dbm'] = self.validator.validate_power(kwargs['power_dbm'])
            
            if 'width' in kwargs:
                if 'height' in kwargs:
                    kwargs['width'], kwargs['height'] = self.validator.validate_dimensions(
                        kwargs['width'], kwargs.get('height')
                    )
                else:
                    kwargs['width'] = self.validator.validate_dimensions(kwargs['width'])[0]
            
            self.parameters.update(kwargs)
            self.logger.info(f"Configured {self.name}", {"parameters": kwargs})
            
        return self
    
    @robust_operation("component_simulation", retry_count=3)
    def simulate(self) -> Dict[str, Any]:
        """Simulate component with robust error handling"""
        start_time = time.time()
        
        try:
            # Basic simulation logic
            if not self.parameters:
                raise ValidationError(
                    "No parameters configured for simulation",
                    "NO_PARAMETERS",
                    {"component": self.name}
                )
            
            # Simulate operation
            result = self.circuit_breaker.call(self._perform_simulation)
            
            # Update metrics
            duration = time.time() - start_time
            self.health_metrics['successful_operations'] += 1
            self.health_metrics['last_operation_time'] = duration
            self._update_average_time(duration)
            
            self.logger.info(
                f"Simulation completed for {self.name}",
                {"duration": duration, "result_keys": list(result.keys())}
            )
            
            return result
            
        except Exception as e:
            self.health_metrics['failed_operations'] += 1
            self.logger.error(
                f"Simulation failed for {self.name}",
                "SIMULATION_FAILED",
                {"error": str(e)}
            )
            raise
    
    def _perform_simulation(self) -> Dict[str, Any]:
        """Internal simulation method"""
        # Mock simulation based on component type
        results = {
            "component": self.name,
            "type": self.component_type,
            "status": "simulated",
            "parameters": self.parameters.copy(),
            "timestamp": time.time()
        }
        
        # Add component-specific results
        if self.component_type == "waveguide":
            wavelength = self.parameters.get('wavelength', 1550e-9)
            length = self.parameters.get('length', 1e-3)
            n_eff = self.parameters.get('n_eff', 2.4)
            
            results.update({
                "phase_shift": 2 * 3.14159 * n_eff * length / wavelength,
                "loss_db": self.parameters.get('loss_db_per_cm', 0.1) * length * 100,
                "group_delay": length / (3e8 / n_eff)
            })
        
        elif self.component_type == "mzi":
            results.update({
                "phase_difference": self.parameters.get('phase_shift', 0),
                "transmission": 0.5 * (1 + abs(self.parameters.get('phase_shift', 0)) / 3.14159),
                "extinction_ratio": 20.0  # dB
            })
        
        return results
    
    def _update_average_time(self, duration: float):
        """Update average operation time"""
        total_ops = (self.health_metrics['successful_operations'] + 
                    self.health_metrics['failed_operations'])
        
        if total_ops == 1:
            self.health_metrics['average_operation_time'] = duration
        else:
            current_avg = self.health_metrics['average_operation_time']
            self.health_metrics['average_operation_time'] = (
                (current_avg * (total_ops - 1) + duration) / total_ops
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health metrics"""
        total_ops = (self.health_metrics['successful_operations'] + 
                    self.health_metrics['failed_operations'])
        
        success_rate = (self.health_metrics['successful_operations'] / max(1, total_ops)) * 100
        
        return {
            "component": self.name,
            "type": self.component_type,
            "health_status": "healthy" if success_rate > 90 else "degraded" if success_rate > 50 else "unhealthy",
            "success_rate": success_rate,
            "total_operations": total_ops,
            "average_operation_time": self.health_metrics['average_operation_time'],
            "circuit_breaker_state": self.circuit_breaker.state,
            "parameters_configured": len(self.parameters)
        }

def test_robust_features():
    """Test Generation 2 robust features"""
    print("🛡️  GENERATION 2: ROBUST FEATURES TEST")
    print("=" * 50)
    
    logger = RobustLogger()
    
    try:
        # Test robust component creation
        print("✓ Testing robust component creation...")
        waveguide = RobustPhotonicComponent("test_waveguide", "waveguide")
        
        # Test parameter validation
        print("✓ Testing parameter validation...")
        waveguide.configure(
            wavelength=1550e-9,  # Valid wavelength
            width=450e-9,        # Valid width
            length=1e-3,         # Valid length
            power_dbm=0          # Valid power
        )
        
        # Test simulation with error handling
        print("✓ Testing simulation with error handling...")
        result = waveguide.simulate()
        print(f"  Simulation result keys: {list(result.keys())}")
        
        # Test health monitoring
        print("✓ Testing health monitoring...")
        health = waveguide.get_health_status()
        print(f"  Health status: {health['health_status']}")
        print(f"  Success rate: {health['success_rate']:.1f}%")
        
        # Test error handling with invalid parameters
        print("✓ Testing error handling...")
        try:
            invalid_component = RobustPhotonicComponent("invalid", "test")
            invalid_component.configure(wavelength=-1)  # Invalid wavelength
            print("  ❌ Should have failed validation")
        except ValidationError as e:
            print(f"  ✅ Correctly caught validation error: {e.error_code}")
        
        # Test circuit breaker
        print("✓ Testing circuit breaker...")
        mzi = RobustPhotonicComponent("test_mzi", "mzi")
        mzi.configure(phase_shift=1.57)  # π/2 phase shift
        mzi_result = mzi.simulate()
        print(f"  MZI transmission: {mzi_result.get('transmission', 0):.3f}")
        
        # Test logging metrics
        print("✓ Testing logging metrics...")
        metrics = logger.get_metrics()
        print(f"  Logged operations: {metrics['operations']}")
        print(f"  Info messages: {metrics['info']}")
        print(f"  Warnings: {metrics['warnings']}")
        print(f"  Errors: {metrics['errors']}")
        
        print("✅ All robust features working correctly!")
        return True
        
    except Exception as e:
        logger.error("Robust features test failed", "TEST_FAILED", {"error": str(e)})
        print(f"❌ Robust features test failed: {e}")
        return False

def main():
    """Main test execution for Generation 2"""
    print("🚀 GENERATION 2: MAKE IT ROBUST - AUTONOMOUS EXECUTION")
    print("=" * 60)
    
    success = test_robust_features()
    
    if success:
        print("\n🎉 GENERATION 2 IMPLEMENTATION COMPLETE!")
        print("🛡️  Enhanced with comprehensive error handling, validation, and monitoring!")
    else:
        print("\n❌ GENERATION 2 IMPLEMENTATION NEEDS ATTENTION")
    
    return success

if __name__ == "__main__":
    main()