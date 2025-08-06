"""
Advanced calibration system for photonic neural networks.

This module provides:
- Adaptive calibration algorithms
- Real-time error detection and correction
- Hardware safety checks and limits
- Calibration data management
- Automated calibration procedures
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
import pickle
from pathlib import Path
import time
import threading
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from ..core.exceptions import (
    CalibrationError, HardwareError, ValidationError,
    validate_parameter, check_array_validity, safe_execution,
    global_error_recovery
)


@dataclass
class CalibrationPoint:
    """Single calibration data point."""
    timestamp: datetime
    parameter_name: str
    target_value: float
    measured_value: float
    applied_correction: float
    error: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationPoint':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass 
class CalibrationResult:
    """Result of a calibration procedure."""
    success: bool
    component: str
    parameter: str
    initial_error: float
    final_error: float
    correction_applied: float
    iterations: int
    convergence_time: float
    confidence: float
    metadata: Dict[str, Any] = None


class CalibrationDatabase:
    """Database for storing and retrieving calibration data."""
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            database_path = Path.home() / ".photon_neuro" / "calibration.json"
        
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.data = self._load_database()
        self._lock = threading.Lock()
    
    def _load_database(self) -> Dict[str, Any]:
        """Load calibration database from disk."""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'version': '1.0',
                    'created': datetime.now().isoformat(),
                    'components': {},
                    'calibration_history': []
                }
        except Exception:
            # Return empty database if loading fails
            return {
                'version': '1.0', 
                'created': datetime.now().isoformat(),
                'components': {},
                'calibration_history': []
            }
    
    def save_database(self):
        """Save calibration database to disk."""
        try:
            with self._lock:
                with open(self.database_path, 'w') as f:
                    json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            raise CalibrationError(f"Failed to save calibration database: {e}")
    
    def add_calibration_point(self, component: str, point: CalibrationPoint):
        """Add calibration point to database."""
        with self._lock:
            if component not in self.data['components']:
                self.data['components'][component] = {
                    'parameters': {},
                    'last_calibrated': None
                }
            
            if point.parameter_name not in self.data['components'][component]['parameters']:
                self.data['components'][component]['parameters'][point.parameter_name] = []
            
            # Add point to parameter history
            self.data['components'][component]['parameters'][point.parameter_name].append(
                point.to_dict()
            )
            
            # Update last calibrated timestamp
            self.data['components'][component]['last_calibrated'] = point.timestamp.isoformat()
            
            # Limit history size (keep last 1000 points)
            param_history = self.data['components'][component]['parameters'][point.parameter_name]
            if len(param_history) > 1000:
                self.data['components'][component]['parameters'][point.parameter_name] = param_history[-1000:]
    
    def get_calibration_history(self, component: str, parameter: str, 
                              n_points: int = None) -> List[CalibrationPoint]:
        """Get calibration history for component parameter."""
        try:
            if component not in self.data['components']:
                return []
            
            if parameter not in self.data['components'][component]['parameters']:
                return []
            
            points_data = self.data['components'][component]['parameters'][parameter]
            
            if n_points is not None:
                points_data = points_data[-n_points:]
            
            points = [CalibrationPoint.from_dict(data) for data in points_data]
            return points
            
        except Exception as e:
            raise CalibrationError(f"Failed to retrieve calibration history: {e}")
    
    def get_latest_correction(self, component: str, parameter: str) -> Optional[float]:
        """Get latest correction value for a parameter."""
        history = self.get_calibration_history(component, parameter, n_points=1)
        return history[0].applied_correction if history else None
    
    def analyze_drift(self, component: str, parameter: str, 
                     time_window_hours: float = 24) -> Dict[str, float]:
        """Analyze parameter drift over time window."""
        try:
            history = self.get_calibration_history(component, parameter)
            
            if len(history) < 2:
                return {'drift_rate': 0.0, 'confidence': 0.0}
            
            # Filter points within time window
            cutoff_time = datetime.now().timestamp() - time_window_hours * 3600
            recent_points = [p for p in history if p.timestamp.timestamp() > cutoff_time]
            
            if len(recent_points) < 2:
                return {'drift_rate': 0.0, 'confidence': 0.0}
            
            # Calculate drift rate
            times = np.array([p.timestamp.timestamp() for p in recent_points])
            errors = np.array([p.error for p in recent_points])
            
            # Linear regression
            coeffs = np.polyfit(times, errors, 1)
            drift_rate = coeffs[0]  # Change per second
            
            # R-squared for confidence
            y_pred = np.polyval(coeffs, times)
            ss_res = np.sum((errors - y_pred) ** 2)
            ss_tot = np.sum((errors - np.mean(errors)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-12))
            
            return {
                'drift_rate': drift_rate * 3600,  # Convert to per hour
                'confidence': r_squared,
                'n_points': len(recent_points)
            }
            
        except Exception as e:
            raise CalibrationError(f"Drift analysis failed: {e}")


class AdaptiveCalibrator(ABC):
    """Abstract base class for adaptive calibration algorithms."""
    
    def __init__(self, name: str, tolerance: float = 1e-3, max_iterations: int = 50):
        self.name = name
        self.tolerance = validate_parameter("tolerance", tolerance,
                                          expected_type=(int, float), valid_range=(1e-6, 0.1))
        self.max_iterations = validate_parameter("max_iterations", max_iterations,
                                               expected_type=int, valid_range=(1, 1000))
        self.calibration_db = CalibrationDatabase()
        
    @abstractmethod
    def calibrate(self, target_value: float, measurement_func: Callable, 
                 adjustment_func: Callable, **kwargs) -> CalibrationResult:
        """Perform calibration procedure."""
        pass
    
    @abstractmethod
    def predict_correction(self, current_error: float, history: List[CalibrationPoint]) -> float:
        """Predict correction based on current error and history."""
        pass


class PIDCalibrator(AdaptiveCalibrator):
    """PID-based adaptive calibrator."""
    
    def __init__(self, name: str, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01, **kwargs):
        super().__init__(name, **kwargs)
        self.kp = validate_parameter("kp", kp, expected_type=(int, float), valid_range=(0, 100))
        self.ki = validate_parameter("ki", ki, expected_type=(int, float), valid_range=(0, 10))
        self.kd = validate_parameter("kd", kd, expected_type=(int, float), valid_range=(0, 10))
        
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.dt = 1.0  # Time step in seconds
    
    def calibrate(self, target_value: float, measurement_func: Callable,
                 adjustment_func: Callable, component: str = "unknown",
                 parameter: str = "unknown", **kwargs) -> CalibrationResult:
        """Perform PID calibration."""
        try:
            start_time = time.time()
            initial_measured = measurement_func()
            initial_error = abs(target_value - initial_measured)
            
            self.integral_error = 0.0
            self.previous_error = initial_error
            
            for iteration in range(self.max_iterations):
                # Measure current value
                current_measured = measurement_func()
                current_error = target_value - current_measured
                
                # PID calculation
                proportional = self.kp * current_error
                self.integral_error += current_error * self.dt
                integral = self.ki * self.integral_error
                derivative = self.kd * (current_error - self.previous_error) / self.dt
                
                correction = proportional + integral + derivative
                
                # Apply safety limits
                correction = np.clip(correction, -10.0, 10.0)  # Limit correction magnitude
                
                # Apply correction
                adjustment_func(correction)
                
                # Log calibration point
                cal_point = CalibrationPoint(
                    timestamp=datetime.now(),
                    parameter_name=parameter,
                    target_value=target_value,
                    measured_value=current_measured,
                    applied_correction=correction,
                    error=abs(current_error),
                    metadata={'iteration': iteration, 'algorithm': 'PID'}
                )
                
                self.calibration_db.add_calibration_point(component, cal_point)
                
                # Check convergence
                if abs(current_error) < self.tolerance:
                    final_time = time.time()
                    confidence = 1.0 / (1.0 + abs(current_error) / self.tolerance)
                    
                    return CalibrationResult(
                        success=True,
                        component=component,
                        parameter=parameter,
                        initial_error=initial_error,
                        final_error=abs(current_error),
                        correction_applied=correction,
                        iterations=iteration + 1,
                        convergence_time=final_time - start_time,
                        confidence=confidence
                    )
                
                self.previous_error = current_error
                time.sleep(0.1)  # Small delay between iterations
            
            # Failed to converge
            final_measured = measurement_func()
            final_error = abs(target_value - final_measured)
            
            return CalibrationResult(
                success=False,
                component=component,
                parameter=parameter,
                initial_error=initial_error,
                final_error=final_error,
                correction_applied=0.0,
                iterations=self.max_iterations,
                convergence_time=time.time() - start_time,
                confidence=0.0,
                metadata={'reason': 'max_iterations_reached'}
            )
            
        except Exception as e:
            raise CalibrationError(f"PID calibration failed: {e}")
    
    def predict_correction(self, current_error: float, 
                          history: List[CalibrationPoint]) -> float:
        """Predict correction using PID algorithm."""
        if not history:
            return self.kp * current_error
        
        # Calculate integral error from history
        integral_error = sum(point.error for point in history[-10:])  # Last 10 points
        
        # Calculate derivative error
        if len(history) >= 2:
            derivative_error = history[-1].error - history[-2].error
        else:
            derivative_error = 0.0
        
        correction = (self.kp * current_error + 
                     self.ki * integral_error * self.dt +
                     self.kd * derivative_error / self.dt)
        
        return np.clip(correction, -10.0, 10.0)


class MLCalibrator(AdaptiveCalibrator):
    """Machine learning-based adaptive calibrator."""
    
    def __init__(self, name: str, model_type: str = "linear", **kwargs):
        super().__init__(name, **kwargs)
        self.model_type = model_type
        self.model = None
        self._training_data = {'inputs': [], 'outputs': []}
        
    def calibrate(self, target_value: float, measurement_func: Callable,
                 adjustment_func: Callable, component: str = "unknown", 
                 parameter: str = "unknown", **kwargs) -> CalibrationResult:
        """Perform ML-based calibration."""
        try:
            start_time = time.time()
            initial_measured = measurement_func()
            initial_error = abs(target_value - initial_measured)
            
            # Get historical data for training
            history = self.calibration_db.get_calibration_history(component, parameter, n_points=100)
            
            for iteration in range(self.max_iterations):
                current_measured = measurement_func()
                current_error = target_value - current_measured
                
                # Predict correction using ML model or history
                if len(history) > 5:
                    correction = self.predict_correction(current_error, history)
                else:
                    # Fallback to simple proportional control
                    correction = 1.0 * current_error
                
                # Apply safety limits
                correction = np.clip(correction, -5.0, 5.0)
                
                # Apply correction
                adjustment_func(correction)
                
                # Log calibration point
                cal_point = CalibrationPoint(
                    timestamp=datetime.now(),
                    parameter_name=parameter,
                    target_value=target_value,
                    measured_value=current_measured,
                    applied_correction=correction,
                    error=abs(current_error),
                    metadata={'iteration': iteration, 'algorithm': 'ML'}
                )
                
                history.append(cal_point)
                self.calibration_db.add_calibration_point(component, cal_point)
                
                # Check convergence
                if abs(current_error) < self.tolerance:
                    confidence = 1.0 / (1.0 + abs(current_error) / self.tolerance)
                    
                    return CalibrationResult(
                        success=True,
                        component=component,
                        parameter=parameter,
                        initial_error=initial_error,
                        final_error=abs(current_error),
                        correction_applied=correction,
                        iterations=iteration + 1,
                        convergence_time=time.time() - start_time,
                        confidence=confidence
                    )
                
                time.sleep(0.1)
            
            # Failed to converge
            final_measured = measurement_func()
            final_error = abs(target_value - final_measured)
            
            return CalibrationResult(
                success=False,
                component=component,
                parameter=parameter,
                initial_error=initial_error,
                final_error=final_error,
                correction_applied=0.0,
                iterations=self.max_iterations,
                convergence_time=time.time() - start_time,
                confidence=0.0
            )
            
        except Exception as e:
            raise CalibrationError(f"ML calibration failed: {e}")
    
    def predict_correction(self, current_error: float, 
                          history: List[CalibrationPoint]) -> float:
        """Predict correction using ML model trained on history."""
        try:
            if len(history) < 3:
                return 1.0 * current_error  # Simple proportional
            
            # Extract features from history
            recent_errors = [point.error for point in history[-5:]]
            recent_corrections = [point.applied_correction for point in history[-5:]]
            
            # Simple linear regression model (in practice, could use more sophisticated models)
            if len(recent_errors) >= 2:
                # Fit linear relationship between error and required correction
                X = np.array(recent_errors[:-1]).reshape(-1, 1)
                y = np.array(recent_corrections[1:])
                
                if len(X) > 0:
                    # Simple least squares
                    slope = np.sum(X.flatten() * y) / (np.sum(X.flatten()**2) + 1e-12)
                    correction = slope * current_error
                    
                    return correction
            
            # Fallback
            return 1.0 * current_error
            
        except Exception:
            return 1.0 * current_error  # Safe fallback


class SafetyChecker:
    """Hardware safety checker for calibration procedures."""
    
    def __init__(self):
        self.safety_limits = {}
        self.emergency_stops = []
        
    def set_safety_limit(self, parameter: str, min_value: float, max_value: float):
        """Set safety limits for a parameter."""
        try:
            min_value = validate_parameter("min_value", min_value, expected_type=(int, float))
            max_value = validate_parameter("max_value", max_value, expected_type=(int, float))
            
            if min_value >= max_value:
                raise ValidationError("min_value must be less than max_value")
            
            self.safety_limits[parameter] = {'min': min_value, 'max': max_value}
            
        except ValidationError as e:
            raise CalibrationError(f"Invalid safety limit: {e}")
    
    def check_safety(self, parameter: str, value: float) -> bool:
        """Check if value is within safety limits."""
        try:
            if parameter not in self.safety_limits:
                return True  # No limits set
            
            limits = self.safety_limits[parameter]
            return limits['min'] <= value <= limits['max']
            
        except Exception:
            return False  # Err on side of caution
    
    def add_emergency_stop(self, stop_func: Callable):
        """Add emergency stop function."""
        self.emergency_stops.append(stop_func)
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger all emergency stop functions."""
        for stop_func in self.emergency_stops:
            try:
                stop_func(reason)
            except Exception as e:
                # Log but continue with other stops
                print(f"Emergency stop function failed: {e}")


class RealTimeErrorCorrector:
    """Real-time error detection and correction system."""
    
    def __init__(self, correction_interval: float = 1.0):
        self.correction_interval = correction_interval
        self.error_monitors = {}
        self.correctors = {}
        self.running = False
        self._thread = None
        self._stop_event = threading.Event()
        
    def add_error_monitor(self, name: str, measurement_func: Callable,
                         target_value: float, tolerance: float = 1e-3):
        """Add error monitor for a parameter."""
        self.error_monitors[name] = {
            'measurement_func': measurement_func,
            'target_value': target_value,
            'tolerance': tolerance,
            'last_error': 0.0,
            'error_count': 0
        }
    
    def add_corrector(self, name: str, calibrator: AdaptiveCalibrator,
                     adjustment_func: Callable):
        """Add corrector for a parameter."""
        self.correctors[name] = {
            'calibrator': calibrator,
            'adjustment_func': adjustment_func
        }
    
    def start_monitoring(self):
        """Start real-time monitoring and correction."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.running:
            return
        
        self.running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                for name, monitor in self.error_monitors.items():
                    # Measure current value
                    current_value = monitor['measurement_func']()
                    error = abs(monitor['target_value'] - current_value)
                    
                    monitor['last_error'] = error
                    
                    # Check if correction is needed
                    if error > monitor['tolerance']:
                        monitor['error_count'] += 1
                        
                        # Apply correction if corrector is available
                        if name in self.correctors:
                            corrector = self.correctors[name]
                            
                            # Get calibration history
                            history = corrector['calibrator'].calibration_db.get_calibration_history(
                                name, name, n_points=10
                            )
                            
                            # Predict correction
                            correction = corrector['calibrator'].predict_correction(
                                monitor['target_value'] - current_value, history
                            )
                            
                            # Apply correction
                            corrector['adjustment_func'](correction)
                            
                            # Log correction
                            cal_point = CalibrationPoint(
                                timestamp=datetime.now(),
                                parameter_name=name,
                                target_value=monitor['target_value'],
                                measured_value=current_value,
                                applied_correction=correction,
                                error=error,
                                metadata={'real_time': True}
                            )
                            
                            corrector['calibrator'].calibration_db.add_calibration_point(name, cal_point)
                    else:
                        monitor['error_count'] = max(0, monitor['error_count'] - 1)
                
                # Wait for next cycle
                if not self._stop_event.wait(self.correction_interval):
                    continue
                    
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop


class CalibrationManager:
    """High-level calibration system manager."""
    
    def __init__(self):
        self.calibrators = {}
        self.safety_checker = SafetyChecker()
        self.error_corrector = RealTimeErrorCorrector()
        self.calibration_db = CalibrationDatabase()
        
        # Set default safety limits
        self._set_default_safety_limits()
    
    def _set_default_safety_limits(self):
        """Set default safety limits for common parameters."""
        self.safety_checker.set_safety_limit('voltage', -10.0, 10.0)
        self.safety_checker.set_safety_limit('current', 0.0, 1.0)
        self.safety_checker.set_safety_limit('power', 0.0, 10.0)
        self.safety_checker.set_safety_limit('temperature', -50.0, 150.0)
    
    def register_calibrator(self, name: str, calibrator: AdaptiveCalibrator):
        """Register a calibrator."""
        self.calibrators[name] = calibrator
    
    def calibrate_component(self, component: str, parameter: str,
                           target_value: float, measurement_func: Callable,
                           adjustment_func: Callable,
                           calibrator_name: str = "pid") -> CalibrationResult:
        """Calibrate a component parameter."""
        try:
            # Check safety limits
            if not self.safety_checker.check_safety(parameter, target_value):
                raise CalibrationError(f"Target value {target_value} outside safety limits for {parameter}")
            
            # Get calibrator
            if calibrator_name not in self.calibrators:
                # Create default PID calibrator
                self.calibrators[calibrator_name] = PIDCalibrator(calibrator_name)
            
            calibrator = self.calibrators[calibrator_name]
            
            # Perform calibration
            result = calibrator.calibrate(
                target_value=target_value,
                measurement_func=measurement_func,
                adjustment_func=adjustment_func,
                component=component,
                parameter=parameter
            )
            
            return result
            
        except Exception as e:
            raise CalibrationError(f"Component calibration failed: {e}")
    
    def start_continuous_calibration(self):
        """Start continuous calibration monitoring."""
        self.error_corrector.start_monitoring()
    
    def stop_continuous_calibration(self):
        """Stop continuous calibration monitoring."""
        self.error_corrector.stop_monitoring()
    
    def get_calibration_report(self, component: str) -> Dict[str, Any]:
        """Generate calibration report for a component."""
        try:
            # Get all parameters for component
            if component not in self.calibration_db.data['components']:
                return {'component': component, 'status': 'not_calibrated'}
            
            component_data = self.calibration_db.data['components'][component]
            parameters = component_data.get('parameters', {})
            
            report = {
                'component': component,
                'last_calibrated': component_data.get('last_calibrated'),
                'parameters': {}
            }
            
            for param_name, param_history in parameters.items():
                if param_history:
                    latest = param_history[-1]
                    drift_analysis = self.calibration_db.analyze_drift(component, param_name)
                    
                    report['parameters'][param_name] = {
                        'latest_error': latest['error'],
                        'latest_correction': latest['applied_correction'],
                        'drift_rate': drift_analysis['drift_rate'],
                        'confidence': drift_analysis['confidence'],
                        'calibration_points': len(param_history)
                    }
            
            return report
            
        except Exception as e:
            raise CalibrationError(f"Failed to generate calibration report: {e}")


# Default global calibration manager instance
global_calibration_manager = CalibrationManager()