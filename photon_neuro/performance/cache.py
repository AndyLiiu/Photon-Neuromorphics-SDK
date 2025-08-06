"""
Adaptive caching system for photonic simulations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, Union
import hashlib
import time
import threading
from collections import OrderedDict
import weakref
import pickle


class AdaptiveCache:
    """Intelligent cache that adapts based on access patterns."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600,
                 adaptation_rate: float = 0.1):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.adaptation_rate = adaptation_rate
        
        # Storage
        self._cache = OrderedDict()
        self._access_times = {}
        self._access_counts = {}
        self._hit_rates = {}
        
        # Adaptation metrics
        self._total_requests = 0
        self._total_hits = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with access pattern tracking."""
        with self._lock:
            self._total_requests += 1
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL
                if current_time - self._access_times.get(key, 0) > self.ttl:
                    self._evict(key)
                    return default
                    
                # Update access patterns
                self._access_times[key] = current_time
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
                self._total_hits += 1
                
                # Move to end (LRU)
                value = self._cache.pop(key)
                self._cache[key] = value
                
                return value
            
            return default
            
    def put(self, key: str, value: Any, priority: float = 1.0):
        """Put item in cache with priority weighting."""
        with self._lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = value
                self._access_times[key] = current_time
                return
                
            # Check capacity and evict if needed
            while len(self._cache) >= self.max_size:
                self._adaptive_eviction()
                
            # Add new item
            self._cache[key] = value
            self._access_times[key] = current_time
            self._access_counts[key] = 0
            self._hit_rates[key] = priority
            
    def _adaptive_eviction(self):
        """Intelligent eviction based on access patterns."""
        if not self._cache:
            return
            
        current_time = time.time()
        eviction_scores = {}
        
        for key in self._cache:
            # Calculate eviction score based on multiple factors
            time_since_access = current_time - self._access_times.get(key, 0)
            access_count = self._access_counts.get(key, 0)
            hit_rate = self._hit_rates.get(key, 1.0)
            
            # Higher score = more likely to be evicted
            score = (time_since_access / self.ttl +
                    1.0 / (access_count + 1) +
                    1.0 / hit_rate)
            
            eviction_scores[key] = score
            
        # Evict item with highest score
        key_to_evict = max(eviction_scores, key=eviction_scores.get)
        self._evict(key_to_evict)
        
    def _evict(self, key: str):
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._access_counts:
            del self._access_counts[key]
        if key in self._hit_rates:
            del self._hit_rates[key]
            
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._hit_rates.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self._total_hits / max(self._total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_requests': self._total_requests,
                'total_hits': self._total_hits,
                'utilization': len(self._cache) / self.max_size
            }
            
    def optimize(self):
        """Optimize cache parameters based on usage patterns."""
        with self._lock:
            stats = self.get_stats()
            
            # Adapt cache size based on hit rate
            if stats['hit_rate'] > 0.9:
                # High hit rate - can reduce size slightly
                self.max_size = max(100, int(self.max_size * 0.95))
            elif stats['hit_rate'] < 0.5:
                # Low hit rate - increase size
                self.max_size = int(self.max_size * 1.1)
                
            # Adapt TTL based on access patterns
            if self._access_times:
                avg_access_interval = np.mean([
                    time.time() - t for t in self._access_times.values()
                ])
                self.ttl = max(300, avg_access_interval * 2)  # At least 5 minutes


class TensorCache(AdaptiveCache):
    """Specialized cache for PyTorch tensors."""
    
    def __init__(self, max_memory_gb: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self._memory_usage = 0
        
    def _tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor memory usage in bytes."""
        return tensor.element_size() * tensor.numel()
        
    def put(self, key: str, tensor: torch.Tensor, priority: float = 1.0):
        """Put tensor in cache with memory management."""
        if not isinstance(tensor, torch.Tensor):
            super().put(key, tensor, priority)
            return
            
        tensor_size = self._tensor_size(tensor)
        
        with self._lock:
            # Evict items to make room
            while (self._memory_usage + tensor_size > self.max_memory_bytes and 
                   len(self._cache) > 0):
                self._adaptive_eviction()
                
            # Add to cache
            if key in self._cache:
                # Update existing
                old_tensor = self._cache[key]
                if isinstance(old_tensor, torch.Tensor):
                    self._memory_usage -= self._tensor_size(old_tensor)
                    
            super().put(key, tensor, priority)
            self._memory_usage += tensor_size
            
    def _evict(self, key: str):
        """Remove tensor and update memory usage."""
        if key in self._cache:
            tensor = self._cache[key]
            if isinstance(tensor, torch.Tensor):
                self._memory_usage -= self._tensor_size(tensor)
                
        super()._evict(key)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including memory usage."""
        stats = super().get_stats()
        stats.update({
            'memory_usage_gb': self._memory_usage / (1024**3),
            'max_memory_gb': self.max_memory_bytes / (1024**3),
            'memory_utilization': self._memory_usage / self.max_memory_bytes
        })
        return stats


class OpticalFieldCache(TensorCache):
    """Specialized cache for optical field simulations."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._field_metadata = {}
        
    def put_field(self, wavelength: float, geometry_hash: str, 
                  field: torch.Tensor, metadata: Dict[str, Any] = None):
        """Cache optical field with wavelength and geometry."""
        key = self._generate_field_key(wavelength, geometry_hash)
        
        self._field_metadata[key] = {
            'wavelength': wavelength,
            'geometry_hash': geometry_hash,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.put(key, field, priority=self._calculate_field_priority(metadata))
        
    def get_field(self, wavelength: float, geometry_hash: str,
                  tolerance: float = 1e-9) -> Optional[torch.Tensor]:
        """Get cached optical field with wavelength tolerance."""
        
        # First try exact match
        key = self._generate_field_key(wavelength, geometry_hash)
        field = self.get(key)
        if field is not None:
            return field
            
        # Search for nearby wavelengths within tolerance
        for cached_key, metadata in self._field_metadata.items():
            if (metadata['geometry_hash'] == geometry_hash and
                abs(metadata['wavelength'] - wavelength) <= tolerance):
                
                cached_field = self.get(cached_key)
                if cached_field is not None:
                    # Optionally interpolate for different wavelength
                    return self._interpolate_wavelength(
                        cached_field, metadata['wavelength'], wavelength
                    )
                    
        return None
        
    def _generate_field_key(self, wavelength: float, geometry_hash: str) -> str:
        """Generate cache key for optical field."""
        return f"field_{wavelength:.12e}_{geometry_hash}"
        
    def _calculate_field_priority(self, metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate priority based on field characteristics."""
        if not metadata:
            return 1.0
            
        priority = 1.0
        
        # Higher priority for computationally expensive fields
        if metadata.get('computation_time', 0) > 1.0:  # > 1 second
            priority *= 2.0
            
        # Higher priority for high-resolution fields
        if metadata.get('resolution', 0) > 1000000:  # > 1M points
            priority *= 1.5
            
        # Higher priority for 3D fields
        if metadata.get('dimensions', 2) >= 3:
            priority *= 1.3
            
        return priority
        
    def _interpolate_wavelength(self, field: torch.Tensor, 
                              cached_wavelength: float, 
                              target_wavelength: float) -> torch.Tensor:
        """Simple wavelength interpolation for optical fields."""
        
        # For small wavelength differences, apply phase correction
        wavelength_ratio = target_wavelength / cached_wavelength
        
        if torch.is_complex(field):
            # Apply phase shift for propagation
            phase_shift = 2 * np.pi * (1 - wavelength_ratio) 
            correction = torch.exp(1j * phase_shift)
            return field * correction
        else:
            # For intensity fields, assume wavelength independence for small changes
            return field
            
    def cleanup_expired(self, max_age: float = 7200):  # 2 hours
        """Remove expired field entries."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, metadata in self._field_metadata.items():
                if current_time - metadata['timestamp'] > max_age:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                self._evict(key)
                if key in self._field_metadata:
                    del self._field_metadata[key]


class ResultCache:
    """Cache for simulation results with automatic invalidation."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._cache = {}
        self._dependencies = {}  # key -> set of dependency hashes
        self._lock = threading.RLock()
        
    def memoize(self, dependencies: Optional[list] = None):
        """Decorator for memoizing function results."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, args, kwargs)
                
                # Check cache
                with self._lock:
                    if key in self._cache:
                        result, dep_hashes = self._cache[key]
                        
                        # Check if dependencies are still valid
                        if self._dependencies_valid(dep_hashes, dependencies):
                            return result
                        else:
                            # Invalidate stale result
                            del self._cache[key]
                            if key in self._dependencies:
                                del self._dependencies[key]
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result with dependencies
                dep_hashes = self._hash_dependencies(dependencies)
                self._store_result(key, result, dep_hashes)
                
                return result
                
            return wrapper
        return decorator
        
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature."""
        # Create deterministic hash from arguments
        key_data = (func_name, args, tuple(sorted(kwargs.items())))
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()
        
    def _hash_dependencies(self, dependencies: Optional[list]) -> Dict[str, str]:
        """Hash dependency objects."""
        if not dependencies:
            return {}
            
        dep_hashes = {}
        for i, dep in enumerate(dependencies):
            if hasattr(dep, '__dict__'):
                # Hash object state
                dep_str = str(sorted(dep.__dict__.items()))
            else:
                # Hash value directly
                dep_str = str(dep)
                
            dep_hashes[f"dep_{i}"] = hashlib.sha256(dep_str.encode()).hexdigest()
            
        return dep_hashes
        
    def _dependencies_valid(self, cached_hashes: Dict[str, str], 
                          current_deps: Optional[list]) -> bool:
        """Check if dependencies are still valid."""
        if not cached_hashes and not current_deps:
            return True
            
        current_hashes = self._hash_dependencies(current_deps)
        return cached_hashes == current_hashes
        
    def _store_result(self, key: str, result: Any, dep_hashes: Dict[str, str]):
        """Store result in cache with capacity management."""
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._dependencies:
                    del self._dependencies[oldest_key]
                    
            self._cache[key] = (result, dep_hashes)
            self._dependencies[key] = dep_hashes
            
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern is None:
                self._cache.clear()
                self._dependencies.clear()
            else:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._cache[key]
                    if key in self._dependencies:
                        del self._dependencies[key]
                        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size
            }


# Global cache instances
_tensor_cache = TensorCache()
_field_cache = OpticalFieldCache()
_result_cache = ResultCache()


def get_tensor_cache() -> TensorCache:
    """Get global tensor cache instance."""
    return _tensor_cache


def get_field_cache() -> OpticalFieldCache:
    """Get global optical field cache instance."""
    return _field_cache


def get_result_cache() -> ResultCache:
    """Get global result cache instance."""
    return _result_cache


def cache_tensor(key: str, tensor: torch.Tensor, priority: float = 1.0):
    """Cache a tensor globally."""
    _tensor_cache.put(key, tensor, priority)


def get_cached_tensor(key: str) -> Optional[torch.Tensor]:
    """Get cached tensor globally."""
    return _tensor_cache.get(key)


def memoize_result(dependencies: Optional[list] = None):
    """Decorator for memoizing expensive computations."""
    return _result_cache.memoize(dependencies)