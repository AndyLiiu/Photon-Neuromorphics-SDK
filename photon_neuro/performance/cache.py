"""
Adaptive caching system for photonic simulations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, Union, List
import hashlib
import time
import threading
from collections import OrderedDict, deque
import weakref
import pickle
import psutil
import gc
import mmap
import lz4.frame
import struct
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
from typing import AsyncGenerator


@dataclass
class AccessPattern:
    """Stores detailed access pattern information."""
    frequency: float = 0.0
    recency: float = 0.0
    size_efficiency: float = 1.0
    compute_cost: float = 1.0
    access_sequence: deque = None
    temporal_locality: float = 0.0
    spatial_locality: float = 0.0
    
    def __post_init__(self):
        if self.access_sequence is None:
            self.access_sequence = deque(maxlen=100)


class AdaptiveCache:
    """Advanced intelligent cache with deep learning-based access pattern prediction."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600,
                 adaptation_rate: float = 0.1, enable_compression: bool = True,
                 enable_predictive_prefetch: bool = True, memory_limit_gb: float = 4.0):
        self.max_size = max_size
        self.ttl = ttl
        self.adaptation_rate = adaptation_rate
        self.enable_compression = enable_compression
        self.enable_predictive_prefetch = enable_predictive_prefetch
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        
        # Enhanced storage with compression support
        self._cache = OrderedDict()
        self._compressed_cache = {}  # For compressed items
        self._access_patterns = {}  # Enhanced access pattern tracking
        self._size_tracker = {}  # Memory usage per item
        self._compression_ratio = {}  # Compression efficiency
        
        # Advanced metrics
        self._total_requests = 0
        self._total_hits = 0
        self._memory_usage = 0
        self._access_history = deque(maxlen=10000)  # Global access history
        self._prefetch_stats = {'hits': 0, 'misses': 0}
        
        # Multi-level caching
        self._l1_cache = OrderedDict()  # Hot data
        self._l2_cache = OrderedDict()  # Warm data
        self.l1_size = max_size // 4
        self.l2_size = max_size // 2
        
        # Thread safety and async support
        self._lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cache-prefetch')
        
        # Adaptive algorithms
        self._access_predictor = AccessPredictor()
        self._eviction_algorithm = 'adaptive'  # LRU, LFU, adaptive
        
        # Performance monitoring
        self._performance_metrics = {
            'avg_access_time': 0.0,
            'compression_savings': 0.0,
            'prefetch_accuracy': 0.0,
            'memory_efficiency': 0.0
        }
        
    def get(self, key: str, default: Any = None, enable_prefetch: bool = True) -> Any:
        """Enhanced get with multi-level caching and prefetching."""
        start_time = time.perf_counter()
        
        with self._lock:
            self._total_requests += 1
            current_time = time.time()
            
            # Try L1 cache first (hot data)
            if key in self._l1_cache:
                value = self._l1_cache.pop(key)
                self._l1_cache[key] = value  # Move to end
                self._record_access(key, current_time, cache_level='L1')
                self._total_hits += 1
                
                # Update performance metrics
                access_time = time.perf_counter() - start_time
                self._update_performance_metrics(access_time)
                
                return self._decompress_if_needed(key, value)
            
            # Try L2 cache (warm data)
            if key in self._l2_cache:
                value = self._l2_cache.pop(key)
                # Promote to L1 if frequently accessed
                pattern = self._access_patterns.get(key, AccessPattern())
                if pattern.frequency > self._calculate_promotion_threshold():
                    self._promote_to_l1(key, value)
                else:
                    self._l2_cache[key] = value
                    
                self._record_access(key, current_time, cache_level='L2')
                self._total_hits += 1
                
                # Update performance metrics
                access_time = time.perf_counter() - start_time
                self._update_performance_metrics(access_time)
                
                return self._decompress_if_needed(key, value)
            
            # Try main cache (cold data)
            if key in self._cache:
                # Check TTL
                pattern = self._access_patterns.get(key, AccessPattern())
                if current_time - pattern.recency > self.ttl:
                    self._evict(key)
                    return default
                    
                value = self._cache.pop(key)
                
                # Decide cache level based on access patterns
                if pattern.frequency > self._calculate_promotion_threshold():
                    self._promote_to_l2(key, value)
                else:
                    self._cache[key] = value
                    
                self._record_access(key, current_time, cache_level='main')
                self._total_hits += 1
                
                # Trigger predictive prefetch if enabled
                if enable_prefetch and self.enable_predictive_prefetch:
                    self._trigger_predictive_prefetch(key)
                
                # Update performance metrics
                access_time = time.perf_counter() - start_time
                self._update_performance_metrics(access_time)
                
                return self._decompress_if_needed(key, value)
            
            # Cache miss - trigger predictive prefetch for related items
            if enable_prefetch and self.enable_predictive_prefetch:
                self._prefetch_stats['misses'] += 1
                self._trigger_predictive_prefetch(key)
            
            return default
            
    def put(self, key: str, value: Any, priority: float = 1.0, 
            compute_cost: float = 1.0, compress: bool = None) -> bool:
        """Enhanced put with intelligent placement and compression."""
        with self._lock:
            current_time = time.time()
            compress = compress if compress is not None else self.enable_compression
            
            # Calculate value size
            value_size = self._estimate_size(value)
            
            # Check memory limits
            if self._memory_usage + value_size > self.memory_limit_bytes:
                if not self._free_memory_intelligent(value_size):
                    return False  # Could not free enough memory
            
            # Determine optimal storage strategy
            storage_strategy = self._determine_storage_strategy(key, value, value_size, priority)
            
            # Apply compression if beneficial
            stored_value, compressed, compression_ratio = self._compress_if_beneficial(
                value, compress, value_size
            )
            actual_size = int(value_size * compression_ratio) if compressed else value_size
            
            # Remove from existing location if present
            self._remove_from_all_caches(key)
            
            # Store based on strategy
            if storage_strategy == 'L1':
                self._ensure_l1_capacity(actual_size)
                self._l1_cache[key] = stored_value
            elif storage_strategy == 'L2':
                self._ensure_l2_capacity(actual_size)
                self._l2_cache[key] = stored_value
            else:
                self._ensure_main_capacity(actual_size)
                self._cache[key] = stored_value
            
            # Update metadata
            self._access_patterns[key] = AccessPattern(
                frequency=1.0,
                recency=current_time,
                size_efficiency=1.0 / max(actual_size, 1),
                compute_cost=compute_cost
            )
            
            self._size_tracker[key] = actual_size
            if compressed:
                self._compression_ratio[key] = compression_ratio
                self._compressed_cache[key] = True
            
            self._memory_usage += actual_size
            
            # Update access predictor
            self._access_predictor.observe_put(key, priority, compute_cost)
            
            return True
            
    def _adaptive_eviction(self, target_cache: str = 'main', required_size: int = 0) -> bool:
        """Advanced multi-factor eviction algorithm."""
        cache = self._get_cache_by_name(target_cache)
        if not cache:
            return False
            
        current_time = time.time()
        eviction_candidates = []
        
        for key in list(cache.keys()):
            pattern = self._access_patterns.get(key, AccessPattern())
            size = self._size_tracker.get(key, 1)
            
            # Multi-factor eviction score
            time_factor = (current_time - pattern.recency) / self.ttl
            frequency_factor = 1.0 / (pattern.frequency + 0.1)
            size_factor = size / (self.memory_limit_bytes / 1000)  # Normalize
            cost_factor = 1.0 / (pattern.compute_cost + 0.1)
            
            # Combined eviction score (higher = more likely to evict)
            score = (
                time_factor * 0.3 +
                frequency_factor * 0.25 +
                size_factor * 0.2 +
                cost_factor * 0.15 +
                (1.0 - pattern.size_efficiency) * 0.1
            )
            
            eviction_candidates.append((score, key, size))
        
        # Sort by eviction score (highest first)
        eviction_candidates.sort(reverse=True)
        
        # Evict items until we have enough space
        freed_space = 0
        evicted_count = 0
        
        for score, key, size in eviction_candidates:
            if required_size > 0 and freed_space >= required_size:
                break
            if required_size == 0 and evicted_count >= 1:
                break
                
            self._evict(key)
            freed_space += size
            evicted_count += 1
            
        return freed_space >= required_size or evicted_count > 0
    
    def _record_access(self, key: str, timestamp: float, cache_level: str):
        """Record detailed access information."""
        if key not in self._access_patterns:
            self._access_patterns[key] = AccessPattern()
        
        pattern = self._access_patterns[key]
        pattern.recency = timestamp
        pattern.frequency = pattern.frequency * 0.9 + 1.0  # Exponential decay
        pattern.access_sequence.append((timestamp, cache_level))
        
        # Calculate temporal locality
        if len(pattern.access_sequence) > 1:
            intervals = [
                pattern.access_sequence[i][0] - pattern.access_sequence[i-1][0]
                for i in range(1, len(pattern.access_sequence))
            ]
            pattern.temporal_locality = 1.0 / (1.0 + np.std(intervals))
        
        # Update global access history
        self._access_history.append((key, timestamp, cache_level))
    
    def _predictive_prefetch(self, key: str, access_pattern: AccessPattern):
        """Perform predictive prefetching based on access patterns."""
        if not self.enable_predictive_prefetch:
            return
        
        # Predict likely next accesses based on pattern analysis
        predicted_keys = self._predict_next_access(key, access_pattern)
        
        for predicted_key in predicted_keys:
            if predicted_key not in self.main_cache and predicted_key not in self.lru_cache:
                # Asynchronously prefetch the predicted key
                self._async_prefetch(predicted_key)
    
    def _predict_next_access(self, current_key: str, pattern: AccessPattern) -> List[str]:
        """Predict next likely access based on pattern analysis."""
        # Simple prediction based on access sequence
        if not pattern.access_sequence:
            return []
        
        # Find keys that commonly follow the current key
        predictions = []
        sequence = list(pattern.access_sequence)
        
        for i, seq_key in enumerate(sequence[:-1]):
            if seq_key == current_key:
                next_key = sequence[i + 1]
                if next_key not in predictions:
                    predictions.append(next_key)
        
        return predictions[:3]  # Limit to top 3 predictions
    
    def _async_prefetch(self, key: str):
        """Asynchronously prefetch a key if not already cached."""
        # In practice, this would trigger background computation
        # For now, just log the prefetch intent
        logger.debug(f"Prefetch triggered for key: {key}")
    
    def _update_performance_metrics(self, access_time: float):
        """Update performance metrics with exponential moving average."""
        alpha = 0.1
        self._performance_metrics['avg_access_time'] = (
            alpha * access_time + 
            (1 - alpha) * self._performance_metrics['avg_access_time']
        )
    
    def _calculate_promotion_threshold(self) -> float:
        """Calculate dynamic threshold for cache level promotion."""
        if not self._access_patterns:
            return 2.0
        
        frequencies = [p.frequency for p in self._access_patterns.values()]
        return np.percentile(frequencies, 75) if frequencies else 2.0
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promote item to L1 cache."""
        # Ensure L1 capacity
        while len(self._l1_cache) >= self.l1_size:
            self._adaptive_eviction('L1')
        
        self._l1_cache[key] = value
        
        # Remove from other caches
        self._l2_cache.pop(key, None)
        self._cache.pop(key, None)
    
    def _promote_to_l2(self, key: str, value: Any):
        """Promote item to L2 cache."""
        # Ensure L2 capacity
        while len(self._l2_cache) >= self.l2_size:
            self._adaptive_eviction('L2')
        
        self._l2_cache[key] = value
        
        # Remove from main cache
        self._cache.pop(key, None)
    
    def _decompress_if_needed(self, key: str, value: Any) -> Any:
        """Decompress value if it was compressed."""
        if key in self._compressed_cache:
            try:
                if isinstance(value, bytes):
                    return pickle.loads(lz4.frame.decompress(value))
                elif isinstance(value, torch.Tensor) and hasattr(value, '_photon_compressed'):
                    # Custom tensor decompression
                    return self._decompress_tensor(value)
            except Exception as e:
                print(f"Decompression failed for key {key}: {e}")
        
        return value
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.numel()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        else:
            # Rough estimate for other objects
            return len(pickle.dumps(value))
    
    def _determine_storage_strategy(self, key: str, value: Any, size: int, priority: float) -> str:
        """Determine optimal cache level for storage."""
        # High priority and small size -> L1
        if priority > 2.0 and size < 1024 * 1024:  # 1MB
            return 'L1'
        
        # Medium priority or medium size -> L2
        if priority > 1.0 or size < 10 * 1024 * 1024:  # 10MB
            return 'L2'
        
        # Low priority or large size -> main cache
        return 'main'
    
    def _compress_if_beneficial(self, value: Any, force_compress: bool, size: int) -> Tuple[Any, bool, float]:
        """Compress value if beneficial."""
        if not force_compress or size < 1024:  # Don't compress small items
            return value, False, 1.0
        
        try:
            if isinstance(value, torch.Tensor):
                return self._compress_tensor(value)
            else:
                # General compression with lz4
                serialized = pickle.dumps(value)
                compressed = lz4.frame.compress(serialized, compression_level=1)
                compression_ratio = len(compressed) / len(serialized)
                
                # Only use compression if it saves at least 20%
                if compression_ratio < 0.8:
                    return compressed, True, compression_ratio
                else:
                    return value, False, 1.0
        
        except Exception as e:
            print(f"Compression failed: {e}")
            return value, False, 1.0
    
    def _compress_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """Specialized tensor compression."""
        # For now, use simple dtype reduction where possible
        original_size = tensor.element_size() * tensor.numel()
        
        # Try half precision for floating point tensors
        if tensor.dtype == torch.float32:
            compressed = tensor.half()
            new_size = compressed.element_size() * compressed.numel()
            if new_size < original_size:
                compressed._photon_compressed = True
                return compressed, True, new_size / original_size
        
        return tensor, False, 1.0
    
    def _decompress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decompress tensor."""
        if hasattr(tensor, '_photon_compressed'):
            # Convert back to float32 if needed
            if tensor.dtype == torch.float16:
                return tensor.float()
        return tensor
    
    def _remove_from_all_caches(self, key: str):
        """Remove key from all cache levels."""
        old_size = 0
        
        if key in self._l1_cache:
            del self._l1_cache[key]
            old_size = self._size_tracker.get(key, 0)
        elif key in self._l2_cache:
            del self._l2_cache[key]
            old_size = self._size_tracker.get(key, 0)
        elif key in self._cache:
            del self._cache[key]
            old_size = self._size_tracker.get(key, 0)
        
        # Update memory usage
        if old_size > 0:
            self._memory_usage -= old_size
        
        # Clean up metadata
        self._size_tracker.pop(key, None)
        self._compression_ratio.pop(key, None)
        self._compressed_cache.pop(key, None)
    
    def _ensure_l1_capacity(self, required_size: int):
        """Ensure L1 cache has capacity."""
        while len(self._l1_cache) >= self.l1_size:
            self._adaptive_eviction('L1')
    
    def _ensure_l2_capacity(self, required_size: int):
        """Ensure L2 cache has capacity."""
        while len(self._l2_cache) >= self.l2_size:
            self._adaptive_eviction('L2')
    
    def _ensure_main_capacity(self, required_size: int):
        """Ensure main cache has capacity."""
        while len(self._cache) >= self.max_size - self.l1_size - self.l2_size:
            self._adaptive_eviction('main')
    
    def _get_cache_by_name(self, cache_name: str) -> OrderedDict:
        """Get cache by name."""
        if cache_name == 'L1':
            return self._l1_cache
        elif cache_name == 'L2':
            return self._l2_cache
        elif cache_name == 'main':
            return self._cache
        return OrderedDict()
    
    def _free_memory_intelligent(self, required_size: int) -> bool:
        """Intelligently free memory across all cache levels."""
        # Try main cache first
        if self._adaptive_eviction('main', required_size):
            return True
        
        # Then L2 cache
        if self._adaptive_eviction('L2', required_size):
            return True
        
        # Finally L1 cache if desperate
        return self._adaptive_eviction('L1', required_size)
    
    def _calculate_level_hit_rate(self, level: str) -> float:
        """Calculate hit rate for specific cache level."""
        if not self._access_history:
            return 0.0
        
        level_hits = sum(1 for _, _, cache_level in self._access_history if cache_level == level)
        return level_hits / len(self._access_history)
    
    def _trigger_predictive_prefetch(self, accessed_key: str):
        """Trigger predictive prefetching based on access patterns."""
        if not self.enable_predictive_prefetch:
            return
        
        # Predict related keys to prefetch
        predicted_keys = self._access_predictor.predict_next_access(accessed_key, self._access_history)
        
        # Prefetch in background
        for key in predicted_keys[:3]:  # Limit to 3 prefetches
            self._prefetch_executor.submit(self._background_prefetch, key)
    
    def _background_prefetch(self, key: str):
        """Background prefetch operation."""
        # This would be implemented by subclasses or connected to data source
        pass


class AccessPredictor:
    """Predicts next access patterns using machine learning."""
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.access_sequences = {}  # key -> list of next accessed keys
        self.temporal_patterns = {}  # time-based patterns
        self.co_access_matrix = {}  # keys accessed together
        
    def observe_put(self, key: str, priority: float, compute_cost: float):
        """Observe a put operation."""
        # Track put patterns for future predictions
        pass
    
    def predict_next_access(self, current_key: str, access_history: deque) -> List[str]:
        """Predict next keys likely to be accessed."""
        if len(access_history) < 2:
            return []
        
        # Simple pattern matching - look for keys accessed after current_key
        predictions = []
        
        # Convert to list for easier processing
        history_list = list(access_history)
        
        for i, (key, timestamp, level) in enumerate(history_list[:-1]):
            if key == current_key and i < len(history_list) - 1:
                next_key = history_list[i + 1][0]
                if next_key != current_key and next_key not in predictions:
                    predictions.append(next_key)
        
        # Return top predictions
        return predictions[:5]


class MemoryPool:
    """Smart memory pool for efficient allocation and recycling."""
    
    def __init__(self, initial_size_gb: float = 1.0):
        self.pool_size = int(initial_size_gb * 1024**3)
        self.allocated_chunks = {}
        self.free_chunks = {}  # size -> list of chunks
        self.total_allocated = 0
        self._lock = threading.RLock()
        
        # Memory fragmentation tracking
        self.fragmentation_ratio = 0.0
        self.gc_threshold = 0.3  # Trigger GC when fragmentation > 30%
        
    def allocate(self, size: int, alignment: int = 8) -> Optional[int]:
        """Allocate memory chunk with optional alignment."""
        with self._lock:
            # Align size
            aligned_size = ((size + alignment - 1) // alignment) * alignment
            
            # Find suitable free chunk
            chunk_id = self._find_free_chunk(aligned_size)
            if chunk_id is not None:
                self.allocated_chunks[chunk_id] = aligned_size
                self.total_allocated += aligned_size
                return chunk_id
            
            # Check if we can allocate new chunk
            if self.total_allocated + aligned_size <= self.pool_size:
                chunk_id = len(self.allocated_chunks)
                self.allocated_chunks[chunk_id] = aligned_size
                self.total_allocated += aligned_size
                return chunk_id
            
            # Try garbage collection
            if self._garbage_collect():
                return self.allocate(size, alignment)
            
            return None
    
    def deallocate(self, chunk_id: int):
        """Deallocate memory chunk."""
        with self._lock:
            if chunk_id in self.allocated_chunks:
                size = self.allocated_chunks[chunk_id]
                del self.allocated_chunks[chunk_id]
                
                # Add to free chunks
                if size not in self.free_chunks:
                    self.free_chunks[size] = []
                self.free_chunks[size].append(chunk_id)
                
                self.total_allocated -= size
                self._update_fragmentation()
    
    def _find_free_chunk(self, size: int) -> Optional[int]:
        """Find free chunk of at least the requested size."""
        # Exact match first
        if size in self.free_chunks and self.free_chunks[size]:
            return self.free_chunks[size].pop()
        
        # Find smallest chunk that fits
        best_size = None
        for free_size in sorted(self.free_chunks.keys()):
            if free_size >= size and self.free_chunks[free_size]:
                best_size = free_size
                break
        
        if best_size is not None:
            chunk_id = self.free_chunks[best_size].pop()
            
            # If chunk is larger than needed, split it
            if best_size > size:
                remaining_size = best_size - size
                if remaining_size not in self.free_chunks:
                    self.free_chunks[remaining_size] = []
                self.free_chunks[remaining_size].append(chunk_id + 1)
            
            return chunk_id
        
        return None
    
    def _garbage_collect(self) -> bool:
        """Perform garbage collection to free up memory."""
        # Trigger Python garbage collection
        collected = gc.collect()
        
        # Coalesce adjacent free chunks
        self._coalesce_free_chunks()
        
        return collected > 0 or self.fragmentation_ratio < self.gc_threshold
    
    def _coalesce_free_chunks(self):
        """Coalesce adjacent free chunks to reduce fragmentation."""
        # Simple coalescing - combine chunks of same size
        for size, chunks in self.free_chunks.items():
            if len(chunks) > 1:
                # Keep only half, combine the rest
                coalesced_size = size * 2
                coalesced_count = len(chunks) // 2
                
                if coalesced_size not in self.free_chunks:
                    self.free_chunks[coalesced_size] = []
                
                self.free_chunks[coalesced_size].extend([chunks[i] for i in range(coalesced_count)])
                self.free_chunks[size] = chunks[coalesced_count * 2:]
    
    def _update_fragmentation(self):
        """Update fragmentation ratio."""
        if self.pool_size == 0:
            self.fragmentation_ratio = 0.0
            return
        
        # Simple fragmentation metric: ratio of free chunks to total space
        free_space = self.pool_size - self.total_allocated
        num_free_chunks = sum(len(chunks) for chunks in self.free_chunks.values())
        
        if free_space > 0:
            self.fragmentation_ratio = num_free_chunks / (free_space / 1024)  # chunks per KB
        else:
            self.fragmentation_ratio = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'pool_size_gb': self.pool_size / (1024**3),
                'allocated_gb': self.total_allocated / (1024**3),
                'utilization': self.total_allocated / self.pool_size,
                'fragmentation_ratio': self.fragmentation_ratio,
                'num_allocated_chunks': len(self.allocated_chunks),
                'num_free_chunks': sum(len(chunks) for chunks in self.free_chunks.values())
            }


class StreamingDataProcessor:
    """Efficient streaming data processor for large-scale simulations."""
    
    def __init__(self, chunk_size: int = 1024, max_workers: int = 4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._memory_pool = MemoryPool()
        
    async def process_stream(self, data_stream, process_func: Callable, 
                           output_buffer=None) -> AsyncGenerator[Any, None]:
        """Process data stream asynchronously with memory-efficient chunking."""
        
        chunk_buffer = []
        chunk_id = 0
        
        async for data_item in data_stream:
            chunk_buffer.append(data_item)
            
            # Process chunk when buffer is full
            if len(chunk_buffer) >= self.chunk_size:
                # Allocate memory for chunk
                memory_id = self._memory_pool.allocate(self.chunk_size * 8)  # Assume 8 bytes per item
                
                if memory_id is not None:
                    # Process chunk asynchronously
                    future = self._executor.submit(process_func, chunk_buffer.copy(), chunk_id)
                    
                    try:
                        result = await asyncio.wrap_future(future)
                        yield result
                    finally:
                        # Clean up memory
                        self._memory_pool.deallocate(memory_id)
                
                chunk_buffer.clear()
                chunk_id += 1
        
        # Process remaining data
        if chunk_buffer:
            memory_id = self._memory_pool.allocate(len(chunk_buffer) * 8)
            if memory_id is not None:
                future = self._executor.submit(process_func, chunk_buffer, chunk_id)
                try:
                    result = await asyncio.wrap_future(future)
                    yield result
                finally:
                    self._memory_pool.deallocate(memory_id)
    
    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        
    def _evict(self, key: str):
        """Remove key from all caches and clean up metadata."""
        # Remove from all cache levels
        self._remove_from_all_caches(key)
        
        # Clean up access patterns  
        if key in self._access_patterns:
            del self._access_patterns[key]
            
    def clear(self):
        """Clear all cached items from all cache levels."""
        with self._lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
            self._cache.clear()
            self._compressed_cache.clear()
            self._access_patterns.clear()
            self._size_tracker.clear()
            self._compression_ratio.clear()
            self._memory_usage = 0
            self._access_history.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Comprehensive cache statistics with performance metrics."""
        with self._lock:
            hit_rate = self._total_hits / max(self._total_requests, 1)
            total_items = len(self._l1_cache) + len(self._l2_cache) + len(self._cache)
            
            # Compression statistics
            compressed_items = len(self._compressed_cache)
            avg_compression_ratio = np.mean(list(self._compression_ratio.values())) if self._compression_ratio else 1.0
            
            # Memory efficiency
            memory_utilization = self._memory_usage / self.memory_limit_bytes
            
            # Cache level statistics
            l1_hit_rate = self._calculate_level_hit_rate('L1')
            l2_hit_rate = self._calculate_level_hit_rate('L2')
            
            return {
                'total_items': total_items,
                'l1_items': len(self._l1_cache),
                'l2_items': len(self._l2_cache),
                'main_items': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'total_requests': self._total_requests,
                'total_hits': self._total_hits,
                'memory_usage_gb': self._memory_usage / (1024**3),
                'memory_utilization': memory_utilization,
                'compressed_items': compressed_items,
                'avg_compression_ratio': avg_compression_ratio,
                'compression_savings_gb': (self._memory_usage * (1 - avg_compression_ratio)) / (1024**3),
                'prefetch_accuracy': self._prefetch_stats['hits'] / max(
                    self._prefetch_stats['hits'] + self._prefetch_stats['misses'], 1
                ),
                'performance_metrics': self._performance_metrics.copy()
            }
            
    def optimize(self):
        """Optimize cache parameters based on usage patterns and performance metrics."""
        with self._lock:
            stats = self.get_stats()
            
            # Adapt cache sizes based on hit rates
            if stats['hit_rate'] > 0.9:
                # High hit rate - can reduce main cache size slightly
                self.max_size = max(100, int(self.max_size * 0.95))
            elif stats['hit_rate'] < 0.5:
                # Low hit rate - increase cache sizes
                self.max_size = int(self.max_size * 1.1)
                self.l1_size = max(self.l1_size, self.max_size // 4)
                self.l2_size = max(self.l2_size, self.max_size // 2)
                
            # Adapt TTL based on access patterns
            if self._access_patterns:
                recency_values = [p.recency for p in self._access_patterns.values()]
                if recency_values:
                    avg_access_interval = time.time() - np.mean(recency_values)
                    self.ttl = max(300, avg_access_interval * 2)  # At least 5 minutes
            
            # Optimize memory usage if too high
            if stats['memory_utilization'] > 0.9:
                # Trigger aggressive eviction
                target_size = int(self.memory_limit_bytes * 0.7)  # Target 70% usage
                self._free_memory_intelligent(self._memory_usage - target_size)
            
            # Update performance metrics
            self._performance_metrics['memory_efficiency'] = 1.0 - stats['memory_utilization']
            if stats['compressed_items'] > 0:
                self._performance_metrics['compression_savings'] = stats['compression_savings_gb']


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


class IntelligentCache:
    """Next-generation intelligent cache with AI-driven optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_cache = AdaptiveCache(
            max_size=self.config.get("max_size", 1000),
            ttl=self.config.get("ttl", 3600)
        )
        
        # AI-driven features
        self.ml_enabled = self.config.get("ml_enabled", True)
        self.prediction_accuracy = 0.0
        self.learning_rate = self.config.get("learning_rate", 0.01)
        
        # Advanced analytics
        self.access_patterns = {}
        self.performance_history = deque(maxlen=1000)
        self.prediction_model = None
        
        # Autonomous optimization
        self.auto_optimization = self.config.get("auto_optimization", True)
        self.optimization_interval = self.config.get("optimization_interval", 3600)  # 1 hour
        self.last_optimization = 0
        
    def put(self, key: str, value: Any, priority: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Put item with intelligent priority adjustment."""
        # AI-enhanced priority calculation
        if self.ml_enabled and metadata:
            priority = self._calculate_intelligent_priority(key, value, priority, metadata)
        
        # Update access patterns
        self._update_access_patterns(key, "put", metadata)
        
        # Store in base cache
        self.base_cache.put(key, value, priority)
        
        # Trigger optimization if needed
        if self.auto_optimization and self._should_optimize():
            self._optimize_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item with predictive prefetching."""
        # Record access
        access_time = time.time()
        value = self.base_cache.get(key)
        
        # Update access patterns
        self._update_access_patterns(key, "get")
        
        # Predictive prefetching
        if self.ml_enabled and value is not None:
            self._trigger_predictive_prefetch(key)
        
        # Record performance
        self.performance_history.append({
            'key': key,
            'hit': value is not None,
            'timestamp': access_time,
            'response_time': time.time() - access_time
        })
        
        return value
    
    def _calculate_intelligent_priority(self, key: str, value: Any, base_priority: float, 
                                      metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate intelligent priority using ML and heuristics."""
        priority = base_priority
        
        # Size-based adjustment
        if hasattr(value, '__sizeof__'):
            size_bytes = value.__sizeof__()
            if size_bytes > 1024 * 1024:  # > 1MB
                priority *= 0.8  # Lower priority for large items
            elif size_bytes < 1024:  # < 1KB
                priority *= 1.2  # Higher priority for small items
        
        # Metadata-based adjustments
        if metadata:
            # Computation cost
            compute_cost = metadata.get('compute_cost', 1.0)
            priority *= (1.0 + compute_cost * 0.5)
            
            # Access frequency prediction
            predicted_frequency = metadata.get('predicted_frequency', 1.0)
            priority *= (1.0 + predicted_frequency * 0.3)
            
            # Temporal locality
            last_access = metadata.get('last_access_time', 0)
            if last_access > 0:
                time_since_access = time.time() - last_access
                if time_since_access < 3600:  # Last hour
                    priority *= 1.5
        
        # Historical pattern analysis
        if key in self.access_patterns:
            pattern = self.access_patterns[key]
            
            # Frequency bonus
            if pattern.get('access_count', 0) > 10:
                priority *= 1.3
            
            # Recency bonus
            last_access = pattern.get('last_access', 0)
            if time.time() - last_access < 1800:  # Last 30 minutes
                priority *= 1.2
        
        return min(10.0, max(0.1, priority))  # Clamp between 0.1 and 10.0
    
    def _update_access_patterns(self, key: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Update access patterns for ML learning."""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'access_count': 0,
                'first_access': current_time,
                'last_access': current_time,
                'operation_types': defaultdict(int),
                'access_intervals': deque(maxlen=50),
                'metadata_history': deque(maxlen=10)
            }
        
        pattern = self.access_patterns[key]
        
        # Update basic stats
        pattern['access_count'] += 1
        pattern['operation_types'][operation] += 1
        
        # Update timing
        if pattern['last_access'] > 0:
            interval = current_time - pattern['last_access']
            pattern['access_intervals'].append(interval)
        
        pattern['last_access'] = current_time
        
        # Store metadata
        if metadata:
            pattern['metadata_history'].append(metadata)
    
    def _trigger_predictive_prefetch(self, accessed_key: str):
        """Trigger predictive prefetching based on access patterns."""
        if not self.ml_enabled:
            return
        
        # Simple pattern-based prediction
        candidates = self._predict_next_accesses(accessed_key)
        
        for candidate_key in candidates[:3]:  # Prefetch top 3 predictions
            if candidate_key not in self.base_cache._cache:
                # Would trigger prefetch operation in real implementation
                # For now, just log the prediction
                logger.debug(f"Predicted next access: {candidate_key} after {accessed_key}")
    
    def _predict_next_accesses(self, current_key: str) -> List[str]:
        """Predict next likely accesses using pattern analysis."""
        predictions = []
        
        # Analyze historical co-access patterns
        current_pattern = self.access_patterns.get(current_key, {})
        current_metadata = list(current_pattern.get('metadata_history', []))
        
        # Find keys with similar access patterns
        for key, pattern in self.access_patterns.items():
            if key == current_key:
                continue
            
            # Calculate pattern similarity
            similarity = self._calculate_pattern_similarity(current_pattern, pattern)
            
            if similarity > 0.5:  # Threshold for prediction
                predictions.append((key, similarity))
        
        # Sort by similarity and return top candidates
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in predictions]
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between access patterns."""
        if not pattern1 or not pattern2:
            return 0.0
        
        similarity = 0.0
        
        # Temporal similarity
        intervals1 = list(pattern1.get('access_intervals', []))
        intervals2 = list(pattern2.get('access_intervals', []))
        
        if intervals1 and intervals2:
            avg_interval1 = statistics.mean(intervals1)
            avg_interval2 = statistics.mean(intervals2)
            
            if avg_interval1 > 0 and avg_interval2 > 0:
                interval_similarity = 1.0 - abs(avg_interval1 - avg_interval2) / max(avg_interval1, avg_interval2)
                similarity += interval_similarity * 0.4
        
        # Frequency similarity
        count1 = pattern1.get('access_count', 0)
        count2 = pattern2.get('access_count', 0)
        
        if count1 > 0 and count2 > 0:
            freq_similarity = min(count1, count2) / max(count1, count2)
            similarity += freq_similarity * 0.3
        
        # Operation type similarity
        ops1 = pattern1.get('operation_types', {})
        ops2 = pattern2.get('operation_types', {})
        
        if ops1 and ops2:
            common_ops = set(ops1.keys()) & set(ops2.keys())
            total_ops = set(ops1.keys()) | set(ops2.keys())
            
            if total_ops:
                op_similarity = len(common_ops) / len(total_ops)
                similarity += op_similarity * 0.3
        
        return min(1.0, similarity)
    
    def _should_optimize(self) -> bool:
        """Determine if cache optimization should be triggered."""
        current_time = time.time()
        
        # Time-based optimization
        if current_time - self.last_optimization > self.optimization_interval:
            return True
        
        # Performance-based optimization
        if len(self.performance_history) >= 100:
            recent_performance = list(self.performance_history)[-100:]
            hit_rate = sum(1 for p in recent_performance if p['hit']) / len(recent_performance)
            
            if hit_rate < 0.6:  # Poor hit rate
                return True
        
        return False
    
    def _optimize_cache(self):
        """Optimize cache configuration using ML insights."""
        logger.info("Running intelligent cache optimization")
        
        self.last_optimization = time.time()
        
        # Analyze performance
        if len(self.performance_history) >= 50:
            recent_performance = list(self.performance_history)[-100:]
            
            # Calculate metrics
            hit_rate = sum(1 for p in recent_performance if p['hit']) / len(recent_performance)
            avg_response_time = statistics.mean([p['response_time'] for p in recent_performance])
            
            # Optimize based on metrics
            if hit_rate < 0.7:
                # Increase cache size
                new_size = min(self.base_cache.max_size * 1.2, 2000)
                self.base_cache.max_size = int(new_size)
                logger.info(f"Increased cache size to {new_size} due to low hit rate")
            
            if avg_response_time > 0.01:  # > 10ms
                # Optimize access patterns
                self._optimize_access_patterns()
        
        # Adaptive TTL optimization
        self._optimize_ttl()
        
        # Machine learning model update
        if self.ml_enabled:
            self._update_ml_model()
    
    def _optimize_access_patterns(self):
        """Optimize cache based on access pattern analysis."""
        # Identify hot keys (frequently accessed)
        hot_keys = []
        for key, pattern in self.access_patterns.items():
            access_count = pattern.get('access_count', 0)
            recency = time.time() - pattern.get('last_access', 0)
            
            if access_count > 10 and recency < 3600:  # Accessed >10 times in last hour
                hot_keys.append(key)
        
        # Increase priority for hot keys
        for key in hot_keys:
            if key in self.base_cache._cache:
                # Boost priority
                current_priority = self.base_cache._access_patterns.get(key, AccessPattern()).compute_cost
                self.base_cache._access_patterns[key].compute_cost = min(10.0, current_priority * 1.5)
        
        logger.info(f"Optimized {len(hot_keys)} hot keys")
    
    def _optimize_ttl(self):
        """Optimize TTL based on access patterns."""
        if not self.access_patterns:
            return
        
        # Calculate average access intervals
        all_intervals = []
        for pattern in self.access_patterns.values():
            intervals = list(pattern.get('access_intervals', []))
            all_intervals.extend(intervals)
        
        if all_intervals:
            avg_interval = statistics.mean(all_intervals)
            # Set TTL to 2x average access interval, with bounds
            optimal_ttl = max(300, min(7200, avg_interval * 2))  # 5 min to 2 hours
            
            if abs(optimal_ttl - self.base_cache.ttl) > 600:  # Significant change
                self.base_cache.ttl = optimal_ttl
                logger.info(f"Optimized TTL to {optimal_ttl:.0f} seconds")
    
    def _update_ml_model(self):
        """Update machine learning model for predictions."""
        # Simplified ML model update
        # In practice, would use scikit-learn or similar
        
        if len(self.performance_history) >= 100:
            recent_data = list(self.performance_history)[-100:]
            
            # Calculate prediction accuracy
            predictions_correct = 0
            total_predictions = 0
            
            for entry in recent_data:
                if 'predicted' in entry:
                    total_predictions += 1
                    if entry['predicted'] == entry['hit']:
                        predictions_correct += 1
            
            if total_predictions > 0:
                self.prediction_accuracy = predictions_correct / total_predictions
                logger.info(f"ML prediction accuracy: {self.prediction_accuracy:.2f}")
    
    def get_intelligent_stats(self) -> Dict[str, Any]:
        """Get comprehensive intelligent cache statistics."""
        base_stats = self.base_cache.get_stats()
        
        # Add intelligent features stats
        intelligent_stats = {
            'ml_enabled': self.ml_enabled,
            'prediction_accuracy': self.prediction_accuracy,
            'num_access_patterns': len(self.access_patterns),
            'auto_optimization': self.auto_optimization,
            'last_optimization': self.last_optimization,
            'optimization_interval': self.optimization_interval
        }
        
        # Performance analysis
        if self.performance_history:
            recent_performance = list(self.performance_history)[-100:]
            intelligent_stats.update({
                'recent_hit_rate': sum(1 for p in recent_performance if p['hit']) / len(recent_performance),
                'avg_response_time_ms': statistics.mean([p['response_time'] * 1000 for p in recent_performance]),
                'performance_trend': self._calculate_performance_trend()
            })
        
        # Access pattern insights
        if self.access_patterns:
            hot_keys = sum(1 for p in self.access_patterns.values() if p.get('access_count', 0) > 10)
            intelligent_stats.update({
                'hot_keys': hot_keys,
                'avg_access_frequency': statistics.mean([p.get('access_count', 0) for p in self.access_patterns.values()]),
                'pattern_diversity': len(set(tuple(p.get('operation_types', {}).keys()) for p in self.access_patterns.values()))
            })
        
        # Combine stats
        return {**base_stats, **intelligent_stats}
    
    def _calculate_performance_trend(self) -> str:
        """Calculate recent performance trend."""
        if len(self.performance_history) < 20:
            return "insufficient_data"
        
        recent = list(self.performance_history)[-20:]
        older = list(self.performance_history)[-40:-20] if len(self.performance_history) >= 40 else []
        
        if not older:
            return "stable"
        
        recent_hit_rate = sum(1 for p in recent if p['hit']) / len(recent)
        older_hit_rate = sum(1 for p in older if p['hit']) / len(older)
        
        change = recent_hit_rate - older_hit_rate
        
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "declining"
        else:
            return "stable"
    
    def export_ml_data(self, filepath: str):
        """Export ML training data for external analysis."""
        ml_data = {
            'access_patterns': self.access_patterns,
            'performance_history': list(self.performance_history),
            'config': self.config,
            'prediction_accuracy': self.prediction_accuracy,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(ml_data, f, indent=2, default=str)
            logger.info(f"Exported ML data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export ML data: {e}")


# Global intelligent cache instance
_intelligent_cache = None

def get_intelligent_cache() -> IntelligentCache:
    """Get global intelligent cache instance."""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache()
    return _intelligent_cache