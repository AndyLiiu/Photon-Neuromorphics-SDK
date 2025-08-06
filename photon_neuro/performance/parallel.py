"""
Parallel processing and GPU acceleration for photonic simulations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import concurrent.futures
import multiprocessing as mp
from functools import partial
import threading
import queue
import time

try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class ParallelSimulator:
    """Parallel execution of photonic simulations."""
    
    def __init__(self, n_workers: Optional[int] = None, use_gpu: bool = True):
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Worker pool
        self._executor = None
        self._gpu_pool = GPUPool() if self.use_gpu else None
        
    def __enter__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            
    def parallel_map(self, func: Callable, tasks: List[Any], 
                    batch_size: Optional[int] = None) -> List[Any]:
        """Execute function in parallel across tasks."""
        
        if not self._executor:
            raise RuntimeError("ParallelSimulator must be used as context manager")
            
        if batch_size is None:
            batch_size = max(1, len(tasks) // self.n_workers)
            
        # Split tasks into batches
        batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
        
        # Submit batch processing tasks
        futures = []
        for batch in batches:
            future = self._executor.submit(self._process_batch, func, batch)
            futures.append(future)
            
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing failed: {e}")
                continue
                
        return results
        
    def _process_batch(self, func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of tasks."""
        results = []
        
        # Get GPU context if available
        gpu_context = self._gpu_pool.acquire() if self._gpu_pool else None
        
        try:
            for task in batch:
                if gpu_context:
                    # Move tensors to GPU if needed
                    task = self._to_device(task, gpu_context.device)
                    
                result = func(task)
                
                # Move result back to CPU if needed
                if gpu_context and isinstance(result, torch.Tensor):
                    result = result.cpu()
                    
                results.append(result)
                
        finally:
            if gpu_context:
                self._gpu_pool.release(gpu_context)
                
        return results
        
    def _to_device(self, data: Any, device: torch.device) -> Any:
        """Move data to specified device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self._to_device(item, device) for item in data)
        elif isinstance(data, dict):
            return {key: self._to_device(value, device) for key, value in data.items()}
        else:
            return data
            
    def frequency_sweep_parallel(self, component, frequency_range: np.ndarray,
                                n_parallel: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel frequency sweep of photonic component."""
        
        n_parallel = n_parallel or self.n_workers
        
        # Split frequency range
        freq_batches = np.array_split(frequency_range, n_parallel)
        
        def sweep_batch(freq_batch):
            transmission = []
            phase = []
            
            for freq in freq_batch:
                # Set component frequency
                if hasattr(component, 'wavelength'):
                    component.wavelength = 3e8 / freq
                    
                # Simulate
                input_field = torch.ones(1, dtype=torch.complex64, device=self.device)
                output_field = component.forward(input_field)
                
                transmission.append(torch.abs(output_field).item())
                phase.append(torch.angle(output_field).item())
                
            return np.array(transmission), np.array(phase)
            
        # Execute in parallel
        results = self.parallel_map(sweep_batch, freq_batches)
        
        # Combine results
        all_transmission = []
        all_phase = []
        
        for transmission, phase in results:
            all_transmission.extend(transmission)
            all_phase.extend(phase)
            
        return np.array(all_transmission), np.array(all_phase)


class GPUPool:
    """Pool of GPU contexts for parallel processing."""
    
    def __init__(self):
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._contexts = queue.Queue()
        self._lock = threading.Lock()
        
        # Initialize GPU contexts
        for gpu_id in range(self.n_gpus):
            context = GPUContext(gpu_id)
            self._contexts.put(context)
            
    def acquire(self, timeout: float = 30.0) -> Optional['GPUContext']:
        """Acquire GPU context."""
        try:
            return self._contexts.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def release(self, context: 'GPUContext'):
        """Release GPU context back to pool."""
        self._contexts.put(context)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get GPU pool statistics."""
        return {
            'total_gpus': self.n_gpus,
            'available_contexts': self._contexts.qsize(),
            'memory_usage': self._get_memory_usage()
        }
        
    def _get_memory_usage(self) -> List[Dict[str, float]]:
        """Get memory usage for all GPUs."""
        memory_info = []
        
        for gpu_id in range(self.n_gpus):
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
            
            memory_info.append({
                'gpu_id': gpu_id,
                'allocated_gb': allocated,
                'cached_gb': cached
            })
            
        return memory_info


class GPUContext:
    """Context for GPU operations."""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self._streams = {}
        
    def get_stream(self, name: str = 'default') -> torch.cuda.Stream:
        """Get or create CUDA stream."""
        if name not in self._streams:
            with torch.cuda.device(self.device):
                self._streams[name] = torch.cuda.Stream()
        return self._streams[name]
        
    def clear_cache(self):
        """Clear GPU memory cache."""
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()


class GPUAccelerator:
    """GPU acceleration utilities for photonic computations."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_mixed_precision = torch.cuda.is_available()
        
        # Create CUDA streams for overlapping computation
        if self.device.type == 'cuda':
            self.compute_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.memory_stream = None
            
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for GPU execution."""
        model = model.to(self.device)
        
        # Enable mixed precision if available
        if self.use_mixed_precision:
            model = model.half()
            
        # Compile with TorchScript if possible
        try:
            model = torch.jit.script(model)
        except Exception:
            print("TorchScript compilation failed, using eager mode")
            
        return model
        
    def batch_fft(self, signals: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """GPU-accelerated batch FFT."""
        signals = signals.to(self.device)
        
        with torch.cuda.stream(self.compute_stream) if self.compute_stream else torch.no_grad():
            return torch.fft.fft(signals, dim=dim)
            
    def parallel_matrix_multiply(self, matrices: List[torch.Tensor]) -> torch.Tensor:
        """Parallel matrix multiplication on GPU."""
        if not matrices:
            raise ValueError("No matrices provided")
            
        # Stack matrices for batch processing
        batch_matrices = torch.stack([m.to(self.device) for m in matrices])
        
        with torch.cuda.stream(self.compute_stream) if self.compute_stream else torch.no_grad():
            # Batch matrix multiplication
            result = batch_matrices[0]
            for i in range(1, len(batch_matrices)):
                result = torch.bmm(result.unsqueeze(0), batch_matrices[i].unsqueeze(0)).squeeze(0)
                
        return result
        
    def async_data_transfer(self, data: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """Asynchronous data transfer to GPU."""
        if self.memory_stream:
            with torch.cuda.stream(self.memory_stream):
                return data.to(self.device, non_blocking=non_blocking)
        else:
            return data.to(self.device)
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if self.device.type != 'cuda':
            return {'allocated_gb': 0, 'cached_gb': 0}
            
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        cached = torch.cuda.memory_reserved(self.device) / 1024**3
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'device': str(self.device)
        }


class DistributedTraining:
    """Distributed training for large photonic models."""
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.initialized = False
        self.rank = 0
        self.world_size = 1
        
    def initialize(self, rank: int, world_size: int, master_addr: str = 'localhost',
                  master_port: str = '12355'):
        """Initialize distributed training."""
        if not DISTRIBUTED_AVAILABLE:
            print("PyTorch distributed not available")
            return False
            
        try:
            import os
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
            
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size
            )
            
            self.rank = rank
            self.world_size = world_size
            self.initialized = True
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            return False
            
    def cleanup(self):
        """Cleanup distributed training."""
        if self.initialized and DISTRIBUTED_AVAILABLE:
            dist.destroy_process_group()
            self.initialized = False
            
    def all_reduce(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """All-reduce operation across all processes."""
        if not self.initialized:
            return tensor
            
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        if average:
            tensor /= self.world_size
            
        return tensor
        
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks."""
        if not self.initialized:
            return tensor
            
        dist.broadcast(tensor, src=src)
        return tensor
        
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
        """Gather tensors from all ranks to destination rank."""
        if not self.initialized:
            return [tensor] if self.rank == dst else None
            
        if self.rank == dst:
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gather_list, dst=dst)
            return gather_list
        else:
            dist.gather(tensor, dst=dst)
            return None
            
    def scatter(self, tensor_list: Optional[List[torch.Tensor]], src: int = 0) -> torch.Tensor:
        """Scatter tensors from source rank to all ranks."""
        if not self.initialized:
            return tensor_list[0] if tensor_list else torch.tensor([])
            
        if self.rank == src and tensor_list:
            output_tensor = torch.zeros_like(tensor_list[0])
            dist.scatter(output_tensor, tensor_list, src=src)
        else:
            # Create dummy tensor with correct shape
            output_tensor = torch.zeros(1)  # Will be overwritten
            dist.scatter(output_tensor, src=src)
            
        return output_tensor
        
    def barrier(self):
        """Synchronization barrier."""
        if self.initialized:
            dist.barrier()
            
    def get_rank(self) -> int:
        """Get current process rank."""
        return self.rank
        
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.world_size
        
    def is_master(self) -> bool:
        """Check if current process is master (rank 0)."""
        return self.rank == 0


class AsyncSimulator:
    """Asynchronous simulation execution."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._futures = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        
    def submit_simulation(self, sim_id: str, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit simulation for asynchronous execution."""
        def wrapped_func():
            with self._semaphore:
                return func(*args, **kwargs)
                
        future = self._executor.submit(wrapped_func)
        self._futures[sim_id] = future
        return future
        
    def get_result(self, sim_id: str, timeout: Optional[float] = None) -> Any:
        """Get simulation result."""
        if sim_id not in self._futures:
            raise ValueError(f"Simulation {sim_id} not found")
            
        future = self._futures[sim_id]
        try:
            result = future.result(timeout=timeout)
            del self._futures[sim_id]  # Clean up completed simulation
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Simulation {sim_id} timed out")
            
    def cancel_simulation(self, sim_id: str) -> bool:
        """Cancel running simulation."""
        if sim_id not in self._futures:
            return False
            
        future = self._futures[sim_id]
        cancelled = future.cancel()
        
        if cancelled:
            del self._futures[sim_id]
            
        return cancelled
        
    def get_status(self) -> Dict[str, str]:
        """Get status of all simulations."""
        status = {}
        
        for sim_id, future in self._futures.items():
            if future.done():
                status[sim_id] = 'completed' if not future.exception() else 'failed'
            elif future.cancelled():
                status[sim_id] = 'cancelled'
            else:
                status[sim_id] = 'running'
                
        return status
        
    def shutdown(self, wait: bool = True):
        """Shutdown the async simulator."""
        self._executor.shutdown(wait=wait)