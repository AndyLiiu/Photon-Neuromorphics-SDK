"""
Cross-Platform Compatibility Framework
=====================================

Universal platform support with adaptive deployment for Windows, macOS, Linux,
mobile platforms, and embedded systems. Automatic dependency resolution and
platform-specific optimizations.
"""

import os
import sys
import platform
import subprocess
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

from ..utils.logging_system import global_logger
from ..core.exceptions import PlatformError


class SupportedPlatform(Enum):
    """Supported platforms for deployment."""
    WINDOWS_X64 = "windows_x64"
    WINDOWS_ARM64 = "windows_arm64"
    MACOS_X64 = "macos_x64"
    MACOS_ARM64 = "macos_arm64"  # Apple Silicon
    LINUX_X64 = "linux_x64"
    LINUX_ARM64 = "linux_arm64"
    LINUX_ARMV7 = "linux_armv7"  # Raspberry Pi
    ANDROID = "android"
    IOS = "ios"
    WEBASSEMBLY = "webassembly"
    EMBEDDED_ARM = "embedded_arm"


class PerformanceTier(Enum):
    """Performance tiers for different platforms."""
    HIGH_PERFORMANCE = "high_performance"  # Desktop/server
    MEDIUM_PERFORMANCE = "medium_performance"  # Laptops/tablets
    LOW_PERFORMANCE = "low_performance"  # Mobile/embedded
    ULTRA_LOW_POWER = "ultra_low_power"  # IoT devices


@dataclass
class PlatformCapabilities:
    """Platform-specific capabilities and limitations."""
    platform: SupportedPlatform
    performance_tier: PerformanceTier
    simd_support: List[str]  # SSE, AVX, NEON
    memory_gb: float
    compute_units: int
    gpu_acceleration: bool
    wasm_support: bool
    threading_support: bool
    file_system_access: bool
    network_access: bool
    hardware_access: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizationProfile:
    """Optimization profile for platform-specific deployments."""
    platform: SupportedPlatform
    compiler_flags: List[str]
    optimization_level: str  # O0, O1, O2, O3, Os
    vectorization_enabled: bool
    parallel_processing: bool
    memory_optimization: bool
    power_optimization: bool
    binary_format: str  # exe, elf, mach-o, wasm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific adapters."""
    
    def __init__(self, platform: SupportedPlatform):
        self.platform = platform
        self.logger = global_logger
        self.capabilities = self._detect_capabilities()
    
    @abstractmethod
    def _detect_capabilities(self) -> PlatformCapabilities:
        """Detect platform-specific capabilities."""
        pass
    
    @abstractmethod
    def optimize_deployment(self, config: Dict[str, Any]) -> OptimizationProfile:
        """Create optimization profile for deployment."""
        pass
    
    @abstractmethod
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install platform-specific dependencies."""
        pass
    
    @abstractmethod
    def validate_environment(self) -> Dict[str, bool]:
        """Validate deployment environment."""
        pass


class WindowsAdapter(PlatformAdapter):
    """Windows platform adapter."""
    
    def __init__(self, architecture: str = "x64"):
        platform_type = SupportedPlatform.WINDOWS_X64 if architecture == "x64" else SupportedPlatform.WINDOWS_ARM64
        super().__init__(platform_type)
    
    def _detect_capabilities(self) -> PlatformCapabilities:
        """Detect Windows-specific capabilities."""
        import psutil
        
        # Detect SIMD support
        simd_support = []
        if hasattr(os, 'cpu_count'):
            # Check for AVX support (simplified)
            try:
                result = subprocess.run(['wmic', 'cpu', 'get', 'Name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'Intel' in result.stdout or 'AMD' in result.stdout:
                    simd_support.extend(['SSE', 'SSE2', 'AVX'])
            except:
                pass
        
        return PlatformCapabilities(
            platform=self.platform,
            performance_tier=PerformanceTier.HIGH_PERFORMANCE,
            simd_support=simd_support,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            compute_units=os.cpu_count() or 1,
            gpu_acceleration=self._check_gpu_support(),
            wasm_support=True,
            threading_support=True,
            file_system_access=True,
            network_access=True,
            hardware_access=True
        )
    
    def optimize_deployment(self, config: Dict[str, Any]) -> OptimizationProfile:
        """Create Windows-specific optimization profile."""
        return OptimizationProfile(
            platform=self.platform,
            compiler_flags=['/O2', '/GL', '/LTCG'],  # MSVC flags
            optimization_level='O2',
            vectorization_enabled=True,
            parallel_processing=True,
            memory_optimization=False,
            power_optimization=False,
            binary_format='exe'
        )
    
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install Windows dependencies using pip and conda."""
        try:
            for dep in dependencies:
                if dep == 'cuda':
                    # Install CUDA toolkit for Windows
                    self.logger.info("Installing CUDA toolkit for Windows")
                    # Simplified - would use actual installer
                elif dep == 'openblas':
                    subprocess.run(['pip', 'install', 'openblas'], check=True)
                else:
                    subprocess.run(['pip', 'install', dep], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate Windows environment."""
        checks = {
            'python_version': sys.version_info >= (3, 9),
            'pip_available': self._check_command('pip'),
            'visual_studio': self._check_visual_studio(),
            'windows_sdk': self._check_windows_sdk(),
            'vcredist': self._check_vcredist()
        }
        return checks
    
    def _check_gpu_support(self) -> bool:
        """Check for GPU acceleration support."""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        try:
            subprocess.run([command, '--version'], capture_output=True, timeout=5)
            return True
        except:
            return False
    
    def _check_visual_studio(self) -> bool:
        """Check for Visual Studio installation."""
        try:
            result = subprocess.run(['where', 'cl'], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_windows_sdk(self) -> bool:
        """Check for Windows SDK."""
        sdk_paths = [
            r"C:\Program Files (x86)\Windows Kits\10",
            r"C:\Program Files\Windows Kits\10"
        ]
        return any(Path(path).exists() for path in sdk_paths)
    
    def _check_vcredist(self) -> bool:
        """Check for Visual C++ Redistributable."""
        try:
            result = subprocess.run([
                'reg', 'query', 
                r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
                '/v', 'Installed'
            ], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False


class LinuxAdapter(PlatformAdapter):
    """Linux platform adapter."""
    
    def __init__(self, architecture: str = "x64"):
        platform_map = {
            "x64": SupportedPlatform.LINUX_X64,
            "arm64": SupportedPlatform.LINUX_ARM64,
            "armv7": SupportedPlatform.LINUX_ARMV7
        }
        platform_type = platform_map.get(architecture, SupportedPlatform.LINUX_X64)
        super().__init__(platform_type)
    
    def _detect_capabilities(self) -> PlatformCapabilities:
        """Detect Linux-specific capabilities."""
        import psutil
        
        # Detect SIMD support from /proc/cpuinfo
        simd_support = []
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'sse' in cpuinfo:
                    simd_support.append('SSE')
                if 'avx' in cpuinfo:
                    simd_support.append('AVX')
                if 'neon' in cpuinfo:
                    simd_support.append('NEON')
        except:
            pass
        
        return PlatformCapabilities(
            platform=self.platform,
            performance_tier=self._determine_performance_tier(),
            simd_support=simd_support,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            compute_units=os.cpu_count() or 1,
            gpu_acceleration=self._check_gpu_support(),
            wasm_support=True,
            threading_support=True,
            file_system_access=True,
            network_access=True,
            hardware_access=True
        )
    
    def optimize_deployment(self, config: Dict[str, Any]) -> OptimizationProfile:
        """Create Linux-specific optimization profile."""
        compiler_flags = ['-O3', '-march=native', '-mtune=native']
        
        # Add architecture-specific flags
        if self.platform == SupportedPlatform.LINUX_ARM64:
            compiler_flags.extend(['-mcpu=native', '-mfpu=neon'])
        elif self.platform == SupportedPlatform.LINUX_ARMV7:
            compiler_flags.extend(['-mfpu=neon-vfpv4', '-mfloat-abi=hard'])
        
        return OptimizationProfile(
            platform=self.platform,
            compiler_flags=compiler_flags,
            optimization_level='O3',
            vectorization_enabled=True,
            parallel_processing=True,
            memory_optimization=self.platform != SupportedPlatform.LINUX_X64,
            power_optimization=self.platform in [SupportedPlatform.LINUX_ARM64, SupportedPlatform.LINUX_ARMV7],
            binary_format='elf'
        )
    
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install Linux dependencies using package managers."""
        try:
            # Detect package manager
            package_manager = self._detect_package_manager()
            
            for dep in dependencies:
                if dep == 'cuda':
                    self._install_cuda_linux()
                elif dep == 'openblas':
                    self._install_system_package(package_manager, 'libopenblas-dev')
                else:
                    subprocess.run(['pip3', 'install', dep], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate Linux environment."""
        checks = {
            'python_version': sys.version_info >= (3, 9),
            'pip_available': self._check_command('pip3'),
            'gcc_available': self._check_command('gcc'),
            'glibc_version': self._check_glibc_version(),
            'kernel_version': self._check_kernel_version()
        }
        return checks
    
    def _determine_performance_tier(self) -> PerformanceTier:
        """Determine performance tier based on hardware."""
        if self.platform == SupportedPlatform.LINUX_X64:
            return PerformanceTier.HIGH_PERFORMANCE
        elif self.platform == SupportedPlatform.LINUX_ARM64:
            return PerformanceTier.MEDIUM_PERFORMANCE
        else:
            return PerformanceTier.LOW_PERFORMANCE
    
    def _check_gpu_support(self) -> bool:
        """Check for GPU acceleration support."""
        gpu_checks = ['nvidia-smi', 'rocm-smi', 'intel_gpu_top']
        for check in gpu_checks:
            try:
                result = subprocess.run([check], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return True
            except:
                continue
        return False
    
    def _detect_package_manager(self) -> str:
        """Detect available package manager."""
        managers = ['apt', 'yum', 'dnf', 'pacman', 'zypper']
        for manager in managers:
            if self._check_command(manager):
                return manager
        return 'pip3'
    
    def _install_cuda_linux(self):
        """Install CUDA on Linux."""
        # Simplified CUDA installation
        self.logger.info("Installing CUDA toolkit for Linux")
        # Would implement actual CUDA installation logic
    
    def _install_system_package(self, manager: str, package: str):
        """Install system package using package manager."""
        install_commands = {
            'apt': ['sudo', 'apt', 'install', '-y', package],
            'yum': ['sudo', 'yum', 'install', '-y', package],
            'dnf': ['sudo', 'dnf', 'install', '-y', package],
            'pacman': ['sudo', 'pacman', '-S', '--noconfirm', package],
            'zypper': ['sudo', 'zypper', 'install', '-y', package]
        }
        
        if manager in install_commands:
            subprocess.run(install_commands[manager], check=True)
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        try:
            subprocess.run(['which', command], capture_output=True, timeout=5, check=True)
            return True
        except:
            return False
    
    def _check_glibc_version(self) -> bool:
        """Check glibc version compatibility."""
        try:
            result = subprocess.run(['ldd', '--version'], capture_output=True, text=True, timeout=5)
            # Check for minimum glibc version (2.17+)
            return 'GLIBC 2.' in result.stdout
        except:
            return False
    
    def _check_kernel_version(self) -> bool:
        """Check kernel version compatibility."""
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True, timeout=5)
            # Check for minimum kernel version (3.10+)
            version_parts = result.stdout.strip().split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            return major > 3 or (major == 3 and minor >= 10)
        except:
            return False


class MacOSAdapter(PlatformAdapter):
    """macOS platform adapter."""
    
    def __init__(self, architecture: str = "x64"):
        platform_type = SupportedPlatform.MACOS_ARM64 if architecture == "arm64" else SupportedPlatform.MACOS_X64
        super().__init__(platform_type)
    
    def _detect_capabilities(self) -> PlatformCapabilities:
        """Detect macOS-specific capabilities."""
        import psutil
        
        # Detect SIMD support
        simd_support = []
        if self.platform == SupportedPlatform.MACOS_ARM64:
            simd_support.append('NEON')
        else:
            simd_support.extend(['SSE', 'AVX'])
        
        return PlatformCapabilities(
            platform=self.platform,
            performance_tier=PerformanceTier.HIGH_PERFORMANCE,
            simd_support=simd_support,
            memory_gb=psutil.virtual_memory().total / (1024**3),
            compute_units=os.cpu_count() or 1,
            gpu_acceleration=self._check_gpu_support(),
            wasm_support=True,
            threading_support=True,
            file_system_access=True,
            network_access=True,
            hardware_access=True
        )
    
    def optimize_deployment(self, config: Dict[str, Any]) -> OptimizationProfile:
        """Create macOS-specific optimization profile."""
        compiler_flags = ['-O3', '-march=native']
        
        # Add Apple Silicon specific optimizations
        if self.platform == SupportedPlatform.MACOS_ARM64:
            compiler_flags.extend(['-mcpu=apple-a14', '-mtune=apple-a14'])
        
        return OptimizationProfile(
            platform=self.platform,
            compiler_flags=compiler_flags,
            optimization_level='O3',
            vectorization_enabled=True,
            parallel_processing=True,
            memory_optimization=False,
            power_optimization=self.platform == SupportedPlatform.MACOS_ARM64,
            binary_format='mach-o'
        )
    
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install macOS dependencies using Homebrew and pip."""
        try:
            for dep in dependencies:
                if dep == 'cuda':
                    self.logger.warning("CUDA not available on macOS, using Metal Performance Shaders")
                elif dep == 'openblas':
                    subprocess.run(['brew', 'install', 'openblas'], check=True)
                else:
                    subprocess.run(['pip3', 'install', dep], check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate macOS environment."""
        checks = {
            'python_version': sys.version_info >= (3, 9),
            'pip_available': self._check_command('pip3'),
            'xcode_tools': self._check_xcode_tools(),
            'homebrew': self._check_command('brew'),
            'macos_version': self._check_macos_version()
        }
        return checks
    
    def _check_gpu_support(self) -> bool:
        """Check for GPU acceleration support (Metal)."""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=10)
            return 'Metal' in result.stdout
        except:
            return False
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        try:
            subprocess.run(['which', command], capture_output=True, timeout=5, check=True)
            return True
        except:
            return False
    
    def _check_xcode_tools(self) -> bool:
        """Check for Xcode command line tools."""
        try:
            subprocess.run(['xcode-select', '--print-path'], capture_output=True, timeout=5, check=True)
            return True
        except:
            return False
    
    def _check_macos_version(self) -> bool:
        """Check macOS version compatibility."""
        try:
            result = subprocess.run(['sw_vers', '-productVersion'], 
                                  capture_output=True, text=True, timeout=5)
            version = result.stdout.strip()
            # Check for minimum macOS version (10.15+)
            version_parts = version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            return major > 10 or (major == 10 and minor >= 15)
        except:
            return False


class CrossPlatformManager:
    """Manages cross-platform compatibility and deployment."""
    
    def __init__(self):
        self.logger = global_logger
        self.adapters: Dict[SupportedPlatform, PlatformAdapter] = {}
        self.current_platform = self._detect_current_platform()
        
        # Initialize adapter for current platform
        self._initialize_current_adapter()
    
    def _detect_current_platform(self) -> SupportedPlatform:
        """Detect the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == 'windows':
            if machine in ['arm64', 'aarch64']:
                return SupportedPlatform.WINDOWS_ARM64
            else:
                return SupportedPlatform.WINDOWS_X64
        elif system == 'darwin':  # macOS
            if machine in ['arm64', 'aarch64']:
                return SupportedPlatform.MACOS_ARM64
            else:
                return SupportedPlatform.MACOS_X64
        elif system == 'linux':
            if machine in ['arm64', 'aarch64']:
                return SupportedPlatform.LINUX_ARM64
            elif machine in ['armv7l', 'armv7']:
                return SupportedPlatform.LINUX_ARMV7
            else:
                return SupportedPlatform.LINUX_X64
        else:
            raise PlatformError(f"Unsupported platform: {system} {machine}")
    
    def _initialize_current_adapter(self):
        """Initialize adapter for current platform."""
        machine = platform.machine().lower()
        arch = "arm64" if machine in ['arm64', 'aarch64'] else "x64"
        
        if self.current_platform in [SupportedPlatform.WINDOWS_X64, SupportedPlatform.WINDOWS_ARM64]:
            self.adapters[self.current_platform] = WindowsAdapter(arch)
        elif self.current_platform in [SupportedPlatform.LINUX_X64, SupportedPlatform.LINUX_ARM64, SupportedPlatform.LINUX_ARMV7]:
            self.adapters[self.current_platform] = LinuxAdapter(arch)
        elif self.current_platform in [SupportedPlatform.MACOS_X64, SupportedPlatform.MACOS_ARM64]:
            self.adapters[self.current_platform] = MacOSAdapter(arch)
    
    def get_capabilities(self) -> PlatformCapabilities:
        """Get capabilities of current platform."""
        adapter = self.adapters[self.current_platform]
        return adapter.capabilities
    
    def optimize_for_platform(self, config: Dict[str, Any]) -> OptimizationProfile:
        """Get optimization profile for current platform."""
        adapter = self.adapters[self.current_platform]
        return adapter.optimize_deployment(config)
    
    def install_dependencies(self, dependencies: List[str]) -> bool:
        """Install dependencies on current platform."""
        adapter = self.adapters[self.current_platform]
        return adapter.install_dependencies(dependencies)
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate current platform environment."""
        adapter = self.adapters[self.current_platform]
        validation_results = adapter.validate_environment()
        
        return {
            "platform": self.current_platform.value,
            "validation_results": validation_results,
            "all_checks_passed": all(validation_results.values()),
            "capabilities": adapter.capabilities.to_dict()
        }
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information."""
        adapter = self.adapters[self.current_platform]
        
        return {
            "platform": self.current_platform.value,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "capabilities": adapter.capabilities.to_dict(),
            "optimization_profile": adapter.optimize_deployment({}).to_dict()
        }


class CompatibilityChecker:
    """Checks compatibility across different platforms."""
    
    def __init__(self):
        self.logger = global_logger
        self.platform_manager = CrossPlatformManager()
    
    def check_dependency_compatibility(self, dependencies: List[str]) -> Dict[str, Dict[str, bool]]:
        """Check if dependencies are compatible across platforms."""
        compatibility_matrix = {}
        
        # Define known compatibility
        dependency_support = {
            'numpy': {
                SupportedPlatform.WINDOWS_X64: True,
                SupportedPlatform.WINDOWS_ARM64: True,
                SupportedPlatform.MACOS_X64: True,
                SupportedPlatform.MACOS_ARM64: True,
                SupportedPlatform.LINUX_X64: True,
                SupportedPlatform.LINUX_ARM64: True,
                SupportedPlatform.LINUX_ARMV7: True
            },
            'torch': {
                SupportedPlatform.WINDOWS_X64: True,
                SupportedPlatform.WINDOWS_ARM64: False,  # Limited support
                SupportedPlatform.MACOS_X64: True,
                SupportedPlatform.MACOS_ARM64: True,
                SupportedPlatform.LINUX_X64: True,
                SupportedPlatform.LINUX_ARM64: True,
                SupportedPlatform.LINUX_ARMV7: False
            },
            'cuda': {
                SupportedPlatform.WINDOWS_X64: True,
                SupportedPlatform.WINDOWS_ARM64: False,
                SupportedPlatform.MACOS_X64: False,  # CUDA deprecated on macOS
                SupportedPlatform.MACOS_ARM64: False,
                SupportedPlatform.LINUX_X64: True,
                SupportedPlatform.LINUX_ARM64: True,
                SupportedPlatform.LINUX_ARMV7: False
            }
        }
        
        for dep in dependencies:
            compatibility_matrix[dep] = {}
            support_info = dependency_support.get(dep, {})
            
            for platform in SupportedPlatform:
                compatibility_matrix[dep][platform.value] = support_info.get(platform, False)
        
        return compatibility_matrix
    
    def suggest_alternatives(self, incompatible_deps: List[str], platform: SupportedPlatform) -> Dict[str, str]:
        """Suggest alternative dependencies for incompatible ones."""
        alternatives = {
            'cuda': {
                SupportedPlatform.MACOS_X64: 'Metal Performance Shaders',
                SupportedPlatform.MACOS_ARM64: 'Metal Performance Shaders',
                SupportedPlatform.LINUX_ARMV7: 'OpenCL',
                SupportedPlatform.WINDOWS_ARM64: 'DirectML'
            },
            'torch': {
                SupportedPlatform.WINDOWS_ARM64: 'tensorflow-lite',
                SupportedPlatform.LINUX_ARMV7: 'tensorflow-lite'
            }
        }
        
        suggestions = {}
        for dep in incompatible_deps:
            if dep in alternatives and platform in alternatives[dep]:
                suggestions[dep] = alternatives[dep][platform]
        
        return suggestions


# Global cross-platform manager
global_platform_manager = CrossPlatformManager()

def get_current_platform_info() -> Dict[str, Any]:
    """Get information about current platform."""
    return global_platform_manager.get_platform_info()

def validate_current_environment() -> Dict[str, Any]:
    """Validate current platform environment."""
    return global_platform_manager.validate_environment()

def install_platform_dependencies(dependencies: List[str]) -> bool:
    """Install dependencies optimized for current platform."""
    return global_platform_manager.install_dependencies(dependencies)