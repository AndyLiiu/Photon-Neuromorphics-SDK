"""
Global-First Implementation
==========================

Multi-region deployment, internationalization, and compliance for the Photon Neuromorphics SDK.
Built-in support for GDPR, CCPA, PDPA compliance and cross-platform compatibility.
"""

from .i18n import InternationalizationManager, LocaleManager, TranslationEngine
from .compliance import GDPRCompliance, CCPACompliance, PDPACompliance, ComplianceOrchestrator
from .deployment import MultiRegionDeployer, GlobalLoadBalancer, RegionManager
from .platform import CrossPlatformManager, PlatformAdapter, CompatibilityChecker

__all__ = [
    "InternationalizationManager",
    "LocaleManager", 
    "TranslationEngine",
    "GDPRCompliance",
    "CCPACompliance",
    "PDPACompliance",
    "ComplianceOrchestrator",
    "MultiRegionDeployer",
    "GlobalLoadBalancer",
    "RegionManager",
    "CrossPlatformManager",
    "PlatformAdapter",
    "CompatibilityChecker"
]