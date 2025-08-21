"""
Multi-Region Deployment Framework
=================================

Global deployment orchestration with intelligent load balancing,
region-aware scaling, and cross-platform compatibility.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from pathlib import Path

from ..utils.logging_system import global_logger
from ..core.exceptions import DeploymentError
from .compliance import global_compliance


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"


class ServiceTier(Enum):
    """Service tier for deployment scaling."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


@dataclass
class RegionConfiguration:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    compliance_requirements: List[str]
    latency_target_ms: float
    availability_target: float  # 0.99 = 99%
    capacity_limits: Dict[str, int]
    data_residency_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LoadBalancerRule:
    """Load balancer routing rule."""
    condition: str  # geographic, latency, capacity
    priority: int
    target_region: DeploymentRegion
    weight: float  # 0.0 to 1.0
    health_check_enabled: bool = True


class RegionManager:
    """Manages individual deployment regions."""
    
    def __init__(self, config: RegionConfiguration):
        self.config = config
        self.logger = global_logger
        self.current_load = 0.0
        self.health_status = "healthy"
        self.deployed_services: Set[str] = set()
        self.performance_metrics = {
            "average_latency_ms": 0.0,
            "error_rate": 0.0,
            "throughput_rps": 0.0,
            "cpu_utilization": 0.0,
            "memory_utilization": 0.0
        }
    
    def deploy_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Deploy a service to this region."""
        try:
            self.logger.info(f"Deploying {service_name} to {self.config.region.value}")
            
            # Validate compliance requirements
            if not self._validate_compliance(service_config):
                raise DeploymentError(f"Compliance validation failed for {service_name}")
            
            # Check capacity limits
            if not self._check_capacity(service_config):
                raise DeploymentError(f"Insufficient capacity for {service_name}")
            
            # Deploy service (simulate)
            self._perform_deployment(service_name, service_config)
            
            self.deployed_services.add(service_name)
            self.logger.info(f"Successfully deployed {service_name} to {self.config.region.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy {service_name}: {e}")
            return False
    
    def scale_service(self, service_name: str, scale_factor: float) -> bool:
        """Scale a deployed service."""
        if service_name not in self.deployed_services:
            self.logger.error(f"Service {service_name} not deployed in {self.config.region.value}")
            return False
        
        try:
            self.logger.info(f"Scaling {service_name} by {scale_factor}x in {self.config.region.value}")
            
            # Check if scaling would exceed capacity limits
            new_capacity_requirement = self._calculate_new_capacity(service_name, scale_factor)
            if not self._can_accommodate_scaling(new_capacity_requirement):
                raise DeploymentError(f"Cannot scale {service_name}: capacity limit exceeded")
            
            # Perform scaling operation
            self._perform_scaling(service_name, scale_factor)
            
            self.logger.info(f"Successfully scaled {service_name} in {self.config.region.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale {service_name}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on region."""
        health_status = {
            "region": self.config.region.value,
            "status": self.health_status,
            "load": self.current_load,
            "deployed_services": list(self.deployed_services),
            "metrics": self.performance_metrics,
            "timestamp": time.time()
        }
        
        # Check if region is healthy
        if (self.performance_metrics["average_latency_ms"] > self.config.latency_target_ms or
            self.performance_metrics["error_rate"] > 0.01):  # 1% error rate threshold
            self.health_status = "degraded"
        else:
            self.health_status = "healthy"
        
        health_status["status"] = self.health_status
        return health_status
    
    def _validate_compliance(self, service_config: Dict[str, Any]) -> bool:
        """Validate service compliance with regional requirements."""
        region_compliance_map = {
            DeploymentRegion.EU_WEST: "EU",
            DeploymentRegion.EU_CENTRAL: "EU",
            DeploymentRegion.US_EAST: "California",  # Assume CCPA applies
            DeploymentRegion.US_WEST: "California",
            DeploymentRegion.ASIA_PACIFIC: "Singapore",
            DeploymentRegion.ASIA_NORTHEAST: "Singapore"
        }
        
        compliance_region = region_compliance_map.get(self.config.region, "EU")
        return global_compliance.validate_processing(
            service_config.get("data_type", "model_data"),
            service_config.get("purpose", "photonic_training"),
            compliance_region
        )
    
    def _check_capacity(self, service_config: Dict[str, Any]) -> bool:
        """Check if region has sufficient capacity."""
        required_cpu = service_config.get("cpu_requirement", 1)
        required_memory = service_config.get("memory_requirement", 1024)
        
        return (required_cpu <= self.config.capacity_limits.get("cpu", 100) and
                required_memory <= self.config.capacity_limits.get("memory_mb", 100000))
    
    def _perform_deployment(self, service_name: str, config: Dict[str, Any]):
        """Perform the actual deployment (simulated)."""
        # Simulate deployment time
        time.sleep(0.1)
        
        # Update load
        self.current_load += config.get("load_impact", 0.1)
    
    def _calculate_new_capacity(self, service_name: str, scale_factor: float) -> Dict[str, float]:
        """Calculate new capacity requirements after scaling."""
        return {
            "cpu": 2.0 * scale_factor,  # Simplified calculation
            "memory": 2048 * scale_factor
        }
    
    def _can_accommodate_scaling(self, capacity_req: Dict[str, float]) -> bool:
        """Check if region can accommodate scaling request."""
        return (capacity_req["cpu"] <= self.config.capacity_limits.get("cpu", 100) and
                capacity_req["memory"] <= self.config.capacity_limits.get("memory_mb", 100000))
    
    def _perform_scaling(self, service_name: str, scale_factor: float):
        """Perform the actual scaling operation (simulated)."""
        # Simulate scaling time
        time.sleep(0.05)
        
        # Update load
        self.current_load *= scale_factor


class GlobalLoadBalancer:
    """Intelligent global load balancer."""
    
    def __init__(self):
        self.logger = global_logger
        self.routing_rules: List[LoadBalancerRule] = []
        self.region_weights: Dict[DeploymentRegion, float] = {}
        self.circuit_breakers: Dict[DeploymentRegion, bool] = {}
        
        # Initialize default routing rules
        self._setup_default_rules()
    
    def add_routing_rule(self, rule: LoadBalancerRule):
        """Add a routing rule to the load balancer."""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority)
        self.logger.info(f"Added routing rule for {rule.target_region.value}")
    
    def route_request(self, request_metadata: Dict[str, Any]) -> DeploymentRegion:
        """Route request to optimal region."""
        user_location = request_metadata.get("location", "US")
        service_type = request_metadata.get("service_type", "photonic_simulation")
        latency_requirement = request_metadata.get("max_latency_ms", 100)
        
        # Apply routing rules in priority order
        for rule in self.routing_rules:
            if self._rule_matches(rule, request_metadata):
                target_region = rule.target_region
                
                # Check circuit breaker
                if self.circuit_breakers.get(target_region, False):
                    self.logger.warning(f"Circuit breaker open for {target_region.value}, trying next rule")
                    continue
                
                self.logger.info(f"Routed request to {target_region.value}")
                return target_region
        
        # Fallback to geographic routing
        return self._geographic_fallback(user_location)
    
    def update_region_health(self, region: DeploymentRegion, health_data: Dict[str, Any]):
        """Update region health and adjust routing."""
        if health_data["status"] == "degraded":
            # Reduce weight for degraded regions
            self.region_weights[region] = min(0.5, self.region_weights.get(region, 1.0))
            self.logger.warning(f"Reduced weight for degraded region {region.value}")
        
        elif health_data["status"] == "unhealthy":
            # Open circuit breaker for unhealthy regions
            self.circuit_breakers[region] = True
            self.logger.error(f"Circuit breaker opened for unhealthy region {region.value}")
        
        else:
            # Restore normal weight and close circuit breaker
            self.region_weights[region] = 1.0
            self.circuit_breakers[region] = False
    
    def _setup_default_rules(self):
        """Setup default routing rules."""
        default_rules = [
            LoadBalancerRule("geographic", 1, DeploymentRegion.US_EAST, 0.3),
            LoadBalancerRule("geographic", 1, DeploymentRegion.EU_WEST, 0.3),
            LoadBalancerRule("geographic", 1, DeploymentRegion.ASIA_PACIFIC, 0.3),
            LoadBalancerRule("latency", 2, DeploymentRegion.US_WEST, 0.1)
        ]
        
        self.routing_rules.extend(default_rules)
    
    def _rule_matches(self, rule: LoadBalancerRule, metadata: Dict[str, Any]) -> bool:
        """Check if routing rule matches request."""
        if rule.condition == "geographic":
            user_location = metadata.get("location", "")
            region_mapping = {
                "US": [DeploymentRegion.US_EAST, DeploymentRegion.US_WEST],
                "EU": [DeploymentRegion.EU_WEST, DeploymentRegion.EU_CENTRAL],
                "Asia": [DeploymentRegion.ASIA_PACIFIC, DeploymentRegion.ASIA_NORTHEAST]
            }
            
            return rule.target_region in region_mapping.get(user_location, [])
        
        elif rule.condition == "latency":
            return metadata.get("max_latency_ms", 1000) < 50
        
        return False
    
    def _geographic_fallback(self, location: str) -> DeploymentRegion:
        """Fallback routing based on geography."""
        geographic_mapping = {
            "US": DeploymentRegion.US_EAST,
            "EU": DeploymentRegion.EU_WEST,
            "Asia": DeploymentRegion.ASIA_PACIFIC,
            "Canada": DeploymentRegion.CANADA,
            "Australia": DeploymentRegion.AUSTRALIA
        }
        
        return geographic_mapping.get(location, DeploymentRegion.US_EAST)


class MultiRegionDeployer:
    """Orchestrates multi-region deployments."""
    
    def __init__(self):
        self.logger = global_logger
        self.region_managers: Dict[DeploymentRegion, RegionManager] = {}
        self.load_balancer = GlobalLoadBalancer()
        self.deployment_topology: Dict[str, List[DeploymentRegion]] = {}
        
        # Initialize default regions
        self._initialize_default_regions()
    
    def _initialize_default_regions(self):
        """Initialize default region configurations."""
        default_configs = [
            RegionConfiguration(
                region=DeploymentRegion.US_EAST,
                compliance_requirements=["CCPA"],
                latency_target_ms=50.0,
                availability_target=0.99,
                capacity_limits={"cpu": 100, "memory_mb": 100000},
                data_residency_required=False
            ),
            RegionConfiguration(
                region=DeploymentRegion.EU_WEST,
                compliance_requirements=["GDPR"],
                latency_target_ms=75.0,
                availability_target=0.995,
                capacity_limits={"cpu": 80, "memory_mb": 80000},
                data_residency_required=True
            ),
            RegionConfiguration(
                region=DeploymentRegion.ASIA_PACIFIC,
                compliance_requirements=["PDPA"],
                latency_target_ms=100.0,
                availability_target=0.99,
                capacity_limits={"cpu": 60, "memory_mb": 60000},
                data_residency_required=True
            )
        ]
        
        for config in default_configs:
            self.region_managers[config.region] = RegionManager(config)
    
    def deploy_globally(self, service_name: str, service_config: Dict[str, Any], 
                       target_regions: Optional[List[DeploymentRegion]] = None) -> Dict[str, bool]:
        """Deploy service to multiple regions."""
        if target_regions is None:
            target_regions = list(self.region_managers.keys())
        
        deployment_results = {}
        
        for region in target_regions:
            if region in self.region_managers:
                manager = self.region_managers[region]
                success = manager.deploy_service(service_name, service_config)
                deployment_results[region.value] = success
            else:
                self.logger.error(f"Region {region.value} not configured")
                deployment_results[region.value] = False
        
        # Update deployment topology
        successful_regions = [
            region for region, success in deployment_results.items() if success
        ]
        self.deployment_topology[service_name] = [
            DeploymentRegion(region) for region in successful_regions
        ]
        
        self.logger.info(f"Global deployment of {service_name} completed: {deployment_results}")
        return deployment_results
    
    def scale_globally(self, service_name: str, scale_factor: float) -> Dict[str, bool]:
        """Scale service across all deployed regions."""
        if service_name not in self.deployment_topology:
            raise DeploymentError(f"Service {service_name} not deployed")
        
        scaling_results = {}
        target_regions = self.deployment_topology[service_name]
        
        for region in target_regions:
            manager = self.region_managers[region]
            success = manager.scale_service(service_name, scale_factor)
            scaling_results[region.value] = success
        
        self.logger.info(f"Global scaling of {service_name} completed: {scaling_results}")
        return scaling_results
    
    def health_check_all_regions(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all regions."""
        health_results = {}
        
        for region, manager in self.region_managers.items():
            health_data = manager.health_check()
            health_results[region.value] = health_data
            
            # Update load balancer with health data
            self.load_balancer.update_region_health(region, health_data)
        
        return health_results
    
    def get_deployment_status(self, service_name: str) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        if service_name not in self.deployment_topology:
            return {"error": f"Service {service_name} not deployed"}
        
        regions = self.deployment_topology[service_name]
        status = {
            "service_name": service_name,
            "deployed_regions": [r.value for r in regions],
            "total_regions": len(regions),
            "health_status": {},
            "load_balancing": {
                "enabled": True,
                "routing_rules": len(self.load_balancer.routing_rules)
            }
        }
        
        # Get health status for each region
        for region in regions:
            manager = self.region_managers[region]
            health = manager.health_check()
            status["health_status"][region.value] = {
                "status": health["status"],
                "load": health["load"],
                "latency_ms": health["metrics"]["average_latency_ms"]
            }
        
        return status
    
    def route_request(self, request_metadata: Dict[str, Any]) -> str:
        """Route request to optimal region."""
        target_region = self.load_balancer.route_request(request_metadata)
        return target_region.value
    
    def add_region(self, config: RegionConfiguration) -> bool:
        """Add a new deployment region."""
        try:
            self.region_managers[config.region] = RegionManager(config)
            self.logger.info(f"Added new region: {config.region.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add region {config.region.value}: {e}")
            return False


# Global deployment manager
global_deployer = MultiRegionDeployer()

def deploy_service_globally(service_name: str, config: Dict[str, Any], 
                          regions: Optional[List[str]] = None) -> Dict[str, bool]:
    """Deploy a service globally."""
    target_regions = None
    if regions:
        target_regions = [DeploymentRegion(r) for r in regions]
    
    return global_deployer.deploy_globally(service_name, config, target_regions)

def get_optimal_region(user_location: str, service_type: str = "photonic_simulation") -> str:
    """Get optimal deployment region for user."""
    request_metadata = {
        "location": user_location,
        "service_type": service_type,
        "max_latency_ms": 100
    }
    
    return global_deployer.route_request(request_metadata)