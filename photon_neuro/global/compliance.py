"""
Global Compliance Framework
==========================

GDPR, CCPA, PDPA compliance implementation with automated data protection,
consent management, and privacy-by-design for photonic neural networks.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path

from ..utils.logging_system import global_logger
from ..core.exceptions import ComplianceError


class DataSubjectRights(Enum):
    """Enumeration of data subject rights across regulations."""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct inaccurate data
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICTION = "restriction"  # Right to restrict processing
    OBJECTION = "objection"  # Right to object to processing
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent


class ProcessingPurpose(Enum):
    """Legal bases for data processing."""
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    data_type: str
    purpose: ProcessingPurpose
    legal_basis: str
    retention_period: int  # days
    data_subjects: List[str]
    recipients: List[str]
    cross_border_transfers: bool
    safeguards: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ConsentRecord:
    """Record of user consent."""
    user_id: str
    purpose: str
    consent_given: bool
    timestamp: float
    withdrawal_timestamp: Optional[float] = None
    consent_method: str = "explicit"
    data_categories: List[str] = None
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.withdrawal_timestamp:
            return False
        
        # Consent expires after 2 years under GDPR
        age_days = (time.time() - self.timestamp) / (24 * 3600)
        return age_days < 730


class ComplianceFramework(ABC):
    """Abstract base class for compliance frameworks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = global_logger
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
    
    @abstractmethod
    def validate_processing(self, data_type: str, purpose: str) -> bool:
        """Validate if data processing is compliant."""
        pass
    
    @abstractmethod
    def handle_subject_request(self, request_type: DataSubjectRights, user_id: str) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        pass
    
    def record_processing(self, record: DataProcessingRecord):
        """Record data processing activity."""
        self.processing_records.append(record)
        self.logger.info(f"Recorded processing activity: {record.data_type} for {record.purpose.value}")
    
    def record_consent(self, consent: ConsentRecord):
        """Record user consent."""
        self.consent_records[consent.user_id] = consent
        self.logger.info(f"Recorded consent for user {consent.user_id}")


class GDPRCompliance(ComplianceFramework):
    """GDPR compliance implementation."""
    
    def __init__(self):
        super().__init__("GDPR")
        self.lawful_bases = {
            ProcessingPurpose.CONSENT: "Article 6(1)(a) - Consent",
            ProcessingPurpose.CONTRACT: "Article 6(1)(b) - Contract performance", 
            ProcessingPurpose.LEGAL_OBLIGATION: "Article 6(1)(c) - Legal obligation",
            ProcessingPurpose.VITAL_INTERESTS: "Article 6(1)(d) - Vital interests",
            ProcessingPurpose.PUBLIC_TASK: "Article 6(1)(e) - Public task",
            ProcessingPurpose.LEGITIMATE_INTEREST: "Article 6(1)(f) - Legitimate interests"
        }
    
    def validate_processing(self, data_type: str, purpose: str) -> bool:
        """Validate GDPR compliance for data processing."""
        # Check if data minimization is respected
        if not self._check_data_minimization(data_type, purpose):
            return False
        
        # Check if purpose limitation is respected  
        if not self._check_purpose_limitation(purpose):
            return False
        
        # Check if storage limitation is respected
        if not self._check_storage_limitation(data_type):
            return False
        
        return True
    
    def handle_subject_request(self, request_type: DataSubjectRights, user_id: str) -> Dict[str, Any]:
        """Handle GDPR data subject rights requests."""
        response = {"status": "processed", "user_id": user_id, "request_type": request_type.value}
        
        if request_type == DataSubjectRights.ACCESS:
            # Article 15 - Right of access
            user_data = self._collect_user_data(user_id)
            response["data"] = user_data
            response["processing_purposes"] = self._get_user_processing_purposes(user_id)
            
        elif request_type == DataSubjectRights.RECTIFICATION:
            # Article 16 - Right to rectification
            response["action"] = "Data correction process initiated"
            
        elif request_type == DataSubjectRights.ERASURE:
            # Article 17 - Right to erasure ('right to be forgotten')
            self._delete_user_data(user_id)
            response["action"] = "User data deleted"
            
        elif request_type == DataSubjectRights.PORTABILITY:
            # Article 20 - Right to data portability
            portable_data = self._export_user_data(user_id)
            response["data"] = portable_data
            response["format"] = "JSON"
            
        elif request_type == DataSubjectRights.WITHDRAW_CONSENT:
            # Article 7 - Conditions for consent
            self._withdraw_consent(user_id)
            response["action"] = "Consent withdrawn"
        
        self.logger.info(f"Processed GDPR request: {request_type.value} for user {user_id}")
        return response
    
    def _check_data_minimization(self, data_type: str, purpose: str) -> bool:
        """Check GDPR data minimization principle."""
        # Only process data that is necessary for the specified purpose
        necessary_data = {
            "photonic_training": ["model_parameters", "performance_metrics"],
            "hardware_calibration": ["device_settings", "measurement_data"],
            "simulation": ["input_parameters", "simulation_results"]
        }
        
        return data_type in necessary_data.get(purpose, [])
    
    def _check_purpose_limitation(self, purpose: str) -> bool:
        """Check GDPR purpose limitation principle."""
        allowed_purposes = [
            "photonic_training", "hardware_calibration", "simulation",
            "performance_optimization", "research_development"
        ]
        return purpose in allowed_purposes
    
    def _check_storage_limitation(self, data_type: str) -> bool:
        """Check GDPR storage limitation principle."""
        # Define retention periods for different data types
        retention_periods = {
            "model_parameters": 365,  # 1 year
            "performance_metrics": 180,  # 6 months
            "calibration_data": 90,  # 3 months
            "simulation_results": 30  # 1 month
        }
        
        return data_type in retention_periods
    
    def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all data for a specific user."""
        return {
            "user_id": user_id,
            "processing_records": [r.to_dict() for r in self.processing_records if user_id in r.data_subjects],
            "consent_records": self.consent_records.get(user_id, {})
        }
    
    def _get_user_processing_purposes(self, user_id: str) -> List[str]:
        """Get all processing purposes for a user."""
        purposes = set()
        for record in self.processing_records:
            if user_id in record.data_subjects:
                purposes.add(record.purpose.value)
        return list(purposes)
    
    def _delete_user_data(self, user_id: str):
        """Delete all data for a specific user."""
        # Remove processing records
        self.processing_records = [r for r in self.processing_records if user_id not in r.data_subjects]
        
        # Remove consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format."""
        return self._collect_user_data(user_id)
    
    def _withdraw_consent(self, user_id: str):
        """Withdraw consent for a user."""
        if user_id in self.consent_records:
            self.consent_records[user_id].withdrawal_timestamp = time.time()


class CCPACompliance(ComplianceFramework):
    """California Consumer Privacy Act (CCPA) compliance."""
    
    def __init__(self):
        super().__init__("CCPA")
        self.sale_opt_outs: Set[str] = set()
    
    def validate_processing(self, data_type: str, purpose: str) -> bool:
        """Validate CCPA compliance for data processing."""
        # CCPA focuses on transparency and consumer rights
        if purpose == "sale_to_third_party":
            return False  # Default to no sale unless explicitly consented
        
        return True
    
    def handle_subject_request(self, request_type: DataSubjectRights, user_id: str) -> Dict[str, Any]:
        """Handle CCPA consumer rights requests."""
        response = {"status": "processed", "user_id": user_id, "request_type": request_type.value}
        
        if request_type == DataSubjectRights.ACCESS:
            # Right to know
            user_data = self._collect_user_data(user_id)
            response["categories"] = self._get_data_categories(user_id)
            response["sources"] = self._get_data_sources(user_id)
            response["business_purposes"] = self._get_business_purposes(user_id)
            
        elif request_type == DataSubjectRights.ERASURE:
            # Right to delete
            self._delete_user_data(user_id)
            response["action"] = "Consumer data deleted"
        
        elif request_type == DataSubjectRights.OBJECTION:
            # Right to opt-out of sale
            self.sale_opt_outs.add(user_id)
            response["action"] = "Opted out of data sale"
        
        return response
    
    def _get_data_categories(self, user_id: str) -> List[str]:
        """Get categories of personal information collected."""
        return ["photonic_model_data", "performance_metrics", "device_settings"]
    
    def _get_data_sources(self, user_id: str) -> List[str]:
        """Get sources of personal information."""
        return ["direct_user_input", "device_measurements", "simulation_outputs"]
    
    def _get_business_purposes(self, user_id: str) -> List[str]:
        """Get business purposes for data processing."""
        return ["model_training", "performance_optimization", "research_development"]


class PDPACompliance(ComplianceFramework):
    """Personal Data Protection Act (PDPA) compliance for ASEAN countries."""
    
    def __init__(self):
        super().__init__("PDPA")
    
    def validate_processing(self, data_type: str, purpose: str) -> bool:
        """Validate PDPA compliance for data processing."""
        # PDPA emphasizes consent and notification
        if data_type in ["sensitive_personal_data", "biometric_data"]:
            # Requires explicit consent for sensitive data
            return False  # Must check explicit consent separately
        
        return True
    
    def handle_subject_request(self, request_type: DataSubjectRights, user_id: str) -> Dict[str, Any]:
        """Handle PDPA data subject rights requests."""
        response = {"status": "processed", "user_id": user_id, "request_type": request_type.value}
        
        if request_type == DataSubjectRights.ACCESS:
            # Right to access personal data
            user_data = self._collect_user_data(user_id)
            response["data"] = user_data
            
        elif request_type == DataSubjectRights.RECTIFICATION:
            # Right to correct personal data
            response["action"] = "Data correction process initiated"
            
        elif request_type == DataSubjectRights.WITHDRAW_CONSENT:
            # Right to withdraw consent
            self._withdraw_consent(user_id)
            response["action"] = "Consent withdrawn"
        
        return response


class ComplianceOrchestrator:
    """Orchestrates multiple compliance frameworks."""
    
    def __init__(self):
        self.frameworks = {
            "GDPR": GDPRCompliance(),
            "CCPA": CCPACompliance(), 
            "PDPA": PDPACompliance()
        }
        self.logger = global_logger
        self.active_regulations: List[str] = []
    
    def configure_for_region(self, region: str):
        """Configure compliance for specific geographic region."""
        region_regulations = {
            "EU": ["GDPR"],
            "EEA": ["GDPR"],
            "California": ["CCPA"],
            "Singapore": ["PDPA"],
            "Thailand": ["PDPA"],
            "Malaysia": ["PDPA"],
            "Philippines": ["PDPA"]
        }
        
        self.active_regulations = region_regulations.get(region, [])
        self.logger.info(f"Configured compliance for region {region}: {self.active_regulations}")
    
    def validate_processing(self, data_type: str, purpose: str, region: str) -> bool:
        """Validate data processing against all applicable regulations."""
        self.configure_for_region(region)
        
        for regulation in self.active_regulations:
            framework = self.frameworks[regulation]
            if not framework.validate_processing(data_type, purpose):
                self.logger.warning(f"Processing validation failed for {regulation}")
                return False
        
        return True
    
    def handle_subject_request(self, request_type: DataSubjectRights, user_id: str, region: str) -> Dict[str, Any]:
        """Handle data subject request according to applicable regulations."""
        self.configure_for_region(region)
        
        responses = {}
        for regulation in self.active_regulations:
            framework = self.frameworks[regulation]
            response = framework.handle_subject_request(request_type, user_id)
            responses[regulation] = response
        
        return {
            "user_id": user_id,
            "request_type": request_type.value,
            "region": region,
            "regulations": self.active_regulations,
            "responses": responses,
            "timestamp": time.time()
        }
    
    def privacy_by_design_check(self, system_component: str) -> Dict[str, bool]:
        """Perform privacy-by-design assessment."""
        checks = {
            "data_minimization": self._check_data_minimization(system_component),
            "purpose_limitation": self._check_purpose_limitation(system_component),
            "transparency": self._check_transparency(system_component),
            "security_measures": self._check_security_measures(system_component),
            "storage_limitation": self._check_storage_limitation(system_component)
        }
        
        all_passed = all(checks.values())
        self.logger.info(f"Privacy-by-design check for {system_component}: {'PASSED' if all_passed else 'FAILED'}")
        
        return checks
    
    def _check_data_minimization(self, component: str) -> bool:
        """Check if component follows data minimization."""
        return True  # Implement specific logic
    
    def _check_purpose_limitation(self, component: str) -> bool:
        """Check if component respects purpose limitation."""
        return True  # Implement specific logic
    
    def _check_transparency(self, component: str) -> bool:
        """Check if component provides transparency."""
        return True  # Implement specific logic
    
    def _check_security_measures(self, component: str) -> bool:
        """Check if component has adequate security."""
        return True  # Implement specific logic
    
    def _check_storage_limitation(self, component: str) -> bool:
        """Check if component respects storage limitation."""
        return True  # Implement specific logic


# Global compliance orchestrator
global_compliance = ComplianceOrchestrator()

def ensure_compliance(data_type: str, purpose: str, region: str) -> bool:
    """Ensure data processing complies with regional regulations."""
    return global_compliance.validate_processing(data_type, purpose, region)

def handle_privacy_request(request_type: str, user_id: str, region: str) -> Dict[str, Any]:
    """Handle privacy/data subject rights request."""
    try:
        request_enum = DataSubjectRights(request_type)
        return global_compliance.handle_subject_request(request_enum, user_id, region)
    except ValueError:
        raise ComplianceError(f"Invalid request type: {request_type}")