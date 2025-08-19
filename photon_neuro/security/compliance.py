"""
Compliance Checker
==================

Compliance checking for various security standards and regulations.
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum

class ComplianceStandard(Enum):
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"
    OWASP = "owasp"

class ComplianceLevel(Enum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class ComplianceCheck:
    """Individual compliance check result."""
    standard: ComplianceStandard
    control_id: str
    control_name: str
    level: ComplianceLevel
    description: str
    evidence: List[str]
    gaps: List[str]
    recommendations: List[str]

@dataclass
class ComplianceReport:
    """Complete compliance report."""
    timestamp: float
    standards_checked: List[ComplianceStandard]
    checks: List[ComplianceCheck]
    summary: Dict[str, Any]

class ComplianceChecker:
    """Comprehensive compliance checker."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def check_compliance(self, standards: List[ComplianceStandard]) -> ComplianceReport:
        """Check compliance against specified standards."""
        all_checks = []
        
        for standard in standards:
            if standard == ComplianceStandard.GDPR:
                all_checks.extend(self._check_gdpr_compliance())
            elif standard == ComplianceStandard.SOC2:
                all_checks.extend(self._check_soc2_compliance())
            elif standard == ComplianceStandard.ISO27001:
                all_checks.extend(self._check_iso27001_compliance())
            elif standard == ComplianceStandard.NIST:
                all_checks.extend(self._check_nist_compliance())
            elif standard == ComplianceStandard.OWASP:
                all_checks.extend(self._check_owasp_compliance())
        
        summary = self._generate_compliance_summary(all_checks)
        
        return ComplianceReport(
            timestamp=time.time(),
            standards_checked=standards,
            checks=all_checks,
            summary=summary
        )
    
    def _check_gdpr_compliance(self) -> List[ComplianceCheck]:
        """Check GDPR compliance requirements."""
        checks = []
        
        # Data Protection by Design
        privacy_files = self._find_privacy_files()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            control_id="GDPR-25",
            control_name="Data Protection by Design",
            level=ComplianceLevel.PARTIAL if privacy_files else ComplianceLevel.NON_COMPLIANT,
            description="Implementation of data protection by design and by default",
            evidence=privacy_files,
            gaps=[] if privacy_files else ["No privacy policy or data protection documentation found"],
            recommendations=["Implement privacy policy", "Document data processing activities"] if not privacy_files else []
        ))
        
        # Data Encryption
        encryption_evidence = self._check_encryption_usage()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.GDPR,
            control_id="GDPR-32",
            control_name="Security of Processing",
            level=ComplianceLevel.PARTIAL if encryption_evidence else ComplianceLevel.NON_COMPLIANT,
            description="Implementation of appropriate technical and organizational measures",
            evidence=encryption_evidence,
            gaps=[] if encryption_evidence else ["No encryption implementation found"],
            recommendations=["Implement data encryption at rest and in transit"] if not encryption_evidence else []
        ))
        
        return checks
    
    def _check_soc2_compliance(self) -> List[ComplianceCheck]:
        """Check SOC 2 compliance requirements."""
        checks = []
        
        # Security Controls
        logging_evidence = self._check_logging_implementation()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.SOC2,
            control_id="SOC2-CC6.1",
            control_name="Logical and Physical Access Controls",
            level=ComplianceLevel.PARTIAL if logging_evidence else ComplianceLevel.NON_COMPLIANT,
            description="System access controls and monitoring",
            evidence=logging_evidence,
            gaps=[] if logging_evidence else ["No comprehensive logging system found"],
            recommendations=["Implement comprehensive access logging"] if not logging_evidence else []
        ))
        
        return checks
    
    def _check_iso27001_compliance(self) -> List[ComplianceCheck]:
        """Check ISO 27001 compliance requirements."""
        checks = []
        
        # Information Security Management System
        security_docs = self._find_security_documentation()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.ISO27001,
            control_id="ISO27001-A.5.1.1",
            control_name="Information Security Policies",
            level=ComplianceLevel.PARTIAL if security_docs else ComplianceLevel.NON_COMPLIANT,
            description="Information security policy management",
            evidence=security_docs,
            gaps=[] if security_docs else ["No security policy documentation found"],
            recommendations=["Create information security policy"] if not security_docs else []
        ))
        
        return checks
    
    def _check_nist_compliance(self) -> List[ComplianceCheck]:
        """Check NIST Cybersecurity Framework compliance."""
        checks = []
        
        # Asset Management
        asset_inventory = self._check_asset_inventory()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.NIST,
            control_id="NIST-ID.AM-1",
            control_name="Asset Management",
            level=ComplianceLevel.PARTIAL if asset_inventory else ComplianceLevel.NON_COMPLIANT,
            description="Physical devices and systems inventory",
            evidence=asset_inventory,
            gaps=[] if asset_inventory else ["No asset inventory documentation found"],
            recommendations=["Create and maintain asset inventory"] if not asset_inventory else []
        ))
        
        return checks
    
    def _check_owasp_compliance(self) -> List[ComplianceCheck]:
        """Check OWASP Top 10 compliance."""
        checks = []
        
        # Input Validation
        validation_evidence = self._check_input_validation()
        checks.append(ComplianceCheck(
            standard=ComplianceStandard.OWASP,
            control_id="OWASP-A03",
            control_name="Injection Prevention",
            level=ComplianceLevel.PARTIAL if validation_evidence else ComplianceLevel.NON_COMPLIANT,
            description="Prevention of injection attacks through input validation",
            evidence=validation_evidence,
            gaps=[] if validation_evidence else ["No input validation implementation found"],
            recommendations=["Implement comprehensive input validation"] if not validation_evidence else []
        ))
        
        return checks
    
    def _find_privacy_files(self) -> List[str]:
        """Find privacy-related files."""
        privacy_files = []
        
        patterns = ["privacy", "gdpr", "data_protection", "consent"]
        for pattern in patterns:
            files = list(self.project_root.rglob(f"*{pattern}*"))
            privacy_files.extend([str(f) for f in files])
        
        return privacy_files
    
    def _check_encryption_usage(self) -> List[str]:
        """Check for encryption implementation."""
        evidence = []
        
        # Look for encryption imports and usage
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if any(crypto_lib in content for crypto_lib in ['cryptography', 'hashlib', 'secrets']):
                    evidence.append(f"Cryptographic libraries used in {py_file}")
                    
            except Exception:
                continue
        
        return evidence
    
    def _check_logging_implementation(self) -> List[str]:
        """Check for logging implementation."""
        evidence = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if 'logging' in content or 'logger' in content:
                    evidence.append(f"Logging implementation found in {py_file}")
                    
            except Exception:
                continue
        
        return evidence
    
    def _find_security_documentation(self) -> List[str]:
        """Find security documentation."""
        docs = []
        
        patterns = ["security", "SECURITY", "security.md", "SECURITY.md"]
        for pattern in patterns:
            files = list(self.project_root.rglob(f"*{pattern}*"))
            docs.extend([str(f) for f in files])
        
        return docs
    
    def _check_asset_inventory(self) -> List[str]:
        """Check for asset inventory documentation."""
        inventory_files = []
        
        patterns = ["inventory", "assets", "components", "dependencies"]
        for pattern in patterns:
            files = list(self.project_root.rglob(f"*{pattern}*"))
            inventory_files.extend([str(f) for f in files])
        
        # Requirements files count as dependency inventory
        req_files = list(self.project_root.rglob("requirements*.txt"))
        inventory_files.extend([str(f) for f in req_files])
        
        return inventory_files
    
    def _check_input_validation(self) -> List[str]:
        """Check for input validation implementation."""
        evidence = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                validation_patterns = ['validate', 'sanitize', 'clean', 'escape']
                if any(pattern in content.lower() for pattern in validation_patterns):
                    evidence.append(f"Input validation found in {py_file}")
                    
            except Exception:
                continue
        
        return evidence
    
    def _generate_compliance_summary(self, checks: List[ComplianceCheck]) -> Dict[str, Any]:
        """Generate compliance summary."""
        by_standard = {}
        by_level = {level.value: 0 for level in ComplianceLevel}
        
        for check in checks:
            standard = check.standard.value
            if standard not in by_standard:
                by_standard[standard] = {level.value: 0 for level in ComplianceLevel}
            
            by_standard[standard][check.level.value] += 1
            by_level[check.level.value] += 1
        
        # Calculate overall compliance score
        weights = {
            ComplianceLevel.COMPLIANT: 1.0,
            ComplianceLevel.PARTIAL: 0.5,
            ComplianceLevel.NON_COMPLIANT: 0.0,
            ComplianceLevel.NOT_APPLICABLE: 0.0
        }
        
        total_applicable = len([c for c in checks if c.level != ComplianceLevel.NOT_APPLICABLE])
        if total_applicable > 0:
            weighted_score = sum(weights[check.level] for check in checks 
                               if check.level != ComplianceLevel.NOT_APPLICABLE)
            compliance_percentage = (weighted_score / total_applicable) * 100
        else:
            compliance_percentage = 0
        
        return {
            "total_checks": len(checks),
            "by_standard": by_standard,
            "by_level": by_level,
            "compliance_percentage": round(compliance_percentage, 2),
            "recommendations_count": sum(len(check.recommendations) for check in checks)
        }