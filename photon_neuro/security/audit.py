"""
Security Auditor
================

Comprehensive security auditing and compliance checking.
"""

import os
import time
import stat
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AuditSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditFinding:
    """Individual audit finding."""
    category: str
    severity: AuditSeverity
    description: str
    file_path: str
    recommendation: str
    details: Dict[str, Any]

@dataclass
class AuditReport:
    """Complete audit report."""
    audit_timestamp: float
    auditor_version: str
    findings: List[AuditFinding]
    summary: Dict[str, Any]

class SecurityAuditor:
    """Comprehensive security auditor."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def run_full_audit(self) -> AuditReport:
        """Run complete security audit."""
        findings = []
        
        # File permission audit
        findings.extend(self._audit_file_permissions())
        
        # Configuration audit
        findings.extend(self._audit_configurations())
        
        # Dependency audit
        findings.extend(self._audit_dependencies())
        
        # Environment audit
        findings.extend(self._audit_environment())
        
        # Generate summary
        summary = self._generate_summary(findings)
        
        return AuditReport(
            audit_timestamp=time.time(),
            auditor_version="1.0.0",
            findings=findings,
            summary=summary
        )
    
    def _audit_file_permissions(self) -> List[AuditFinding]:
        """Audit file permissions."""
        findings = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    file_stat = file_path.stat()
                    permissions = stat.filemode(file_stat.st_mode)
                    
                    # Check for world-writable files
                    if file_stat.st_mode & stat.S_IWOTH:
                        findings.append(AuditFinding(
                            category="file_permissions",
                            severity=AuditSeverity.WARNING,
                            description="File is world-writable",
                            file_path=str(file_path),
                            recommendation="Remove world write permissions",
                            details={"permissions": permissions}
                        ))
                    
                    # Check for executable config files
                    if (file_path.suffix in ['.json', '.yaml', '.yml', '.ini', '.cfg'] and
                        file_stat.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                        findings.append(AuditFinding(
                            category="file_permissions",
                            severity=AuditSeverity.WARNING,
                            description="Configuration file is executable",
                            file_path=str(file_path),
                            recommendation="Remove execute permissions from configuration files",
                            details={"permissions": permissions}
                        ))
                        
                except (OSError, PermissionError):
                    continue
        
        return findings
    
    def _audit_configurations(self) -> List[AuditFinding]:
        """Audit configuration files."""
        findings = []
        
        config_files = []
        for pattern in ["*.json", "*.yaml", "*.yml", "*.ini", "*.cfg"]:
            config_files.extend(self.project_root.rglob(pattern))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read().lower()
                
                # Check for debug mode enabled
                if any(debug_pattern in content for debug_pattern in ['debug = true', 'debug: true', '"debug": true']):
                    findings.append(AuditFinding(
                        category="configuration",
                        severity=AuditSeverity.WARNING,
                        description="Debug mode enabled in configuration",
                        file_path=str(config_file),
                        recommendation="Disable debug mode in production",
                        details={}
                    ))
                
                # Check for default passwords
                if any(default in content for default in ['password123', 'admin', 'default']):
                    findings.append(AuditFinding(
                        category="configuration",
                        severity=AuditSeverity.CRITICAL,
                        description="Default password found in configuration",
                        file_path=str(config_file),
                        recommendation="Change default passwords immediately",
                        details={}
                    ))
                    
            except Exception:
                continue
        
        return findings
    
    def _audit_dependencies(self) -> List[AuditFinding]:
        """Audit dependencies for known vulnerabilities."""
        findings = []
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    requirements = f.read()
                
                # Basic checks for outdated packages
                if "django==" in requirements.lower():
                    findings.append(AuditFinding(
                        category="dependencies",
                        severity=AuditSeverity.INFO,
                        description="Django version should be regularly updated",
                        file_path=str(req_file),
                        recommendation="Regularly update Django to latest stable version",
                        details={}
                    ))
                
                # Check for unpinned versions
                if any(line.strip() and "==" not in line and ">=" not in line and line.strip()[0].isalpha() 
                       for line in requirements.split('\n')):
                    findings.append(AuditFinding(
                        category="dependencies",
                        severity=AuditSeverity.WARNING,
                        description="Dependencies without version pinning found",
                        file_path=str(req_file),
                        recommendation="Pin dependency versions for reproducible builds",
                        details={}
                    ))
                    
            except Exception:
                pass
        
        return findings
    
    def _audit_environment(self) -> List[AuditFinding]:
        """Audit environment security."""
        findings = []
        
        # Check for .env files
        env_files = list(self.project_root.rglob(".env*"))
        for env_file in env_files:
            findings.append(AuditFinding(
                category="environment",
                severity=AuditSeverity.WARNING,
                description="Environment file found - ensure it's not committed to version control",
                file_path=str(env_file),
                recommendation="Add .env files to .gitignore",
                details={}
            ))
        
        # Check Python version
        import sys
        python_version = sys.version_info
        if python_version < (3, 8):
            findings.append(AuditFinding(
                category="environment",
                severity=AuditSeverity.WARNING,
                description=f"Python version {python_version.major}.{python_version.minor} is outdated",
                file_path="system",
                recommendation="Upgrade to Python 3.8 or newer for security updates",
                details={"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
            ))
        
        return findings
    
    def _generate_summary(self, findings: List[AuditFinding]) -> Dict[str, Any]:
        """Generate audit summary."""
        severity_counts = {
            AuditSeverity.CRITICAL: 0,
            AuditSeverity.ERROR: 0,
            AuditSeverity.WARNING: 0,
            AuditSeverity.INFO: 0
        }
        
        category_counts = {}
        
        for finding in findings:
            severity_counts[finding.severity] += 1
            
            if finding.category not in category_counts:
                category_counts[finding.category] = 0
            category_counts[finding.category] += 1
        
        return {
            "total_findings": len(findings),
            "by_severity": {s.value: count for s, count in severity_counts.items()},
            "by_category": category_counts,
            "risk_score": self._calculate_risk_score(severity_counts)
        }
    
    def _calculate_risk_score(self, severity_counts: Dict[AuditSeverity, int]) -> int:
        """Calculate overall risk score."""
        weights = {
            AuditSeverity.CRITICAL: 10,
            AuditSeverity.ERROR: 5,
            AuditSeverity.WARNING: 2,
            AuditSeverity.INFO: 1
        }
        
        score = sum(severity_counts[severity] * weights[severity] 
                   for severity in severity_counts)
        
        return min(score, 100)  # Cap at 100