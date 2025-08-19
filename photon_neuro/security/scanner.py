"""
Security Scanner
================

Comprehensive security scanning for code vulnerabilities and security issues.
"""

import os
import re
import ast
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueType(Enum):
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_RANDOM = "insecure_random"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    INFORMATION_DISCLOSURE = "information_disclosure"

@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    issue_type: IssueType
    severity: Severity
    file_path: str
    line_number: int
    description: str
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None

@dataclass 
class VulnerabilityReport:
    """Report of all vulnerabilities found during security scan."""
    scan_timestamp: float
    scanned_files: int
    total_issues: int
    issues_by_severity: Dict[Severity, int]
    issues: List[SecurityIssue]
    scan_duration_ms: float

class SecurityScanner:
    """Comprehensive security scanner for Python code."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.patterns = self._load_security_patterns()
        
    def scan_project(self) -> VulnerabilityReport:
        """Scan the entire project for security vulnerabilities."""
        import time
        start_time = time.time()
        
        issues = []
        scanned_files = 0
        
        # Scan Python files
        for py_file in self.project_root.rglob("*.py"):
            if self._should_scan_file(py_file):
                file_issues = self._scan_file(py_file)
                issues.extend(file_issues)
                scanned_files += 1
        
        # Scan configuration files
        for config_file in self._find_config_files():
            file_issues = self._scan_config_file(config_file)
            issues.extend(file_issues)
            scanned_files += 1
        
        scan_duration = (time.time() - start_time) * 1000
        
        # Generate report
        issues_by_severity = {
            Severity.CRITICAL: len([i for i in issues if i.severity == Severity.CRITICAL]),
            Severity.HIGH: len([i for i in issues if i.severity == Severity.HIGH]),
            Severity.MEDIUM: len([i for i in issues if i.severity == Severity.MEDIUM]),
            Severity.LOW: len([i for i in issues if i.severity == Severity.LOW])
        }
        
        return VulnerabilityReport(
            scan_timestamp=time.time(),
            scanned_files=scanned_files,
            total_issues=len(issues),
            issues_by_severity=issues_by_severity,
            issues=issues,
            scan_duration_ms=scan_duration
        )
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if file should be scanned."""
        # Skip test files, migrations, and virtual environments
        exclude_patterns = [
            "test_", "_test.py", "tests/", "__pycache__",
            ".venv/", "venv/", "env/", ".git/",
            "migrations/", "node_modules/"
        ]
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in exclude_patterns)
    
    def _scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single Python file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST for advanced analysis
            try:
                tree = ast.parse(content)
                ast_issues = self._analyze_ast(tree, file_path, lines)
                issues.extend(ast_issues)
            except SyntaxError:
                # File has syntax errors, skip AST analysis
                pass
            
            # Pattern-based scanning
            pattern_issues = self._scan_patterns(content, file_path, lines)
            issues.extend(pattern_issues)
            
        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[SecurityIssue]:
        """Analyze AST for security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for unsafe function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name == "eval":
                        issues.append(SecurityIssue(
                            issue_type=IssueType.COMMAND_INJECTION,
                            severity=Severity.CRITICAL,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description="Use of eval() function can execute arbitrary code",
                            code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            recommendation="Use ast.literal_eval() for safe evaluation of literals",
                            cwe_id="CWE-94"
                        ))
                    
                    elif func_name == "exec":
                        issues.append(SecurityIssue(
                            issue_type=IssueType.COMMAND_INJECTION,
                            severity=Severity.HIGH,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description="Use of exec() function can execute arbitrary code",
                            code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            recommendation="Avoid using exec() or validate input thoroughly",
                            cwe_id="CWE-94"
                        ))
            
            # Check for subprocess calls without shell=False
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Name) and 
                    node.func.value.id == "subprocess"):
                    
                    # Check for shell=True
                    for keyword in node.keywords:
                        if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                issues.append(SecurityIssue(
                                    issue_type=IssueType.COMMAND_INJECTION,
                                    severity=Severity.HIGH,
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    description="subprocess call with shell=True can lead to command injection",
                                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                                    recommendation="Use shell=False and pass arguments as a list",
                                    cwe_id="CWE-78"
                                ))
        
        return issues
    
    def _scan_patterns(self, content: str, file_path: Path, lines: List[str]) -> List[SecurityIssue]:
        """Scan content using regex patterns."""
        issues = []
        
        for pattern_info in self.patterns:
            pattern = pattern_info["pattern"]
            issue_type = pattern_info["type"]
            severity = pattern_info["severity"]
            description = pattern_info["description"]
            recommendation = pattern_info["recommendation"]
            
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                line_num = content[:match.start()].count('\n') + 1
                code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                
                issues.append(SecurityIssue(
                    issue_type=issue_type,
                    severity=severity,
                    file_path=str(file_path),
                    line_number=line_num,
                    description=description,
                    code_snippet=code_snippet.strip(),
                    recommendation=recommendation
                ))
        
        return issues
    
    def _scan_config_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan configuration files for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for hardcoded secrets in config files
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
                r'secret[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']'
            ]
            
            for pattern in secret_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    issues.append(SecurityIssue(
                        issue_type=IssueType.HARDCODED_SECRET,
                        severity=Severity.HIGH,
                        file_path=str(file_path),
                        line_number=line_num,
                        description="Hardcoded secret found in configuration file",
                        code_snippet=code_snippet.strip(),
                        recommendation="Use environment variables or secure secret management",
                        cwe_id="CWE-798"
                    ))
                    
        except Exception as e:
            logger.warning(f"Could not scan config file {file_path}: {e}")
        
        return issues
    
    def _find_config_files(self) -> List[Path]:
        """Find configuration files to scan."""
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.ini", "*.cfg", "*.conf"]
        config_files = []
        
        for pattern in config_patterns:
            config_files.extend(self.project_root.rglob(pattern))
        
        # Filter out common non-config files
        filtered_files = []
        exclude_patterns = ["package.json", "tsconfig.json", "__pycache__"]
        
        for file_path in config_files:
            if not any(exclude in str(file_path) for exclude in exclude_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _load_security_patterns(self) -> List[Dict]:
        """Load security patterns for scanning."""
        return [
            {
                "pattern": r'password\s*=\s*["\'][^"\']+["\']',
                "type": IssueType.HARDCODED_SECRET,
                "severity": Severity.HIGH,
                "description": "Hardcoded password found",
                "recommendation": "Use environment variables or secure credential storage"
            },
            {
                "pattern": r'random\.random\(\)',
                "type": IssueType.INSECURE_RANDOM,
                "severity": Severity.MEDIUM,
                "description": "Use of insecure random number generator",
                "recommendation": "Use secrets.SystemRandom() for cryptographic purposes"
            },
            {
                "pattern": r'hashlib\.md5\(',
                "type": IssueType.WEAK_CRYPTO,
                "severity": Severity.MEDIUM,
                "description": "Use of weak MD5 hash algorithm",
                "recommendation": "Use SHA-256 or stronger hash algorithms"
            },
            {
                "pattern": r'hashlib\.sha1\(',
                "type": IssueType.WEAK_CRYPTO,
                "severity": Severity.MEDIUM,
                "description": "Use of weak SHA-1 hash algorithm",
                "recommendation": "Use SHA-256 or stronger hash algorithms"
            },
            {
                "pattern": r'os\.system\(',
                "type": IssueType.COMMAND_INJECTION,
                "severity": Severity.HIGH,
                "description": "Use of os.system() can lead to command injection",
                "recommendation": "Use subprocess.run() with shell=False"
            },
            {
                "pattern": r'pickle\.loads?\(',
                "type": IssueType.UNSAFE_DESERIALIZATION,
                "severity": Severity.HIGH,
                "description": "Unsafe deserialization with pickle",
                "recommendation": "Use safe serialization formats like JSON"
            },
            {
                "pattern": r'input\([^)]*\)',
                "type": IssueType.COMMAND_INJECTION,
                "severity": Severity.MEDIUM,
                "description": "Direct use of input() without validation",
                "recommendation": "Validate and sanitize user input"
            }
        ]
    
    def generate_sarif_report(self, report: VulnerabilityReport, output_path: str):
        """Generate SARIF format report for security tools integration."""
        import json
        
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "PhotonNeuroSecurityScanner",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/danieleschmidt/Photon-Neuromorphics-SDK"
                    }
                },
                "results": []
            }]
        }
        
        for issue in report.issues:
            result = {
                "ruleId": issue.issue_type.value,
                "level": self._severity_to_sarif_level(issue.severity),
                "message": {
                    "text": issue.description
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": issue.file_path
                        },
                        "region": {
                            "startLine": issue.line_number
                        }
                    }
                }]
            }
            
            if issue.cwe_id:
                result["properties"] = {"cwe": issue.cwe_id}
            
            sarif_report["runs"][0]["results"].append(result)
        
        with open(output_path, 'w') as f:
            json.dump(sarif_report, f, indent=2)
    
    def _severity_to_sarif_level(self, severity: Severity) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            Severity.CRITICAL: "error",
            Severity.HIGH: "error", 
            Severity.MEDIUM: "warning",
            Severity.LOW: "note"
        }
        return mapping.get(severity, "warning")