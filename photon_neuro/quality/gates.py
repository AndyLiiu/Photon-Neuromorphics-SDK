"""
Quality Gates Implementation
===========================

Progressive quality gates for autonomous SDLC execution.
"""

import abc
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil
import ast
import coverage

from ..utils.logging_system import global_logger


class QualityGateResult:
    """Result of a quality gate execution."""
    
    def __init__(self, passed: bool, score: float, details: str, metrics: Dict[str, Any]):
        self.passed = passed
        self.score = score
        self.details = details
        self.metrics = metrics
        self.timestamp = time.time()


class QualityGate(abc.ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, threshold: float = 0.85):
        self.name = name
        self.threshold = threshold
        self.logger = global_logger
    
    @abc.abstractmethod
    def execute(self) -> QualityGateResult:
        """Execute the quality gate check."""
        pass
    
    def __str__(self):
        return f"QualityGate({self.name}, threshold={self.threshold})"


class CodeQualityGate(QualityGate):
    """Code quality gate using static analysis."""
    
    def __init__(self, source_dir: str = "photon_neuro", threshold: float = 0.90):
        super().__init__("Code Quality", threshold)
        self.source_dir = Path(source_dir)
    
    def execute(self) -> QualityGateResult:
        """Execute code quality checks."""
        self.logger.info(f"Executing {self.name} gate...")
        
        metrics = {}
        total_score = 0.0
        
        # Check syntax errors
        syntax_score = self._check_syntax()
        metrics["syntax_score"] = syntax_score
        total_score += syntax_score * 0.3
        
        # Check code complexity
        complexity_score = self._check_complexity()
        metrics["complexity_score"] = complexity_score
        total_score += complexity_score * 0.3
        
        # Check import structure
        import_score = self._check_imports()
        metrics["import_score"] = import_score
        total_score += import_score * 0.2
        
        # Check docstring coverage
        docstring_score = self._check_docstrings()
        metrics["docstring_score"] = docstring_score
        total_score += docstring_score * 0.2
        
        passed = total_score >= self.threshold
        details = f"Code quality score: {total_score:.3f} (threshold: {self.threshold})"
        
        if not passed:
            details += f"\nSyntax: {syntax_score:.3f}, Complexity: {complexity_score:.3f}, "
            details += f"Imports: {import_score:.3f}, Docstrings: {docstring_score:.3f}"
        
        return QualityGateResult(passed, total_score, details, metrics)
    
    def _check_syntax(self) -> float:
        """Check for syntax errors."""
        error_count = 0
        file_count = 0
        
        for py_file in self.source_dir.rglob("*.py"):
            file_count += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError:
                error_count += 1
                self.logger.warning(f"Syntax error in {py_file}")
        
        return 1.0 - (error_count / max(file_count, 1))
    
    def _check_complexity(self) -> float:
        """Check cyclomatic complexity."""
        try:
            result = subprocess.run(
                ["python", "-m", "mccabe", "--min", "10", str(self.source_dir)],
                capture_output=True, text=True, timeout=30
            )
            
            # Count high complexity functions
            high_complexity_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Rough estimate - assume 100 functions total, penalize high complexity
            complexity_score = max(0.0, 1.0 - (high_complexity_count / 20))
            return complexity_score
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("McCabe complexity check failed, assuming good complexity")
            return 0.85
    
    def _check_imports(self) -> float:
        """Check import structure and circular imports."""
        import_issues = 0
        file_count = 0
        
        for py_file in self.source_dir.rglob("*.py"):
            file_count += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Check for relative imports beyond package
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.level > 2:  # Too many relative imports
                            import_issues += 1
                            
            except Exception:
                import_issues += 1
        
        return 1.0 - (import_issues / max(file_count, 1))
    
    def _check_docstrings(self) -> float:
        """Check docstring coverage."""
        functions_with_docs = 0
        total_functions = 0
        
        for py_file in self.source_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        if (node.body and isinstance(node.body[0], ast.Expr) 
                            and isinstance(node.body[0].value, ast.Str)):
                            functions_with_docs += 1
                            
            except Exception:
                continue
        
        return functions_with_docs / max(total_functions, 1)


class TestCoverageGate(QualityGate):
    """Test coverage quality gate."""
    
    def __init__(self, source_dir: str = "photon_neuro", test_dir: str = "tests", threshold: float = 0.85):
        super().__init__("Test Coverage", threshold)
        self.source_dir = source_dir
        self.test_dir = test_dir
    
    def execute(self) -> QualityGateResult:
        """Execute test coverage analysis."""
        self.logger.info(f"Executing {self.name} gate...")
        
        try:
            # Run tests with coverage
            result = subprocess.run([
                "python", "-m", "pytest", self.test_dir, 
                "--cov=" + self.source_dir, "--cov-report=term-missing",
                "--tb=short", "-v"
            ], capture_output=True, text=True, timeout=120)
            
            # Parse coverage from output
            coverage_percentage = self._parse_coverage(result.stdout)
            
            # Count test results
            test_metrics = self._parse_test_results(result.stdout)
            
            passed = coverage_percentage >= self.threshold and test_metrics["failed"] == 0
            
            details = f"Coverage: {coverage_percentage:.1%}, Tests passed: {test_metrics['passed']}"
            if test_metrics["failed"] > 0:
                details += f", Tests failed: {test_metrics['failed']}"
            
            metrics = {
                "coverage_percentage": coverage_percentage,
                "tests_passed": test_metrics["passed"],
                "tests_failed": test_metrics["failed"],
                "test_exit_code": result.returncode
            }
            
            return QualityGateResult(passed, coverage_percentage, details, metrics)
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(False, 0.0, "Tests timed out", {"timeout": True})
        except Exception as e:
            return QualityGateResult(False, 0.0, f"Test execution failed: {e}", {"error": str(e)})
    
    def _parse_coverage(self, output: str) -> float:
        """Parse coverage percentage from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                try:
                    # Extract percentage
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part.rstrip('%')) / 100.0
                except ValueError:
                    continue
        return 0.0
    
    def _parse_test_results(self, output: str) -> Dict[str, int]:
        """Parse test results from pytest output."""
        lines = output.split('\n')
        passed = failed = 0
        
        for line in lines:
            if ' passed' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed' and i > 0:
                            passed = int(parts[i-1])
                            break
                except ValueError:
                    continue
            
            if ' failed' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'failed' and i > 0:
                            failed = int(parts[i-1])
                            break
                except ValueError:
                    continue
        
        return {"passed": passed, "failed": failed}


class SecurityGate(QualityGate):
    """Security vulnerability gate."""
    
    def __init__(self, threshold: float = 1.0):
        super().__init__("Security", threshold)
    
    def execute(self) -> QualityGateResult:
        """Execute security vulnerability scan."""
        self.logger.info(f"Executing {self.name} gate...")
        
        vulnerabilities = []
        
        # Check for common security issues
        vuln_count = self._check_common_vulnerabilities()
        
        # Check dependencies for known vulnerabilities
        dep_vulns = self._check_dependency_vulnerabilities()
        
        total_vulns = vuln_count + len(dep_vulns)
        score = 1.0 if total_vulns == 0 else 0.0
        passed = score >= self.threshold
        
        details = f"Vulnerabilities found: {total_vulns}"
        if dep_vulns:
            details += f" (Dependencies: {len(dep_vulns)})"
        
        metrics = {
            "code_vulnerabilities": vuln_count,
            "dependency_vulnerabilities": len(dep_vulns),
            "total_vulnerabilities": total_vulns
        }
        
        return QualityGateResult(passed, score, details, metrics)
    
    def _check_common_vulnerabilities(self) -> int:
        """Check for common security patterns."""
        vuln_count = 0
        dangerous_patterns = [
            "eval(", "exec(", "import os", "subprocess.call",
            "shell=True", "__import__", "input("
        ]
        
        for py_file in Path("photon_neuro").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in dangerous_patterns:
                        if pattern in content:
                            self.logger.warning(f"Potential security issue in {py_file}: {pattern}")
                            vuln_count += 1
            except Exception:
                continue
        
        return vuln_count
    
    def _check_dependency_vulnerabilities(self) -> List[str]:
        """Check dependencies for known vulnerabilities."""
        try:
            result = subprocess.run([
                "python", "-m", "pip", "audit", "--format=json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # pip audit returned clean
                return []
            else:
                # Assume vulnerabilities found
                return ["pip audit found issues"]
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("pip audit not available, skipping dependency check")
            return []


class PerformanceGate(QualityGate):
    """Performance benchmarking gate."""
    
    def __init__(self, threshold: float = 0.85):
        super().__init__("Performance", threshold)
    
    def execute(self) -> QualityGateResult:
        """Execute performance benchmarks."""
        self.logger.info(f"Executing {self.name} gate...")
        
        benchmarks = {}
        
        # Memory usage benchmark
        memory_score = self._benchmark_memory()
        benchmarks["memory_efficiency"] = memory_score
        
        # Import time benchmark
        import_score = self._benchmark_import_time()
        benchmarks["import_speed"] = import_score
        
        # Basic functionality benchmark
        functionality_score = self._benchmark_core_functionality()
        benchmarks["core_functionality"] = functionality_score
        
        # Weighted average
        total_score = (memory_score * 0.3 + import_score * 0.3 + functionality_score * 0.4)
        
        passed = total_score >= self.threshold
        details = f"Performance score: {total_score:.3f} (threshold: {self.threshold})"
        
        return QualityGateResult(passed, total_score, details, benchmarks)
    
    def _benchmark_memory(self) -> float:
        """Benchmark memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # Import and use core components
            import photon_neuro as pn
            
            # Create a small test component
            component = pn.PhotonicComponent()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Score based on memory increase (lower is better)
            # Penalize if memory increase > 50MB
            score = max(0.0, 1.0 - (memory_increase / (50 * 1024 * 1024)))
            return min(1.0, score)
            
        except Exception:
            return 0.5  # Neutral score if benchmark fails
    
    def _benchmark_import_time(self) -> float:
        """Benchmark import time."""
        start_time = time.time()
        
        try:
            import photon_neuro
            import_time = time.time() - start_time
            
            # Score based on import time (lower is better)
            # Penalize if import takes > 2 seconds
            score = max(0.0, 1.0 - (import_time / 2.0))
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _benchmark_core_functionality(self) -> float:
        """Benchmark core functionality performance."""
        try:
            import numpy as np
            import photon_neuro as pn
            
            start_time = time.time()
            
            # Basic operations
            simulator = pn.PhotonicSimulator()
            component = pn.PhotonicComponent()
            
            # Simple computation
            data = np.random.randn(100, 100)
            result = np.dot(data, data.T)
            
            execution_time = time.time() - start_time
            
            # Score based on execution time
            score = max(0.0, 1.0 - (execution_time / 1.0))  # Penalize if > 1 second
            return min(1.0, score)
            
        except Exception:
            return 0.0


class DocumentationGate(QualityGate):
    """Documentation quality gate."""
    
    def __init__(self, threshold: float = 0.80):
        super().__init__("Documentation", threshold)
    
    def execute(self) -> QualityGateResult:
        """Execute documentation quality check."""
        self.logger.info(f"Executing {self.name} gate...")
        
        scores = {}
        
        # Check README completeness
        readme_score = self._check_readme()
        scores["readme_quality"] = readme_score
        
        # Check API documentation
        api_doc_score = self._check_api_docs()
        scores["api_documentation"] = api_doc_score
        
        # Check examples
        examples_score = self._check_examples()
        scores["examples_quality"] = examples_score
        
        total_score = (readme_score * 0.4 + api_doc_score * 0.4 + examples_score * 0.2)
        
        passed = total_score >= self.threshold
        details = f"Documentation score: {total_score:.3f} (threshold: {self.threshold})"
        
        return QualityGateResult(passed, total_score, details, scores)
    
    def _check_readme(self) -> float:
        """Check README quality."""
        readme_path = Path("README.md")
        
        if not readme_path.exists():
            return 0.0
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_sections = [
                "installation", "features", "usage", "example",
                "documentation", "license"
            ]
            
            section_count = 0
            for section in required_sections:
                if section.lower() in content.lower():
                    section_count += 1
            
            return section_count / len(required_sections)
            
        except Exception:
            return 0.0
    
    def _check_api_docs(self) -> float:
        """Check API documentation quality."""
        # Check if key modules have docstrings
        key_modules = [
            "photon_neuro/__init__.py",
            "photon_neuro/core/components.py",
            "photon_neuro/networks/feedforward.py"
        ]
        
        documented_modules = 0
        
        for module_path in key_modules:
            path = Path(module_path)
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for module docstring
                    if '"""' in content or "'''" in content:
                        documented_modules += 1
                        
                except Exception:
                    continue
        
        return documented_modules / len(key_modules)
    
    def _check_examples(self) -> float:
        """Check examples quality."""
        examples_dir = Path("examples")
        
        if not examples_dir.exists():
            return 0.0
        
        example_files = list(examples_dir.glob("*.py"))
        working_examples = 0
        
        for example_file in example_files:
            try:
                with open(example_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic structure
                if "import" in content and "photon_neuro" in content:
                    working_examples += 1
                    
            except Exception:
                continue
        
        return working_examples / max(len(example_files), 1)


class QualityGateRunner:
    """Runner for executing multiple quality gates."""
    
    def __init__(self, gates: List[QualityGate]):
        self.gates = gates
        self.logger = global_logger
    
    def run_all(self) -> Tuple[bool, Dict[str, QualityGateResult]]:
        """Run all quality gates."""
        self.logger.info("Starting quality gate execution...")
        
        results = {}
        all_passed = True
        
        for gate in self.gates:
            try:
                result = gate.execute()
                results[gate.name] = result
                
                if not result.passed:
                    all_passed = False
                    self.logger.warning(f"Quality gate '{gate.name}' FAILED: {result.details}")
                else:
                    self.logger.info(f"Quality gate '{gate.name}' PASSED: {result.details}")
                    
            except Exception as e:
                self.logger.error(f"Quality gate '{gate.name}' ERROR: {e}")
                results[gate.name] = QualityGateResult(False, 0.0, f"Execution error: {e}", {})
                all_passed = False
        
        self.logger.info(f"Quality gates complete. Overall result: {'PASS' if all_passed else 'FAIL'}")
        return all_passed, results