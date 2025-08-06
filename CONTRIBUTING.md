# Contributing to Photon Neuromorphics SDK

We welcome contributions to the Photon Neuromorphics SDK! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- PyTorch 1.10+
- CUDA toolkit (optional, for GPU support)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Photon-Neuromorphics-SDK.git
   cd Photon-Neuromorphics-SDK
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install package in editable mode
   pip install -e .
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   python -c "import photon_neuro; print('Installation successful!')"
   pytest tests/ -x
   ```

## Contributing Process

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports** - Help us identify and fix issues
2. **Feature Requests** - Suggest new functionality
3. **Code Contributions** - Implement features or fix bugs
4. **Documentation** - Improve or add documentation
5. **Examples** - Add tutorials and example notebooks
6. **Testing** - Improve test coverage and quality

### Contribution Workflow

1. **Create an Issue**
   - For bugs: Use the bug report template
   - For features: Use the feature request template
   - Discuss your proposal before starting work

2. **Development**
   - Create a feature branch: `git checkout -b feature/your-feature-name`
   - Make your changes following our coding guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Testing**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test categories
   pytest tests/ -m "not slow"  # Skip slow tests
   pytest tests/ -m integration  # Run integration tests
   
   # Check coverage
   pytest tests/ --cov=photon_neuro --cov-report=html
   ```

4. **Quality Checks**
   ```bash
   # Format code
   black photon_neuro/ tests/
   isort photon_neuro/ tests/
   
   # Lint
   flake8 photon_neuro/ tests/
   pylint photon_neuro/
   
   # Type checking
   mypy photon_neuro/
   
   # Security scanning
   bandit -r photon_neuro/
   ```

5. **Submit Pull Request**
   - Push your branch: `git push origin feature/your-feature-name`
   - Create a pull request using our template
   - Address review feedback

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some specific guidelines:

- **Line Length**: Maximum 127 characters
- **Imports**: Use isort for import sorting
- **Formatting**: Use Black for code formatting
- **Docstrings**: Use Google-style docstrings

### Example Code Style

```python
"""Example module showing code style."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from photon_neuro.core import PhotonicComponent


class ExampleComponent(PhotonicComponent):
    """Example photonic component demonstrating code style.
    
    This component serves as an example of proper code formatting,
    documentation, and structure within the Photon Neuromorphics SDK.
    
    Args:
        parameter1: Description of first parameter.
        parameter2: Description of second parameter with
            multi-line description.
        
    Attributes:
        attribute1: Description of attribute.
        
    Example:
        Basic usage example:
        
        >>> component = ExampleComponent(param1=1.0, param2="test")
        >>> result = component.process_data(input_data)
    """
    
    def __init__(self, parameter1: float, parameter2: str = "default"):
        super().__init__()
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self._internal_state = None
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Process input through the component.
        
        Args:
            input_field: Complex tensor representing optical field.
            
        Returns:
            Processed optical field tensor.
            
        Raises:
            ValueError: If input dimensions are invalid.
        """
        if input_field.dim() != 2:
            raise ValueError(f"Expected 2D input, got {input_field.dim()}D")
            
        # Process the input
        result = self._internal_processing(input_field)
        
        return result
        
    def _internal_processing(self, data: torch.Tensor) -> torch.Tensor:
        """Internal processing method."""
        # Implementation details
        return data * self.parameter1
        
    def to_netlist(self) -> Dict[str, Any]:
        """Export component as netlist."""
        return {
            "type": "example_component",
            "parameter1": self.parameter1,
            "parameter2": self.parameter2
        }
```

### Documentation Style

- Use Google-style docstrings for all public functions and classes
- Include examples in docstrings where helpful
- Document all parameters, return values, and exceptions
- Use type hints for all function signatures

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(core): add MZI mesh decomposition algorithm
fix(simulation): resolve FDTD boundary conditions
docs(readme): update installation instructions
test(networks): add tests for SNN training
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Testing

### Test Categories

We use pytest markers to categorize tests:

- `@pytest.mark.fast` - Quick unit tests (< 1s)
- `@pytest.mark.slow` - Slower integration tests (> 10s)
- `@pytest.mark.gpu` - Tests requiring GPU
- `@pytest.mark.hardware` - Tests requiring actual hardware
- `@pytest.mark.integration` - Integration tests

### Writing Tests

```python
import pytest
import torch
from photon_neuro.core import SiliconWaveguide


class TestSiliconWaveguide:
    """Test suite for SiliconWaveguide component."""
    
    def test_waveguide_creation(self):
        """Test basic waveguide creation."""
        wg = SiliconWaveguide(length=1e-3, width=450e-9)
        assert wg.length == 1e-3
        assert wg.width == 450e-9
        
    @pytest.mark.slow
    def test_long_propagation(self):
        """Test propagation over long distances."""
        wg = SiliconWaveguide(length=1e-2)  # 1 cm
        input_field = torch.ones(1000, dtype=torch.complex64)
        output = wg.forward(input_field)
        assert output.shape == input_field.shape
        
    @pytest.mark.gpu
    def test_gpu_acceleration(self):
        """Test GPU-accelerated simulation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device('cuda')
        wg = SiliconWaveguide(length=1e-3).to(device)
        input_field = torch.ones(100, dtype=torch.complex64, device=device)
        output = wg.forward(input_field)
        assert output.device == device
        
    @pytest.fixture
    def sample_waveguide(self):
        """Fixture providing a sample waveguide."""
        return SiliconWaveguide(length=1e-3, width=500e-9)
```

### Test Coverage

- Maintain >85% test coverage
- Cover edge cases and error conditions
- Include performance benchmarks for critical paths

## Documentation

### Types of Documentation

1. **API Documentation** - Automatically generated from docstrings
2. **User Guide** - High-level usage documentation
3. **Tutorials** - Step-by-step examples
4. **Developer Guide** - Implementation details
5. **Examples** - Jupyter notebooks and scripts

### Building Documentation

```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Provide mathematical context where relevant
- Link to relevant papers and references

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up-to-date with main

### PR Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks** - CI/CD pipeline runs
2. **Code Review** - At least one maintainer reviews
3. **Testing** - Reviewer tests functionality
4. **Approval** - Maintainer approves changes
5. **Merge** - Squash and merge to main

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, package versions
- **Steps to Reproduce**: Minimal example
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Additional Context**: Screenshots, logs, etc.

### Feature Requests

Use the feature request template and include:

- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: High-level design
- **Alternatives Considered**: Other approaches
- **Additional Context**: References, examples

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussion
- **Slack**: Real-time chat (invitation only)
- **Email**: Contact maintainers directly

### Getting Help

1. Check existing documentation and examples
2. Search GitHub issues for similar problems
3. Ask in GitHub Discussions
4. Create a new issue with detailed information

### Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments
- GitHub contributor graphs

## Development Guidelines

### Architecture Principles

1. **Modularity**: Components should be loosely coupled
2. **Extensibility**: Easy to add new components
3. **Performance**: Optimize for speed and memory
4. **Reliability**: Handle errors gracefully
5. **Usability**: Simple, intuitive APIs

### Performance Considerations

- Use vectorized operations where possible
- Leverage GPU acceleration for heavy computations
- Profile performance-critical code
- Cache expensive computations
- Minimize memory allocations

### Security Guidelines

- Never commit secrets or credentials
- Validate all user inputs
- Use secure coding practices
- Regular security audits with bandit

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Git tag created
- [ ] PyPI package published
- [ ] Docker images built
- [ ] GitHub release created

Thank you for contributing to Photon Neuromorphics SDK! 🌟🧠