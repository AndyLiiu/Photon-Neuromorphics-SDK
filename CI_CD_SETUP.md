# CI/CD Setup Instructions

## 🚀 GitHub Actions Workflow Setup

Due to GitHub App permissions, the CI/CD workflow needs to be manually added to the repository. Follow these steps:

### Step 1: Create the Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Create the CI/CD Workflow File
Create `.github/workflows/ci-cd.yml` with the following content:

```yaml
name: Photon Neuromorphics CI/CD

on:
  push:
    branches: [ main, develop, 'terragon/*' ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  PYTHON_VERSION: '3.9'
  NODE_VERSION: '18'

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black flake8 mypy isort
        pip install -e ".[dev]"
    
    - name: Format check with Black
      run: black --check --diff .
    
    - name: Lint with flake8
      run: flake8 photon_neuro tests examples --max-line-length=100 --ignore=E501,W503
    
    - name: Type check with mypy
      run: mypy photon_neuro --ignore-missing-imports --no-strict-optional
    
    - name: Import sorting with isort
      run: isort --check-only --diff .

  test-python:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies (Ubuntu)
      if: matrix.os == 'ubuntu-latest'
      run: |
        sudo apt-get update
        sudo apt-get install -y libopenblas-dev libfftw3-dev
    
    - name: Install system dependencies (macOS)
      if: matrix.os == 'macos-latest'
      run: |
        brew install openblas fftw
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -e ".[dev,hardware]"
    
    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=photon_neuro --cov-report=xml --cov-report=term-missing
    
    - name: Run WASM integration tests
      run: |
        pytest tests/test_wasm_integration.py -v
    
    - name: Run performance benchmarks (quick)
      run: |
        pytest tests/test_performance_benchmarks.py::TestMZIPerformance::test_mzi_scaling_performance -v
    
    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
        pip install -e .
    
    - name: Run Bandit security scan
      run: |
        bandit -r photon_neuro -f json -o bandit-report.json || true
        bandit -r photon_neuro --severity-level medium
    
    - name: Check for known vulnerabilities
      run: |
        safety check --json --output safety-report.json || true
        safety check

  docker-build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: photonneuro/photon-neuromorphics
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
    
    - name: Build Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64
        push: false
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
    
    - name: Test Docker image
      run: |
        docker run --rm photonneuro/photon-neuromorphics:latest python -c "import photon_neuro; print('Docker image working:', photon_neuro.__version__)"
```

### Step 3: Configure Repository Secrets
Add these secrets to your GitHub repository settings:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token
- `PYPI_API_TOKEN`: Your PyPI API token for package publishing
- `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

### Step 4: Enable GitHub Actions
1. Go to your repository settings
2. Navigate to "Actions" → "General"
3. Ensure "Allow all actions and reusable workflows" is selected
4. Save the settings

### Step 5: Test the Workflow
1. Commit and push the workflow file
2. Create a test commit to trigger the workflow
3. Monitor the "Actions" tab in GitHub to see the workflow execution

## 🔧 Alternative CI/CD Solutions

If you prefer other CI/CD platforms, here are configurations:

### GitLab CI (.gitlab-ci.yml)
```yaml
stages:
  - test
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.9"

test:
  stage: test
  image: python:$PYTHON_VERSION
  script:
    - pip install -e ".[dev]"
    - pytest tests/ --cov=photon_neuro
  coverage: '/TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t photon-neuromorphics:$CI_COMMIT_SHA .
    - docker tag photon-neuromorphics:$CI_COMMIT_SHA photon-neuromorphics:latest
```

### CircleCI (.circleci/config.yml)
```yaml
version: 2.1

jobs:
  test:
    docker:
      - image: python:3.9
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            pip install -e ".[dev]"
      - run:
          name: Run tests
          command: |
            pytest tests/ --junitxml=test-results/junit.xml --cov=photon_neuro
      - store_test_results:
          path: test-results

workflows:
  test_and_build:
    jobs:
      - test
```

### Jenkins (Jenkinsfile)
```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'python -m venv venv'
                sh '. venv/bin/activate && pip install -e ".[dev]"'
            }
        }
        
        stage('Test') {
            steps {
                sh '. venv/bin/activate && pytest tests/ --junitxml=results.xml'
            }
            post {
                always {
                    junit 'results.xml'
                }
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t photon-neuromorphics:${BUILD_NUMBER} .'
            }
        }
    }
}
```

## 🚀 Quick Start Commands

After setting up CI/CD, test your setup:

```bash
# Run local tests
python -m pytest tests/ -v

# Run security scan
bandit -r photon_neuro

# Build Docker image
docker build -t photon-neuromorphics:local .

# Test installation
pip install -e ".[dev]"
python -c "import photon_neuro; print('Installation successful!')"
```

## 📋 Checklist

- [ ] Created `.github/workflows/ci-cd.yml`
- [ ] Added repository secrets
- [ ] Enabled GitHub Actions
- [ ] Tested workflow with a commit
- [ ] Verified all jobs pass
- [ ] Configured branch protection rules
- [ ] Set up status checks

Your CI/CD pipeline is now ready for production use! 🎉