# Multi-stage build for Photon Neuromorphics SDK
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Run tests to ensure everything works
RUN python -m pytest tests/ -x --disable-warnings

# Production stage
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/photon-neuro/bin:$PATH"

# Create non-root user
RUN groupadd --gid 1000 photon && \
    useradd --uid 1000 --gid photon --shell /bin/bash --create-home photon

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /opt/photon-neuro

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=photon:photon photon_neuro ./photon_neuro/
COPY --chown=photon:photon setup.py README.md LICENSE ./
COPY --chown=photon:photon examples ./examples/
COPY --chown=photon:photon docs ./docs/

# Install package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /opt/photon-neuro/data /opt/photon-neuro/models && \
    chown -R photon:photon /opt/photon-neuro

# Switch to non-root user
USER photon

# Set up environment
ENV HOME=/home/photon
ENV PHOTON_NEURO_HOME=/opt/photon-neuro
ENV PHOTON_NEURO_DATA_DIR=/opt/photon-neuro/data
ENV PHOTON_NEURO_MODEL_DIR=/opt/photon-neuro/models

# Expose port for web interface (if any)
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import photon_neuro; print('Photon Neuromorphics SDK is healthy')" || exit 1

# Default command
CMD ["python", "-m", "photon_neuro.cli", "--help"]

# Development stage
FROM production as development

USER root

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Install Jupyter and development tools
RUN pip install jupyter jupyterlab ipywidgets

# Switch back to non-root user
USER photon

# Set up Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# GPU-enabled stage
FROM production as gpu

USER root

# Install CUDA runtime (simplified - adjust version as needed)
RUN apt-get update && apt-get install -y \
    nvidia-cuda-runtime \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

USER photon

# Verify GPU support
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"