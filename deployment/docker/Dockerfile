# Use Python 3.9 slim as base
FROM python:3.9-slim-buster

# Set up a non-root user and workspace
RUN useradd appuser && mkdir /app
WORKDIR /app

# Make environment non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV MPLBACKEND=Agg
ENV PYTHONPATH=/app:/app/roug_ml:/app/cancer_subtype_prediction
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies (Added `libjpeg-dev`, `libpng-dev`, `libopenblas-dev`)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-tk \
    libssl-dev \
    libffi-dev \
    git \
    curl \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for some packages)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create necessary directories for matplotlib
RUN mkdir -p /home/appuser/.config/matplotlib && \
    mkdir -p /home/appuser/.aws && \
    chown -R appuser:appuser /home/appuser
    # Add this line with the other mkdir commands

# Copy requirements first to leverage Docker cache
COPY model-serving/requirements.tcga.txt .

# Install dependencies in stages to better handle conflicts
RUN pip install --upgrade pip && \
    # Core data science
    pip install 'numpy==1.24.3' && \
    pip install 'scipy==1.10.1' && \
    pip install 'pandas==2.0.3' && \
    pip install 'scikit-learn==1.1.3' && \
    pip install 'fsspec==2024.3.1' && \
    # DL/ML packages
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers==4.41.2 && \
    # Install the rest
    pip install -r requirements.tcga.txt

# Copy application code
COPY cancer_subtype_prediction /app/cancer_subtype_prediction
COPY roug_ml /app/roug_ml/
COPY model-serving /app/model-serving/

# Create necessary directories and set permissions
RUN mkdir -p /app/model-serving/gtf_files && \
    mkdir -p /app/results/tcga && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7040

# Start Gunicorn with 4 workers and 300s timeout
CMD ["gunicorn", "--bind", "0.0.0.0:7040", "--workers", "4", "--timeout", "300", "cancer_subtype_prediction.tcga_app:app"]
