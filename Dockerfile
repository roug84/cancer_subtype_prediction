# Use the official python image
FROM python:3.8-slim-buster

# Set up a non-root user and workspace
RUN useradd appuser && mkdir /app
WORKDIR /app

# Make environment non-interactive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential libssl-dev libffi-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files into the container
COPY cancer_subtype_prediction /app/cancer_subtype_prediction
COPY roug_ml /app/roug_ml/
COPY data /app/data/
COPY results /app/results/

RUN apt-get update && apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r cancer_subtype_prediction/requirements.txt

# Additional Python packages
RUN pip install gunicorn flask python-dotenv pandas requests

# Set the PYTHONPATH
ENV PYTHONPATH=/app:/app/cancer_subtype_prediction:/app/roug_ml

# Change ownership of the app directory, create results directory, and create uploads directory
USER root
RUN chown -R appuser:appuser /app #&&
    #mkdir -p /results/tcga/TCGA_BRCA_vf_250 && chown -R appuser:appuser /results/tcga/TCGA_BRCA_vf_250 && \
#    mkdir -p /app/uploads && chown -R appuser:appuser /app/uploads  # <-- Added this line for the uploads directory

# Switch to the app user for better security
USER appuser

# Create and adjust permissions for the gtf_files directory
RUN mkdir -p /app/gtf_files && chown -R appuser:appuser /app/gtf_files

# Expose port 1000
EXPOSE 1000
#  export PYTHONPATH="${PYTHONPATH}:/Users/hector/DiaHecDev/roug_ml"

# Command to run the application:  gunicorn --timeout 0 -b 0.0.0.0:1000 tcga_app:app
CMD ["gunicorn", "--workers", "4", "--timeout", "0", "-b", "0.0.0.0:1000", "cancer_subtype_prediction.tcga_app:app"]

