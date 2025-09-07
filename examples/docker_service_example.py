#!/usr/bin/env python3
"""Docker deployment example for Excel-to-CSV Converter.

This script demonstrates how to deploy the Excel-to-CSV converter
as a containerized service using Docker.
"""

# Dockerfile content
dockerfile_content = '''
FROM python:3.11-slim

LABEL maintainer="Excel-to-CSV Team"
LABEL description="Excel-to-CSV Converter Service"

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY excel_to_csv_converter.py .
COPY pyproject.toml .

# Install application
RUN pip install -e .

# Create directories for data
RUN mkdir -p /data/input /data/output /app/logs

# Create non-root user
RUN groupadd -r excel2csv && useradd -r -g excel2csv excel2csv
RUN chown -R excel2csv:excel2csv /app /data
USER excel2csv

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; from excel_to_csv.cli import main; sys.exit(0)"

# Default command
CMD ["python", "excel_to_csv_converter.py", "service", "--config", "config/docker.yaml"]

# Expose ports if needed (for monitoring endpoints)
# EXPOSE 8080
'''

# Docker Compose configuration
docker_compose_content = '''
version: '3.8'

services:
  excel-to-csv:
    build: .
    container_name: excel-to-csv-service
    restart: unless-stopped
    
    volumes:
      # Mount input and output directories
      - ./data/input:/data/input:ro
      - ./data/output:/data/output:rw
      - ./config:/app/config:ro
      - ./logs:/app/logs:rw
    
    environment:
      # Environment variable overrides
      - EXCEL_TO_CSV_LOG_LEVEL=INFO
      - EXCEL_TO_CSV_PROCESSING_MAX_CONCURRENT=4
    
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "from excel_to_csv import __version__; print(__version__)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional: Add monitoring service
  # monitoring:
  #   image: prom/prometheus
  #   ports:
  #     - "9090:9090"
  #   volumes:
  #     - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
'''

# Docker-specific configuration
docker_config_content = '''
# Docker-specific configuration
monitoring:
  folders:
    - "/data/input"
  file_patterns:
    - "*.xlsx"
    - "*.xls"
  process_existing: true

output:
  folder: "/data/output"
  naming_pattern: "{timestamp}_{filename}_{worksheet}.csv"
  include_timestamp: true

confidence:
  threshold: 0.9

processing:
  max_concurrent: 4
  max_file_size_mb: 100

logging:
  level: "INFO"
  format: "json"
  file:
    path: "/app/logs/excel_to_csv.log"
    rotation_size: "50MB"
    retention_days: 30
  console:
    enabled: true
    level: "INFO"
'''

# Deployment script
deployment_script = '''#!/bin/bash
# Docker deployment script for Excel-to-CSV Converter

set -e

echo "Deploying Excel-to-CSV Converter with Docker..."

# Create required directories
mkdir -p data/input data/output logs config

# Create docker configuration
cat > config/docker.yaml << 'EOF'
{docker_config_content}
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
{dockerfile_content}
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
{docker_compose_content}
EOF

# Build and start the service
echo "Building Docker image..."
docker-compose build

echo "Starting Excel-to-CSV service..."
docker-compose up -d

echo "Checking service status..."
docker-compose ps

echo "Viewing logs..."
docker-compose logs -f excel-to-csv

echo ""
echo "Service deployed successfully!"
echo ""
echo "To manage the service:"
echo "  Start:   docker-compose up -d"
echo "  Stop:    docker-compose down"
echo "  Logs:    docker-compose logs -f excel-to-csv"
echo "  Status:  docker-compose ps"
echo ""
echo "Data directories:"
echo "  Input:   ./data/input"
echo "  Output:  ./data/output"
echo "  Logs:    ./logs"
echo "  Config:  ./config"
'''.format(
    dockerfile_content=dockerfile_content.strip(),
    docker_compose_content=docker_compose_content.strip(),
    docker_config_content=docker_config_content.strip()
)


def main():
    """Create Docker deployment files."""
    print("Excel-to-CSV Converter - Docker Deployment Example")
    print("=" * 60)
    print()
    
    print("This example shows how to deploy the Excel-to-CSV converter")
    print("as a containerized service using Docker and Docker Compose.")
    print()
    
    # Write deployment files
    files_to_create = {
        'Dockerfile': dockerfile_content.strip(),
        'docker-compose.yml': docker_compose_content.strip(),
        'config/docker.yaml': docker_config_content.strip(),
        'deploy.sh': deployment_script.strip()
    }
    
    print("Files that would be created:")
    for filename, content in files_to_create.items():
        lines = len(content.split('\n'))
        print(f"  - {filename} ({lines} lines)")
    
    print()
    print("To deploy:")
    print("1. Copy these files to your project directory")
    print("2. Run: chmod +x deploy.sh")
    print("3. Run: ./deploy.sh")
    print()
    print("Docker deployment features:")
    print("  ✓ Containerized service with health checks")
    print("  ✓ Volume mounts for data persistence")
    print("  ✓ Resource limits and monitoring")
    print("  ✓ Automatic restart on failure")
    print("  ✓ Structured logging with rotation")
    print("  ✓ Environment variable configuration")
    print()
    print("For production deployment, also consider:")
    print("  - Using Docker Swarm or Kubernetes")
    print("  - Adding monitoring and alerting")
    print("  - Setting up log aggregation")
    print("  - Implementing backup strategies")
    print("  - Configuring network security")


if __name__ == "__main__":
    main()