#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Set up the stable repository
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker

    # Add user to docker group
    sudo usermod -aG docker $USER

    echo "Docker installation completed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Project root: $PROJECT_ROOT"

# Build and start the application
echo "Building and starting the application..."
cd "$PROJECT_ROOT"
docker-compose -f deployment/config/docker-compose.prod.yml down || true
docker-compose -f deployment/config/docker-compose.prod.yml up --build -d

# Wait for the application to start
echo "Waiting for the application to start..."
sleep 10

# Check if the application is running
if docker-compose -f deployment/config/docker-compose.prod.yml ps | grep -q "Up"; then
    echo "Deployment completed successfully!"
    echo "Application is running on port 7040"
else
    echo "Error: Application failed to start"
    docker-compose -f deployment/config/docker-compose.prod.yml logs
    exit 1
fi

# Print the logs
echo "Recent logs:"
docker-compose -f deployment/config/docker-compose.prod.yml logs --tail=50