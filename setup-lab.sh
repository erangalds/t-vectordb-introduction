#!/bin/bash

# Function to clear the previous lab setup
clear_lab() {
    echo "Stopping and removing containers..."
    containers=("vectordb-lab-dev-coding" "vectordb-lab-postgres-db" "vectordb-lab-elastic" "vectordb-lab-chroma-db")
    for container in "${containers[@]}"; do
        docker stop "$container" 2>/dev/null
        docker rm "$container" 2>/dev/null
        echo "Container $container stopped and removed."
    done

    echo "Removing volumes..."
    volumes=("vectordb-source-code-base" "vectordb-postgres-db" "vectordb-elastic-db" "vectordb-chroma-db")
    for volume in "${volumes[@]}"; do
        docker volume rm "$volume" 2>/dev/null
        echo "Volume $volume removed."
    done

    echo "Previous lab setup cleared."
}

# Function to start Docker Compose services
start_docker_compose() {
    echo "Starting Docker Compose services..."
    docker-compose up -d
    echo "Docker Compose services started."
}

# Function to copy data to the container
copy_data() {
    echo "Copying data to the container..."
    container_name="vectordb-lab-dev-coding"
    source_path="/temp-data/"
    destination_path="/source-code/data/"
    docker exec $container_name bash -c "mkdir -p $destination_path"
    docker exec $container_name bash -c "cp -r $source_path* $destination_path/"
    echo "Data copied to the container."
}

# Main script execution
clear_lab
start_docker_compose
sleep 10  # Wait for containers to be fully up
copy_data

echo "Lab setup completed successfully."
