#!/bin/bash

# Build the Docker image if it doesn't exist
if ! docker image inspect indy7-env &>/dev/null; then
    echo "Building Docker image..."
    docker build -t indy7-env .
fi

#export DISPLAY="127.0.0.1:10.0"
xhost + #+local:root
# Allow X11 connections from any host

# Run the Docker container with X11 forwarding
docker run -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    --network host \
    --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    indy7-env

    
