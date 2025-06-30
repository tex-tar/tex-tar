#!/bin/bash

IMAGE_NAME="synthetic_qa_image"
CONTAINER_NAME="textar"
DOCKERFILE_PATH="/projects/data/vision-team/swaroopa_jinka/TEXTAR/Dockerfile"
CODE_MOUNT="/projects/data/vision-team/swaroopa_jinka/TEXTAR"
DATA_MOUNT="/projects/data/vision-team/swaroopa_jinka/data/"
HF_CACHE="/projects/data/vision-team/swaroopa_jinka/huggingface-cache/"
OUTPUT_MOUNT="/projects/data/vision-team/swaroopa_jinka/data/textar_outputs"
ENVIRONMENTS_MOUNT="/projects/data/vision-team/swaroopa_jinka/envs"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Docker image '$IMAGE_NAME' not found. Building..."
    docker build \
        --build-arg UID=$(id -u) \
        --build-arg GID=$(id -g) \
        -f $DOCKERFILE_PATH \
        -t $IMAGE_NAME .
else
    echo "Docker image '$IMAGE_NAME' already exists."
fi


CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)

if [[ "$CONTAINER_STATUS" == "running" ]]; then
    echo "Container '$CONTAINER_NAME' is already running."
elif [[ "$CONTAINER_STATUS" == "exited" ]]; then
    echo "Container '$CONTAINER_NAME' exists but is stopped. Starting it..."
    docker start $CONTAINER_NAME
else
    echo "Creating and starting new container '$CONTAINER_NAME'..."
    docker run -dit \
        --name $CONTAINER_NAME \
        --gpus all \
        --network host \
        -v "$CODE_MOUNT:/code" \
        -v "$DATA_MOUNT:/data" \
        -v "$HF_CACHE:/home/swaroopa_jinka/hf_cache" \
        -v "$OUTPUT_MOUNT:/metrics_output" \
        -v "$ENVIRONMENTS_MOUNT:/environments" \
        $IMAGE_NAME
fi


echo "Attaching to container '$CONTAINER_NAME'..."
docker exec -it $CONTAINER_NAME bash