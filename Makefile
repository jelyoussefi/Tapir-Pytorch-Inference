# Settings
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CACHE_DIR := $(CURRENT_DIR)/.cache
MODELS_DIR := /workspace/models

DEVICE ?= GPU


# Docker Configuration
DOCKER_IMAGE_NAME := tapir_inference
export DOCKER_BUILDKIT := 1

DOCKER_RUN_PARAMS := \
	-it --rm \
	--network=host \
	-a stdout -a stderr \
	--privileged \
	-v /dev:/dev \
	-e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v $(CURRENT_DIR):/workspace \
	-w /workspace \
	$(DOCKER_IMAGE_NAME)

DOCKER_BUILD_PARAMS := \
	--rm \
	--network=host \
	-t $(DOCKER_IMAGE_NAME) . 

# Targets
.PHONY: default build run bash models

default: run

build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME)..."
	@docker build ${DOCKER_BUILD_PARAMS}

run: 	build
	@xhost +local:docker
	@echo "🚀 Running Tapir Inference demo ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./example_video_tracking.py -m ./models/causal_bootstapir_checkpoint.pt -i ./videos/streat.mp4 -d ${DEVICE}"

run2: 	build
	@xhost +local:docker
	@echo "🚀 Running Tapir Inference demo ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./example_video_tracking.py -m ./models/causal_bootstapir_checkpoint.pt -i ./videos/streat.mp4 -d ${DEVICE} "

bash: build
	@echo "🐚 Starting bash in container ..."
	@docker run $(DOCKER_RUN_PARAMS) bash
