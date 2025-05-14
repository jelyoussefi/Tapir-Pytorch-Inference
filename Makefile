# ----------------------------------
# General Settings
# ----------------------------------
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Directories
DATASET_DIR := /workspace/dataset
MODELS_DIR := /workspace/models

# Default Parameters
DEVICE      ?= CPU
INPUT_SIZE  ?= 480
NUM_POINTS  ?= 100
NUM_SAMPLES ?= 100
PRECISION   ?= FP32

# Model Paths
PYTORCH_MODEL  ?= $(MODELS_DIR)/$(PRECISION)/tapir.pt
OPENVINO_MODEL ?= $(MODELS_DIR)/$(PRECISION)/tapir.xml

# Input Video
INPUT ?= /opt/videos/horse.mp4

# ----------------------------------
# Docker Configuration
# ----------------------------------
DOCKER_IMAGE_NAME := tapir_inference
export DOCKER_BUILDKIT := 1

# Docker Build Parameters
DOCKER_BUILD_PARAMS := \
    --rm \
    --network=host \
    --build-arg http_proxy=$(HTTP_PROXY) \
    --build-arg https_proxy=$(HTTPS_PROXY) \
    --build-arg no_proxy=$(NO_PROXY) \
    -t $(DOCKER_IMAGE_NAME) .

# Docker Run Parameters
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
    -e http_proxy=$(HTTP_PROXY) \
    -e https_proxy=$(HTTPS_PROXY) \
    -e no_proxy=$(NO_PROXY) \
    $(DOCKER_IMAGE_NAME)

# ----------------------------------
# Targets
# ----------------------------------
.PHONY: default build run ov export models dataset quantize eval bash

# Default target: run inference
default: run

# Build the Docker image
build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME)..."
	@docker build $(DOCKER_BUILD_PARAMS)

# Run Tapir inference with PyTorch model
run: build
	@echo "🚀 Running Tapir Inference demo in $(PRECISION)..."
	@[ -n "$$DISPLAY" ] && xhost +local:root > /dev/null 2>&1 || true
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./tracker.py -m $(PYTORCH_MODEL) -i $(INPUT) -d $(DEVICE) -r $(INPUT_SIZE) -n $(NUM_POINTS) -p $(PRECISION)"

# Run Tapir inference with OpenVINO model
ov: build
	@echo "🚀 Running Tapir Inference demo with OpenVINO in $(PRECISION)..."
	@[ -n "$$DISPLAY" ] && xhost +local:root > /dev/null 2>&1 || true
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./tracker.py -m $(OPENVINO_MODEL) -i $(INPUT) -d $(DEVICE) -r $(INPUT_SIZE) -n $(NUM_POINTS) -p $(PRECISION)"

# Export PyTorch model to ONNX and convert to OpenVINO IR
export: build
	@echo "🚀 Exporting PyTorch model to ONNX and converting to OpenVINO IR..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python ./onnx_export.py --model $(MODELS_DIR)/FP32/tapir.pt --resolution $(INPUT_SIZE) --num_points $(NUM_POINTS) --output_dir $(MODELS_DIR)/FP32/ && \
		cd $(MODELS_DIR)/FP32/ && ovc tapir.onnx"

# Download models
models: build
	@echo "📥 Downloading models to $(MODELS_DIR)/FP32..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"./download_models.sh $(MODELS_DIR)/FP32"

# Prepare dataset
dataset: build
	@echo "📂 Preparing dataset in $(DATASET_DIR)..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"./prepare_dataset.sh $(DATASET_DIR)"

# Quantize the model to INT8
quantize: build models dataset
	@echo "⚙️ Quantizing model to INT8..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./quantize.py -m $(MODELS_DIR)/FP32/tapir.xml --resize $(INPUT_SIZE) $(INPUT_SIZE) --num_samples $(NUM_SAMPLES) --output $(MODELS_DIR)/INT8/tapir.xml"

# Evaluate the model
eval: build
	@echo "📊 Evaluating model..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./eval.py -m $(PYTORCH_MODEL) --device $(DEVICE)"

# Start an interactive bash session in the container
bash: build
	@echo "🐚 Starting bash in container..."
	@docker run $(DOCKER_RUN_PARAMS) bash