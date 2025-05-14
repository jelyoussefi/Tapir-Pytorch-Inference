# Settings
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
DATASET_DIR := /workspace/dataset
MODELS_DIR := /workspace/models

DEVICE ?= CPU

INPUT_SIZE  ?= 480
NUM_POINTS ?= 100
NUM_SAMPLES ?= 100

PRECISION ?= FP32

PYTORCH_MODEL ?=  $(MODELS_DIR)/$(PRECISION)/tapir.pt
OPENVINO_MODEL ?= $(MODELS_DIR)/$(PRECISION)/tapir.xml

INPUT ?= ./videos/horse.mp4

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
.PHONY: default build run bash dataset models export ov

default: run

build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME)..."
	@docker build ${DOCKER_BUILD_PARAMS}

run : 	build
	@xhost +local:docker
	@echo "🚀 Running Tapir Inference demo in $(PRECISION) ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./tracker.py -m $(PYTORCH_MODEL) -i $(INPUT) -d ${DEVICE} -r $(INPUT_SIZE) -n $(NUM_POINTS) -p $(PRECISION)"

ov : 	build
	@xhost +local:docker
	@echo "🚀 Running Tapir Inference demo in $(PRECISION) ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./tracker.py -m $(OPENVINO_MODEL) -i $(INPUT) -d ${DEVICE} -r $(INPUT_SIZE) -n $(NUM_POINTS) -p $(PRECISION)"

export: 	build
	@xhost +local:docker
	@echo "🚀 Exporting the Pytorch model to ONNX ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python ./onnx_export.py --model $(MODELS_DIR)/FP32/tapir.pt --resolution $(INPUT_SIZE) --num_points $(NUM_POINTS) --output_dir $(MODELS_DIR)/FP32/ && \
		 cd $(MODELS_DIR)/FP32/ && ovc tapir.onnx"

models: build
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"./download_models.sh ${MODELS_DIR}/FP32"
		
dataset: build
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"./prepare_dataset.sh ${DATASET_DIR}"


quantize:  build models dataset
	@echo "🚀 Quantizing ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./quantize.py -m $(MODELS_DIR)/FP32/tapir.xml --resize $(INPUT_SIZE) $(INPUT_SIZE) --num_samples $(NUM_SAMPLES) --output  $(MODELS_DIR)/INT8/tapir.xml"

eval: 	build
	@echo "🚀 Evaluating ..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c \
		"python3 ./eval.py -m $(PYTORCH_MODEL) --device ${DEVICE}"

bash: build
	@echo "🐚 Starting bash in container ..."
	@docker run $(DOCKER_RUN_PARAMS) bash
