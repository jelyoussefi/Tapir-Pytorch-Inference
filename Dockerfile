# Base image with Intel Extension for PyTorch and XPU support
FROM intel/intel-extension-for-pytorch:2.7.10-xpu

# Set non-interactive frontend for Debian package installation
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------
# 1. Install System Dependencies
# ----------------------------------
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    wget \
    gpg \
    libtbb12 \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------
# 2. Install NPU Driver
# ----------------------------------
WORKDIR /tmp
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-fw-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-level-zero-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    dpkg -i *.deb && \
    rm -f *.deb

# ----------------------------------
# 3. Install Python Dependencies
# ----------------------------------
RUN pip install --no-cache-dir \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    cap_from_youtube \
    onnx \
    onnxruntime \
    onnxsim

RUN apt-get update -y && apt-get install -y \
    libqt5widgets5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tqdm \
    pandas \
    matplotlib \
    seaborn \
    nncf

# ----------------------------------
# 4. Download Sample Video
# ----------------------------------
WORKDIR /opt/videos
RUN wget https://videos.pexels.com/video-files/8624901/8624901-hd_1920_1080_30fps.mp4 && \
    mv 8624901-hd_1920_1080_30fps.mp4 horse.mp4

# ----------------------------------
# 5. Set Working Directory
# ----------------------------------
WORKDIR /workspace

# ----------------------------------
# 6. Add Metadata Labels
# ----------------------------------
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Docker image for YOLOX inference with Intel XPU and MMDeploy"
