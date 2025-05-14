FROM intel/intel-extension-for-pytorch:2.7.10-xpu

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# 1. Install system dependencies
# ----------------------------
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    wget \
    gpg \
    libtbb12 \
    python3-opencv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 4. Install NPU Driver
# ----------------------------
WORKDIR /tmp
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-fw-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-level-zero-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    dpkg -i *.deb && rm *.deb

# ----------------------------
# 7. Install Python dependencies
# ----------------------------
RUN pip install --no-cache-dir \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino

RUN pip install --no-cache-dir "numpy<2.0.0" cap_from_youtube onnx onnxruntime onnxsim

RUN apt update
RUN apt install -y libqt5widgets5
RUN pip install tqdm
RUN pip install pandas
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install nncf
# ----------------------------
# 9. Set working directory
# ----------------------------
WORKDIR /workspace
