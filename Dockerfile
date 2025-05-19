FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies
RUN apt update -y && apt install -y \
    build-essential \
    wget \
    gpg \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libqt5widgets5 \
    libtbb12 

# ----------------------------------
# 1. Install Intel Graphic Drivers
# ----------------------------------
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics
RUN apt install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
RUN apt install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
RUN apt install -y libze-dev intel-ocloc

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
RUN apt install  -y   python3-setuptools 
RUN pip install --no-cache-dir --break-system-packages \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino
RUN pip install --no-cache-dir --break-system-packages \
    nncf \
    cap_from_youtube 

RUN pip install --break-system-packages \
	torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --break-system-packages \
	openvino==2024.6.0 tqdm

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
