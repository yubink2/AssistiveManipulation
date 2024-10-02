# Use the NVIDIA CUDA base image for Ubuntu 22.04 with CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install dependencies and Miniconda
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget bzip2 git \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    freeglut3-dev \
    libosmesa6-dev \
    patchelf \
    x11-apps \
    libedit-dev \
    libffi-dev \
    libncurses-dev \
    libssl-dev \
    tk-dev \
    zlib1g-dev \
    xz-utils \
    wget \
    bzip2 \
    ca-certificates

# Install Python 3.8 and other packages via pip if not available in apt
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install specific versions of dependencies using pip or apt
RUN python3 -m pip install \
    pip==24.2 \
    setuptools==75.1.0 \
    wheel==0.44.0

# Install other dependencies
RUN pip install numpy==1.23.5
RUN pip install trimesh
RUN pip install autolab_core
RUN pip install h5py
RUN pip install pyrender
RUN pip install pyglet==1.5.27
RUN pip install python-fcl
RUN pip install open3d
RUN pip install pybullet

# Install PyTorch, Torchvision, and Torchaudio with CUDA 11.8 support
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install scipy
RUN pip install wheel
RUN pip install pytorch-kinematics
RUN pip install ipywidgets

# Clone the AssistiveManipulation repository
RUN git clone https://github.com/yubink2/AssistiveManipulation.git

# Set the working directory to the cloned repository
WORKDIR /app/AssistiveManipulation

# Install the 'csdf' module from the 'extern/csdf' directory
RUN (cd extern/csdf && pip install .)

# Install pytorch3d (this may take some time)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Set the display environment variable to allow PyBullet GUI to run
ENV DISPLAY=:0

# Expose the PyBullet GUI port
EXPOSE 8080