FROM nvcr.io/nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Add build arguments for UID and GID
ARG USER_UID=1000
ARG USER_GID=1000

# Some general settings for the headless systems
RUN apt-get update || true && \
  apt-get install --no-install-recommends -y locales && \
  locale-gen en_US en_US.UTF-8 && \
  update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
  export LANG=en_US.UTF-8 && \
  locale  # verify settings && \
  echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections && \
  rm -rf /var/lib/apt/lists/*

ENV DEBIAN_frontend=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Install NVIDIA driver, required by ZED SDK
RUN apt-get update && \
    apt-get install -y nvidia-driver-565 && \
    rm -rf /var/lib/apt/lists/*

# General ZED and Python dependencies
RUN apt-get update && \
    apt-get install -y sudo zstd wget git libturbojpeg* build-essential cmake lsb-release python3-pip python3-wheel python3-tk && \
    rm -rf /var/lib/apt/lists/*


# Download and install ZED SDK
RUN wget -q -O ZED_SDK_Linux_Ubuntu.run https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.0/ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run && \
    chmod +x ZED_SDK_Linux_Ubuntu.run && \
    ./ZED_SDK_Linux_Ubuntu.run -- silent skip_cuda


# Create a non-root user with specified UID/GID
RUN adduser --disabled-password --gecos '' --uid ${USER_UID} --gid ${USER_GID} captain && \
    adduser captain sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# Hack for making ZED SDK available to the non-root user
RUN chmod -R a+rx /usr/local/zed/include/ && \
    chmod -R a+rx /usr/local/zed/lib/ && \
    chmod -R a+rx /usr/local/zed/firmware/ && \
    chmod -R a+rx /usr/local/zed/settings/ && \
    chmod -R a+rwx /usr/local/lib/python3.10/dist-packages/*

USER captain
WORKDIR /home/captain/
RUN chmod a+rwx /home/captain/

# Install common dependencies
RUN pip3 install --user fire matplotlib tqdm pandas opencv-python==4.10.0.82 git+https://github.com/lilohuang/PyTurboJPEG.git numpy==1.26.4
