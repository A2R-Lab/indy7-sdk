FROM ros:humble-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-numpy \
    python3-pip \
    ros-humble-urdfdom \
    ros-humble-hpp-fcl \
    ros-humble-urdfdom-headers \
    python3-colcon-common-extensions \
    python3-rosdep \
    libxinerama-dev \
    libglfw3-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    x11-apps \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxfixes-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# # Install pybind11
# RUN pip3 install pybind11

# Install MuJoCo
RUN git clone https://github.com/deepmind/mujoco.git \ 
    && cd mujoco \
    && mkdir build \
    && cd build \
    && cmake .. \
    && cmake --build . \
    && cmake --install .

# Set working directory
WORKDIR /workspace

# Source ROS environment in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash" >> ~/.bashrc

# Set entrypoint to bash
ENTRYPOINT ["/bin/bash"]