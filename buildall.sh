#!/bin/bash

# Source ROS 2
source /opt/ros/humble/setup.bash

# Build only indy7_msgs and mujoco_sim packages
colcon build --symlink-install --packages-select indy7_msgs mujoco_sim

# Source the workspace
source install/setup.bash

# Optional: Echo success message
echo "Build completed for indy7_msgs and mujoco_sim. Remember to source the workspace with:"
echo "source install/setup.bash"
