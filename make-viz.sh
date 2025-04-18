#!/bin/bash

source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select mujoco_sim  --cmake-args -DENABLE_VISUALIZATION=ON

source install/setup.bash

echo "Build completed for mujoco_sim. Source the workspace with:"
echo "      source install/setup.bash"
