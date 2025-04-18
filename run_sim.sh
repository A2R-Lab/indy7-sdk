#!/bin/bash

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source the workspace
if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "Workspace not built. Building now..."
    ./buildall.sh
    source install/setup.bash
fi

echo "Running MuJoCo simulator..."
ros2 run mujoco_sim mujoco_sim_node $(pwd)/src/mujoco_sim/models/indy7.xml 0.01 true