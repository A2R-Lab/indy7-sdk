from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    driver_node = Node(
        package='robot_driver',
        executable='robot_driver_node',
        name='robot_driver',
        output='screen'
    )

    trajopt_node = Node(
        package='trajopt',
        executable='trajopt_node',
        name='trajopt',
        arguments=["indy7-control/src/trajopt/trajectories/figure8_traj_eePos_meters.csv"],
        output='screen'
    )

    return LaunchDescription([
        driver_node,
        trajopt_node
    ])
