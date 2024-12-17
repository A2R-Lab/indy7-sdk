cmake_minimum_required(VERSION 3.10)
project(indy7_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointTrajectory.msg"
  "msg/JointTrajectoryPoint.msg"
  "msg/JointState.msg"
  "msg/JointCommand.msg"
  "msg/JointTorque.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

