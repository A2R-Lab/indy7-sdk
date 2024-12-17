cmake_minimum_required(VERSION 3.10)
project(robot_driver LANGUAGES CXX)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(indy7_msgs REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
set(_GRPC_GRPCPP gRPC::grpc++)

set(_NEUROMEKA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/neuromeka-package/cpp/neuromeka_cpp)

include_directories(
  include
  ${_NEUROMEKA_CPP_DIR}
)

set(INDY_SOURCES
    ${_NEUROMEKA_CPP_DIR}/indydcp3.cpp
    ${_NEUROMEKA_CPP_DIR}/proto/common_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/config_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/control_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/device_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/ethercat_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/moby_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/rtde_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/hri_msgs.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/shared_msgs.pb.cc

    ${_NEUROMEKA_CPP_DIR}/proto/common_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/config_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/control_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/device_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/ethercat_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/moby_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/rtde_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/hri_msgs.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/shared_msgs.grpc.pb.cc

    ${_NEUROMEKA_CPP_DIR}/proto/config.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/control.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/device.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/ethercat.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/eyetask.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/indyeye.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/moby.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/rtde.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/cri.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/hri.pb.cc

    ${_NEUROMEKA_CPP_DIR}/proto/config.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/control.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/device.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/ethercat.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/eyetask.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/indyeye.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/moby.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/rtde.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/cri.grpc.pb.cc
    ${_NEUROMEKA_CPP_DIR}/proto/hri.grpc.pb.cc
)

# Create the robot driver node executable
add_executable(robot_driver_node 
  src/robot_driver_node.cpp
  src/joint_controller.cpp
  ${INDY_SOURCES}
)

target_link_libraries(robot_driver_node
  ${_PROTOBUF_LIBPROTOBUF}
  ${_GRPC_GRPCPP}
  ${_REFLECTION}
)

ament_target_dependencies(robot_driver_node
  rclcpp
  indy7_msgs
)

add_executable(sim_driver_node src/sim_driver_node.cpp)
ament_target_dependencies(sim_driver_node
  rclcpp
  indy7_msgs
)

install(TARGETS
  robot_driver_node
  sim_driver_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package() 