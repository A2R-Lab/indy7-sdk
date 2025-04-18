cmake_minimum_required(VERSION 3.10)
project(mujoco_sim)

# Set OpenGL policy to prefer GLVND
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

# Option to enable/disable visualization
option(ENABLE_VISUALIZATION "Enable MuJoCo visualization with GLFW" ON)

# Set environment variable for package.xml conditions
set(ENV{ENABLE_VISUALIZATION} ${ENABLE_VISUALIZATION})

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
#find_package(indy7_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(mujoco REQUIRED)

# Only find GLFW if visualization is enabled
if(ENABLE_VISUALIZATION)
  find_package(glfw3 REQUIRED)
  add_definitions(-DENABLE_VISUALIZATION)
endif()

add_executable(mujoco_sim_node src/mujoco_sim_node.cpp)
target_compile_features(mujoco_sim_node PUBLIC c_std_99 cxx_std_17) 

target_include_directories(mujoco_sim_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(mujoco_sim_node
  mujoco::mujoco
)

# Only link GLFW if visualization is enabled
if(ENABLE_VISUALIZATION)
  target_link_libraries(mujoco_sim_node
    glfw
  )
endif()

ament_target_dependencies(mujoco_sim_node
  rclcpp
  #indy7_msgs
  sensor_msgs
)

install(TARGETS mujoco_sim_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
