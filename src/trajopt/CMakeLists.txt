cmake_minimum_required(VERSION 3.10)
project(trajopt LANGUAGES CXX CUDA)

add_compile_options(-Wall -Wextra -Wpedantic)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(indy7_msgs REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.2/bin/nvcc" CACHE PATH "CUDA compiler")

# Include directories
include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/MPCGPU/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/MPCGPU/include/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/MPCGPU/GBD-PCG/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../external/MPCGPU/GLASS

)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-diag-suppress=970,737)

# Add the executable
cuda_add_executable(trajopt_node
  src/trajopt_node.cu
)


# Link libraries
target_link_libraries(trajopt_node
  ${CUDA_LIBRARIES} cublas
)

ament_target_dependencies(trajopt_node
  rclcpp
  indy7_msgs
)

# Install targets
install(TARGETS
  trajopt_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() 