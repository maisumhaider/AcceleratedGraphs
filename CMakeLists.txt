cmake_minimum_required(VERSION 3.12)
project(NeuralSynthesis)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)

add_executable(NeuralSynthesis main.cpp)
target_link_libraries(NeuralSynthesis OpenCL::OpenCL)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/Kernels/vadd.cl ${CMAKE_CURRENT_BINARY_DIR}/vadd.cl COPYONLY)