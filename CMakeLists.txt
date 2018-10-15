cmake_minimum_required(VERSION 3.10)
project(deletethis)
find_package(OpenCL REQUIRED)
set(CMAKE_CXX_STANDARD 11)

add_executable(deletethis main.cpp)
target_link_libraries(deletethis OpenCL::OpenCL)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel.cl COPYONLY)