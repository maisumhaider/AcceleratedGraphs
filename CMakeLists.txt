cmake_minimum_required(VERSION 3.10)
project(compGraphs)
find_package(OpenCL REQUIRED)
set(CMAKE_CXX_STANDARD 11)

add_executable(compGraphs
        main.cpp)
target_link_libraries(compGraphs OpenCL::OpenCL)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/kernel/kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel.cl COPYONLY)