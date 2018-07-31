cmake_minimum_required(VERSION 3.10)
project(AcceleratedGraphs)
set(CMAKE_CXX_STANDARD 17)

file(GLOB MY_KERNELS
        "kernel/*.cl")
file(COPY ${MY_KERNELS} DESTINATION .)
find_package(OpenCL REQUIRED)

add_executable(
        AcceleratedGraphs
        main.cpp
)
target_link_libraries(AcceleratedGraphs OpenCL)




