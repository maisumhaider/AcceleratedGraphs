cmake_minimum_required(VERSION 3.10)
project(NeuralSynthesis)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
add_subdirectory(pugixml-1.9)
include_directories(pugixml-1.9/src)
include_directories(headers)
include_directories(sources)

set(HEADERS headers/ComputationGraph.h headers/ComputationNode.hpp headers/ComputationRuntime.h
        headers/Compute_Platform.h headers/utils.h headers/CompGraphParser.h )
set(PUGI_THIRD_PARTY pugixml-1.9/src/pugiconfig.hpp pugixml-1.9/src/pugixml.hpp pugixml-1.9/src/pugixml.cpp)
set(SOURCES main.cpp
        sources/CompGraphParser.cpp
        sources/ComputationGraph.cpp
        sources/ComputationNode.cpp
        sources/ComputationRuntime.cpp
        sources/Compute_Platform.cpp
        sources/utils.cpp)
add_executable(NeuralSynthesis ${HEADERS} ${PUGI_THIRD_PARTY} ${SOURCES})
target_link_libraries(NeuralSynthesis OpenCL::OpenCL )
