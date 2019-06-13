cmake_minimum_required(VERSION 3.0)
project(GeneralFeatureTracker
VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(GeneralFeatureTracker)

option( BUILD_EXAMPLES "Build Tools" OFF)
if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()