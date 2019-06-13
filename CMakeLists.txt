cmake_minimum_required(VERSION 3.0)
project(GeneralFeatureTracker
VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(GeneralFeatureTracker)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

option( BUILD_EXAMPLES "Build Tools" OFF)
if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()