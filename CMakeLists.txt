cmake_minimum_required(VERSION 3.0)
project(GIFT
VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(GIFT)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

option( BUILD_EXAMPLES "Build Tools" ON)
if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()