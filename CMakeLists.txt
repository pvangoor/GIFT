cmake_minimum_required(VERSION 3.0)
project(GIFT
VERSION 0.2.0)
set(CMAKE_CXX_STANDARD 17)

# turn on optimisation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# enable neon vectorizatin for RPi4
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mfpu=neon -mfloat-abi=hard")

# use single precision floating point
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SINGLE_FLOAT")

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

option( BUILD_EXAMPLES "Build Examples" ON)
if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()

option( BUILD_TESTS "Build Tests" ON)
if(BUILD_TESTS)
    message("Building the tests...")
    add_subdirectory(test)
endif()
