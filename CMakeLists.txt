cmake_minimum_required(VERSION 3.13)
project(GeneralFeatureTracker
VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(GeneralFeatureTracker)

option( BUILD_EXAMPLES "Build Tools" ON )
# add_subdirectory(examples)

# Add files from within the project
# add_library(${PROJECT_NAME}_LIB
#         src/FeatureTracker.cpp
# )

# install(TARGETS ${PROJECT_NAME}_LIB DESTINATION lib)
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake share/${PROJECT_NAME}/cmake)







if(BUILD_EXAMPLES)
    message("Building the examples...")
    add_subdirectory(examples)
endif()