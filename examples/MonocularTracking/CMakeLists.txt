# include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(GeneralFeatureTracker)

add_executable(MonocularTracking main.cpp)

target_link_libraries(MonocularTracking GeneralFeatureTracker)