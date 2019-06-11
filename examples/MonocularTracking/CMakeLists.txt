# include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(GeneralFeatureTracker)
find_package(OpenCV 3 REQUIRED)

add_executable(MonocularTracking main.cpp)

target_include_directories(MonocularTracking PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(MonocularTracking GeneralFeatureTracker ${OpenCV_LIBS})